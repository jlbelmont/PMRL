#!/usr/bin/env python3
"""
Actor-only runner for distributed training across machines.

This script runs actors that pull weights from a shared directory synchronized
via rsync. This enables:

  1. Training on a cluster (SLURM) with the main jobs producing weights
  2. Syncing weights to your Mac via rsync (using sync_weights.sh)
  3. Running additional actors on your Mac that contribute training

Usage:
    # On Mac (actor-only mode):
    python -m Daddy.run_actors_only \
        --shared-dir ~/shared_weights \
        --user-id $(hostname) \
        --num-envs auto

    # With autotune to detect optimal settings:
    python -m Daddy.run_actors_only \
        --shared-dir ~/shared_weights \
        --user-id mac_actor \
        --num-envs auto \
        --autotune

The actors will:
  - Pull the latest weights from --shared-dir
  - Run environments and collect experience
  - Apply Polyak averaging when merging weights
  - Optionally contribute weights back to shared_dir
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .env_utils import SafeRedEnv, SimpleVectorEnv
from .agent import AgentConfig, SlimHierarchicalDQN
from .flags import FlagEncoder
from .logging_utils import PerformanceTracker, RewardLogger
from .curriculum import CurriculumManager
from .streaming import StreamClient, BestAgentSelector, progress_score


# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[actors] Shutdown signal received...")


def _detect_hardware() -> Dict[str, any]:
    """Detect available hardware resources."""
    import psutil
    
    info = {
        "cpu_count": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["gpu_available"]:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
    
    return info


def _recommend_num_envs(hardware: Dict) -> int:
    """Recommend number of environments based on hardware."""
    # Conservative defaults for laptop
    cpu_cores = hardware.get("cpu_count", 2)
    ram_gb = hardware.get("ram_gb", 8)
    
    # Each env uses ~0.5-1GB RAM and some CPU
    # Leave headroom for OS and other apps
    max_by_cpu = max(1, cpu_cores - 1)
    max_by_ram = max(1, int((ram_gb - 4) / 1.0))  # Reserve 4GB for system
    
    recommended = min(max_by_cpu, max_by_ram, 8)  # Cap at 8 for laptops
    return max(1, recommended)


def _load_profile(user_id: str) -> Optional[Dict]:
    """Load autotune profile if it exists."""
    profile_dir = Path.home() / ".pokemon_red_rl"
    profile_path = profile_dir / f"{user_id}.json"
    
    if not profile_path.exists():
        return None
    
    try:
        with open(profile_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_profile(user_id: str, profile: Dict) -> Path:
    """Save autotune profile."""
    profile_dir = Path.home() / ".pokemon_red_rl"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{user_id}.json"
    
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    
    return profile_path


class RemoteWeightFetcher:
    """
    Watches a shared directory for weight updates and loads them into the agent.
    Performs Polyak averaging when configured.
    """
    
    def __init__(
        self,
        shared_dir: str,
        user_id: str,
        polyak_alpha: float = 0.5,
        enable_polyak: bool = True,
    ):
        self.shared_dir = Path(shared_dir).expanduser()
        self.user_id = user_id
        self.polyak_alpha = polyak_alpha
        self.enable_polyak = enable_polyak
        self.last_seen_mtime = 0.0
        self.current_state: Optional[Dict] = None
    
    def _find_latest_weight_file(self) -> Optional[Path]:
        """Find the most recent .pt file in shared_dir."""
        if not self.shared_dir.exists():
            return None
        
        best_path = None
        best_mtime = self.last_seen_mtime
        
        for entry in self.shared_dir.iterdir():
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue
            if entry.name.startswith(".tmp_"):
                continue
            
            try:
                mtime = entry.stat().st_mtime
                if mtime <= self.last_seen_mtime:
                    continue
                
                # Skip files authored by us
                payload = torch.load(entry, map_location="cpu", weights_only=False)
                author = payload.get("user", "")
                if author == self.user_id:
                    continue
                
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_path = entry
            except Exception:
                continue
        
        return best_path
    
    def _polyak_merge(self, local_state: Dict, remote_state: Dict) -> Dict:
        """Polyak-average local and remote states."""
        merged = {}
        alpha = self.polyak_alpha
        
        for name, local_tensor in local_state.items():
            remote_tensor = remote_state.get(name)
            if remote_tensor is not None and remote_tensor.shape == local_tensor.shape:
                merged[name] = local_tensor * (1.0 - alpha) + remote_tensor.to(local_tensor.dtype) * alpha
            else:
                merged[name] = local_tensor
        
        return merged
    
    def sync_to_agent(self, agent: SlimHierarchicalDQN) -> bool:
        """
        Check for new weights and update agent if found.
        Returns True if weights were updated.
        """
        latest_path = self._find_latest_weight_file()
        if latest_path is None:
            return False
        
        try:
            payload = torch.load(latest_path, map_location="cpu", weights_only=False)
            remote_state = payload.get("state") or payload.get("network_state_dict")
            
            if remote_state is None:
                return False
            
            self.last_seen_mtime = latest_path.stat().st_mtime
            
            # Get current agent state
            local_state = agent.network.state_dict()
            
            # Apply Polyak averaging if enabled
            if self.enable_polyak and self.current_state is not None:
                final_state = self._polyak_merge(local_state, remote_state)
            else:
                final_state = remote_state
            
            self.current_state = final_state
            
            # Load into agent
            agent.network.load_state_dict(final_state, strict=False)
            agent.target_network.load_state_dict(final_state, strict=False)
            
            print(f"[weight_sync] loaded weights from {latest_path.name}", flush=True)
            return True
        
        except Exception as e:
            print(f"[weight_sync] error: {e}", flush=True)
            return False
    
    def push_weights(self, agent: SlimHierarchicalDQN, step: int) -> bool:
        """Push current weights to shared directory."""
        if not self.shared_dir.exists():
            self.shared_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            payload = {
                "state": agent.network.state_dict(),
                "user": self.user_id,
                "step": step,
                "timestamp": time.time(),
            }
            
            filename = f"{self.user_id}_{step:08d}.pt"
            tmp_path = self.shared_dir / f".tmp_{filename}"
            final_path = self.shared_dir / filename
            
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)
            
            print(f"[weight_sync] pushed weights: {filename}", flush=True)
            return True
        
        except Exception as e:
            print(f"[weight_sync] push error: {e}", flush=True)
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run actors only (no learner) - pulls weights from shared directory."
    )
    # Required
    parser.add_argument("--shared-dir", required=True, help="Path to shared weights directory (synced via rsync).")
    
    # Identity
    parser.add_argument("--user-id", default=None, help="Unique identifier for this machine (default: hostname).")
    
    # Environment settings
    parser.add_argument("--num-envs", default="auto", help="Number of environments ('auto' to detect).")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default: True).")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    
    # Weight sync settings
    parser.add_argument("--sync-interval", type=int, default=1000, help="Steps between weight sync checks.")
    parser.add_argument("--polyak-alpha", type=float, default=0.5, help="Polyak averaging coefficient.")
    parser.add_argument("--no-polyak", action="store_true", help="Disable Polyak averaging.")
    parser.add_argument("--contribute", action="store_true", help="Push weights back to shared_dir.")
    parser.add_argument("--contribute-interval", type=int, default=10000, help="Steps between weight pushes.")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="runs/actors_only")
    parser.add_argument("--log-interval", type=int, default=500)
    
    # Autotune
    parser.add_argument("--autotune", action="store_true", help="Run autotune to detect optimal settings.")
    
    # Streaming (optional)
    parser.add_argument("--stream", action="store_true", help="Enable streaming to laptop viewer.")
    parser.add_argument("--stream-port", type=int, default=9999)
    parser.add_argument("--stream-interval", type=int, default=60)
    
    # Model architecture (should match cluster)
    parser.add_argument("--cnn-channels", type=str, default="32,64,64")
    parser.add_argument("--gru-size", type=int, default=128)
    parser.add_argument("--lstm-size", type=int, default=128)
    parser.add_argument("--ssm-size", type=int, default=128)
    parser.add_argument("--structured-hidden", type=int, default=128)
    
    # Device
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if not specified).")
    
    return parser.parse_args()


def load_env_config(headless: bool) -> Dict:
    """Load environment config from pokemonred_puffer."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "pokemonred_puffer" / "config.yaml"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    
    config = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    env_cfg["headless"] = headless
    env_cfg["save_video"] = False
    env_cfg["two_bit"] = False
    
    # Resolve asset locations
    assets_root = repo_root / "archive" / "pokemonred_puffer_assets"
    rom_path = assets_root / "red.gb"
    state_dir = assets_root / "pyboy_states"
    
    if not rom_path.exists():
        rom_path = repo_root / "pokemonred_puffer" / "red.gb"
    if not state_dir.exists():
        state_dir = repo_root / "pokemonred_puffer" / "pyboy_states"
    
    env_cfg["gb_path"] = str(rom_path)
    env_cfg["state_dir"] = str(state_dir)
    env_cfg["use_rubinstein_reward"] = True
    
    # Find init state
    init_state = env_cfg.get("init_state", "Bulbasaur")
    init_state_file = state_dir / f"{init_state}.state"
    if not init_state_file.exists():
        candidates = sorted(state_dir.glob("*.state"))
        if candidates:
            env_cfg["init_state"] = candidates[0].stem
    
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path}")
    
    return env_cfg


def make_env(env_cfg: Dict, seed: int):
    cfg = OmegaConf.create(env_cfg)
    env = SafeRedEnv(cfg)
    env.reset(seed=seed)
    return env


def obs_to_dict(obs) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, np.ndarray):
        if obs.dtype.names:
            return {k: obs[k] for k in obs.dtype.names}
        return {"screen": obs}
    raise TypeError(f"Unsupported observation type: {type(obs)}")


def stack_frames(frame_stacks: List[Deque[np.ndarray]]) -> np.ndarray:
    stacked = []
    for frames in frame_stacks:
        arr = np.stack(list(frames), axis=0)
        if arr.ndim == 4:
            arr = arr[..., 0]
        stacked.append(arr)
    return np.stack(stacked, axis=0)


def to_torch_batch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def main() -> None:
    global _shutdown_requested
    
    args = parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Determine user ID
    if args.user_id is None:
        import socket
        args.user_id = socket.gethostname().replace(".", "_")
    
    print(f"[actors] User ID: {args.user_id}")
    print(f"[actors] Shared weights dir: {args.shared_dir}")
    
    # Detect hardware
    hardware = _detect_hardware()
    print(f"[actors] Hardware: {hardware['cpu_count']} cores, {hardware['ram_gb']:.1f}GB RAM")
    if hardware["gpu_available"]:
        print(f"[actors] GPU: {hardware['gpu_name']} ({hardware['gpu_memory_gb']:.1f}GB)")
    elif hardware["mps_available"]:
        print("[actors] MPS (Apple Silicon) available")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif hardware["gpu_available"]:
        device = torch.device("cuda")
    elif hardware["mps_available"]:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[actors] Using device: {device}")
    
    # Load or create profile
    profile = _load_profile(args.user_id)
    if args.autotune or profile is None:
        print("[actors] Running autotune...")
        profile = {
            "hardware": hardware,
            "recommended_num_envs": _recommend_num_envs(hardware),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        profile_path = _save_profile(args.user_id, profile)
        print(f"[actors] Profile saved: {profile_path}")
    
    # Determine num_envs
    if args.num_envs == "auto":
        num_envs = profile.get("recommended_num_envs", 2)
    else:
        num_envs = int(args.num_envs)
    print(f"[actors] Running {num_envs} environments")
    
    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load environment config
    env_cfg = load_env_config(headless=args.headless)
    
    # Create environments
    def env_fn(idx: int):
        return make_env(env_cfg, seed=args.seed + idx)
    
    envs = SimpleVectorEnv([lambda idx=i: env_fn(idx) for i in range(num_envs)])
    
    obs, _ = envs.reset(seed=args.seed)
    obs = obs_to_dict(obs)
    num_actions = envs.single_action_space.n
    
    # Frame stacks per environment
    frame_stacks: List[Deque[np.ndarray]] = [deque(maxlen=args.frame_stack) for _ in range(num_envs)]
    for i in range(num_envs):
        for _ in range(args.frame_stack):
            frame_stacks[i].append(obs["screen"][i])
    
    # Create agent
    flag_encoder = FlagEncoder()
    structured_dim = flag_encoder.output_dim
    
    agent_cfg = AgentConfig(
        num_actions=num_actions,
        frame_stack=args.frame_stack,
        structured_dim=structured_dim,
        replay_capacity=10000,  # Smaller for actor-only mode
        target_update_interval=2000,
    )
    
    cnn_channels = tuple(int(x) for x in args.cnn_channels.split(",")) if args.cnn_channels else None
    agent = SlimHierarchicalDQN(
        agent_cfg,
        device=device,
        cnn_channels=cnn_channels,
        gru_size=args.gru_size,
        lstm_size=args.lstm_size,
        ssm_size=args.ssm_size,
        structured_hidden=args.structured_hidden,
    )
    agent.flag_encoder = flag_encoder
    agent.epsilon = 0.1  # Low epsilon for actor-only mode (mostly exploit learned policy)
    agent.epsilon_min = 0.05
    
    # Setup weight fetcher
    weight_fetcher = RemoteWeightFetcher(
        shared_dir=args.shared_dir,
        user_id=args.user_id,
        polyak_alpha=args.polyak_alpha,
        enable_polyak=not args.no_polyak,
    )
    
    # Initial weight sync
    print("[actors] Waiting for initial weights...")
    for _ in range(30):  # Try for up to 30 seconds
        if weight_fetcher.sync_to_agent(agent):
            break
        if _shutdown_requested:
            envs.close()
            return
        time.sleep(1.0)
    else:
        print("[actors] Warning: No weights found, starting with random initialization")
    
    # Setup streaming if enabled
    stream_client: Optional[StreamClient] = None
    best_selector: Optional[BestAgentSelector] = None
    
    if args.stream:
        print(f"[actors] Streaming enabled on port {args.stream_port}")
        stream_client = StreamClient(
            port=args.stream_port,
            interval_steps=args.stream_interval,
            metadata={"user_id": args.user_id},
        )
        best_selector = BestAgentSelector(
            candidates_path=str(log_dir / "stream_candidates.json"),
            top_k=2,
        )
    
    # Initialize tracking
    perf = PerformanceTracker()
    reward_logger = RewardLogger(log_dir=log_dir)
    
    states = agent.reset_hidden(num_envs)
    dones = np.zeros(num_envs, dtype=bool)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_total = np.zeros(num_envs, dtype=np.float32)
    episode_ids = np.zeros(num_envs, dtype=np.int64)
    
    print("[actors] Starting actor loop...")
    step = 0
    last_sync_step = 0
    last_contribute_step = 0
    
    try:
        while not _shutdown_requested:
            # Periodic weight sync
            if step - last_sync_step >= args.sync_interval:
                weight_fetcher.sync_to_agent(agent)
                last_sync_step = step
            
            # Periodic weight contribution
            if args.contribute and step - last_contribute_step >= args.contribute_interval:
                weight_fetcher.push_weights(agent, step)
                last_contribute_step = step
            
            # Run one step
            frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
            obs_tensors = to_torch_batch(obs, device=device)
            structured, milestone_flags = agent.flag_encoder(obs_tensors)
            done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)
            
            actions, new_states, aux = agent.act(frame_batch, structured, states, done_mask)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
            next_obs = obs_to_dict(next_obs)
            dones = np.logical_or(terminated, truncated)
            
            episode_returns += rewards
            episode_total += rewards
            
            # Handle episode ends
            for env_idx, done in enumerate(dones):
                if done:
                    perf.episode()
                    reward_logger.log_episode({
                        "episode_id": int(episode_ids[env_idx]),
                        "env_id": env_idx,
                        "return_env": float(episode_returns[env_idx]),
                    })
                    episode_ids[env_idx] += 1
                    episode_returns[env_idx] = 0.0
                    episode_total[env_idx] = 0.0
                    
                    # Reset this env
                    reset_obs, _ = envs.envs[env_idx].reset(seed=args.seed + env_idx)
                    reset_obs = obs_to_dict(reset_obs)
                    for key in next_obs:
                        next_obs[key][env_idx] = reset_obs[key]
                    frame_stacks[env_idx].clear()
                    for _ in range(args.frame_stack):
                        frame_stacks[env_idx].append(reset_obs["screen"])
                    new_states[env_idx] = agent.network.initial_state(batch_size=1, device=device)
                    dones[env_idx] = False
                else:
                    frame_stacks[env_idx].append(next_obs["screen"][env_idx])
                
                # Streaming
                if stream_client is not None and best_selector is not None:
                    info = infos[env_idx] if isinstance(infos, (list, tuple)) else {}
                    stream_info = {"badge_count": 0, "story_flags": {}}
                    if isinstance(info, dict):
                        stats = info.get("stats", {})
                        if isinstance(stats, dict):
                            stream_info["badge_count"] = stats.get("badge", 0)
                    
                    agent_id = f"{args.user_id}_env{env_idx}"
                    should_stream = best_selector.update(
                        agent_id=agent_id,
                        info=stream_info,
                        episode_reward=float(episode_total[env_idx]),
                        env_idx=env_idx,
                    )
                    
                    if should_stream:
                        frame = obs["screen"][env_idx]
                        if frame.ndim == 2:
                            frame = np.expand_dims(frame, -1)
                        if frame.shape[-1] == 1:
                            frame = np.repeat(frame, 3, axis=-1)
                        
                        score, _ = progress_score(stream_info, float(episode_total[env_idx]))
                        stream_client.maybe_send(
                            frame=frame,
                            step=step,
                            agent_id=agent_id,
                            extra_meta={"score": score, "episode_reward": float(episode_total[env_idx])},
                        )
            
            states = new_states
            obs = next_obs
            
            perf.step(num_envs)
            step += 1
            
            if step % args.log_interval == 0:
                stats = perf.stats()
                print(f"[actors] step {step} | SPS {stats['sps']:.1f} | episodes/hr {stats['episodes_per_hour']:.1f}")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("[actors] Shutting down...")
        if stream_client:
            stream_client.close()
        reward_logger.close()
        envs.close()
        print("[actors] Done.")


if __name__ == "__main__":
    main()

