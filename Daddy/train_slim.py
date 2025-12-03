"""
Multi-env DQN training loop for the slim hierarchical agent.

Follows the contracts in DESIGN_SLIM_MODEL_V2.md and REWARD_LOGGING_SPEC.md.
"""

from __future__ import annotations

import argparse
import atexit
import random
import csv
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from .env_utils import SafeRedEnv, SimpleVectorEnv
from .agent import AgentConfig, SlimHierarchicalDQN
from .flags import FlagEncoder
from .logging_utils import PerformanceTracker, RewardLogger
from .curriculum import CurriculumManager
from .video_utils import maybe_save_video
from pokemonred_puffer.environment import RedGymEnv


def _cleanup_shared_memory():
    """Close/unlink the global shared memory used by RedGymEnv to silence resource_tracker warnings."""
    try:
        RedGymEnv.env_id.close()
        RedGymEnv.env_id.unlink()
    except Exception:
        pass


atexit.register(_cleanup_shared_memory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the slim hierarchical DQN agent.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=2_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default="runs/slim")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rnd-weight", type=float, default=0.1)
    parser.add_argument("--novelty-weight", type=float, default=0.05)
    parser.add_argument("--bayes-weight", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint for eval.")
    parser.add_argument("--save-state-dir", type=str, default="Daddy/savestates", help="Where to save new .state files.")
    parser.add_argument("--save-state-every", type=int, default=0, help="Save a savestate every N episodes per env (0 to disable).")
    parser.add_argument("--save-state-on-bayes", action="store_true", help="Save a savestate when Bayes milestones update.")
    parser.add_argument("--use-puffer", action="store_true", help="Use pufferlib vectorization (optional).")
    parser.add_argument("--log-interval", type=int, default=500, help="Steps between console log prints.")
    parser.add_argument("--record-video-every", dest="record_video_every", type=int, default=0, help="Record an mp4 for env0 every N episodes (0 to disable).")
    parser.add_argument("--video-dir", dest="video_dir", type=str, default="runs/videos", help="Directory to store recorded videos.")
    # model size overrides
    parser.add_argument("--cnn-channels", type=str, default="", help="Comma-separated CNN channels, e.g., 32,64,64")
    parser.add_argument("--gru-size", type=int, default=128)
    parser.add_argument("--lstm-size", type=int, default=128)
    parser.add_argument("--ssm-size", type=int, default=128)
    parser.add_argument("--structured-hidden", type=int, default=128)
    parser.add_argument("--headless", action="store_true", help="Force headless emulator")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_env_config(headless: bool) -> Dict:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "pokemonred_puffer" / "config.yaml"
    config = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    env_cfg["headless"] = headless
    env_cfg["save_video"] = False
    # Ensure spatial dims stay reasonable for the CNN (avoid two_bit shrink).
    env_cfg["two_bit"] = False
    # Resolve asset locations (ROM + savestates). Prefer archive consolidation.
    assets_root = repo_root / "archive" / "pokemonred_puffer_assets"
    rom_path = assets_root / "red.gb"
    state_dir = assets_root / "pyboy_states"
    if not rom_path.exists():
        # fallback to legacy location
        rom_path = repo_root / "pokemonred_puffer" / "red.gb"
    if not state_dir.exists():
        state_dir = repo_root / "pokemonred_puffer" / "pyboy_states"
    # Always override so training uses the consolidated assets, not stale defaults.
    env_cfg["gb_path"] = str(rom_path)
    env_cfg["state_dir"] = str(state_dir)
    # Default to the richer Rubinstein reward (blended with our intrinsic/Bayes heads downstream).
    env_cfg["use_rubinstein_reward"] = True
    # Prefer configured init_state if it exists; otherwise fall back to first available,
    # otherwise SafeRedEnv will create a fresh state on first reset.
    init_state = env_cfg.get("init_state", "Bulbasaur")
    init_state_file = state_dir / f"{init_state}.state"
    if not init_state_file.exists():
        candidates = sorted(state_dir.glob("*.state"))
        if candidates:
            env_cfg["init_state"] = candidates[0].stem
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found at {rom_path}. Place red.gb there before training.")
    return env_cfg  # plain dict for OmegaConf.create later


def make_env(env_cfg: Dict, seed: int):
    cfg = OmegaConf.create(env_cfg)
    env = SafeRedEnv(cfg)
    env.reset(seed=seed)
    return env


def obs_to_dict(obs) -> Dict[str, np.ndarray]:
    """
    Normalize observations to a dict.
    - Dict stays as-is.
    - Structured ndarray (with dtype.names) becomes dict of fields.
    - Plain ndarray is assumed to be the screen tensor -> {"screen": obs}
    """
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
        # frames already (H, W, 1); stack -> (T, H, W)
        arr = np.stack(list(frames), axis=0)
        if arr.ndim == 4:
            arr = arr[..., 0]
        stacked.append(arr)
    return np.stack(stacked, axis=0)


def to_torch_batch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_config(headless=args.headless)

    # curriculum setup
    state_dir = Path(env_cfg["state_dir"])
    initial_states = sorted(state_dir.glob("*.state"))
    save_state_dir = Path(args.save_state_dir)
    save_state_dir.mkdir(parents=True, exist_ok=True)
    curriculum = CurriculumManager(savestates=initial_states)

    def env_fn(idx: int):
        return make_env(env_cfg, seed=args.seed + idx)

    envs = SimpleVectorEnv([lambda idx=i: env_fn(idx) for i in range(args.num_envs)])

    obs, _ = envs.reset(seed=args.seed)
    obs = obs_to_dict(obs)
    num_actions = envs.single_action_space.n

    # Frame stacks per environment
    frame_stacks: List[Deque[np.ndarray]] = [deque(maxlen=args.frame_stack) for _ in range(args.num_envs)]
    for i in range(args.num_envs):
        for _ in range(args.frame_stack):
            frame_stacks[i].append(obs["screen"][i])

    flag_encoder = FlagEncoder()
    structured_dim = flag_encoder.output_dim

    agent_cfg = AgentConfig(
        num_actions=num_actions,
        frame_stack=args.frame_stack,
        structured_dim=structured_dim,
        replay_capacity=args.buffer_size,
        target_update_interval=args.target_update,
        rnd_weight=args.rnd_weight,
        novelty_weight=args.novelty_weight,
        bayes_weight=args.bayes_weight,
    )
    # apply model size overrides if provided
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
    reward_logger = RewardLogger(log_dir=log_dir)
    perf = PerformanceTracker()
    # Per-step coarse state logging for downstream visuals (map frequency, entropy timelines).
    positions_path = log_dir / "positions.csv"
    positions_file = open(positions_path, "w", newline="")
    atexit.register(positions_file.close)
    pos_writer = csv.DictWriter(
        positions_file,
        fieldnames=[
            "step_global",
            "env",
            "episode",
            "map_id",
            "coord",
            "badge",
            "deaths",
            "reward",
            "entropy",
        ],
    )
    pos_writer.writeheader()
    if args.mode == "eval" and args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0

    # video buffer (only env 0 to limit size)
    frames_for_video: List[np.ndarray] = []

    states = agent.reset_hidden(args.num_envs)
    dones = np.zeros(args.num_envs, dtype=bool)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_rnd = np.zeros(args.num_envs, dtype=np.float32)
    episode_novel = np.zeros(args.num_envs, dtype=np.float32)
    episode_bayes = np.zeros(args.num_envs, dtype=np.float32)
    episode_total = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_ids = np.zeros(args.num_envs, dtype=np.int64)
    agent.epsilon_decay = (1.0 - agent.epsilon_min) / max(1, args.total_steps)

    for step in range(args.total_steps):
        frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        obs_tensors = to_torch_batch(obs, device=device)
        structured, milestone_flags = agent.flag_encoder(obs_tensors)
        done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)

        actions, new_states, aux = agent.act(frame_batch, structured, states, done_mask)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        next_obs = obs_to_dict(next_obs)
        dones = np.logical_or(terminated, truncated)
        done_flags = dones.copy()
        episode_returns += rewards
        episode_lengths += 1

        intrinsic = agent.intrinsic_rewards(aux, milestone_flags, env_ids=range(args.num_envs))
        r_env = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        total_reward = (
            r_env
            + args.rnd_weight * intrinsic["rnd"]
            + args.novelty_weight * intrinsic["novel"]
            + args.bayes_weight * intrinsic["bayes"]
        )
        bayes_updates: Dict[int, Dict[str, Dict[str, float]]] = {}
        rnd_np = intrinsic["rnd"].detach().cpu().numpy()
        novel_np = intrinsic["novel"].detach().cpu().numpy()
        bayes_np = intrinsic["bayes"].detach().cpu().numpy()
        total_np = total_reward.detach().cpu().numpy()
        episode_rnd += rnd_np
        episode_novel += novel_np
        episode_bayes += bayes_np
        episode_total += total_np

        reset_obs_batch = None
        # per-env done handling
        reset_obs_batch = None

        full_reset = False
        for env_idx, done in enumerate(done_flags):
            if done:
                perf.episode()
                bayes_updates[env_idx] = agent.update_bayes_from_milestones(
                    milestone_flags[env_idx].detach().cpu()
                )
                reward_logger.log_episode(
                    {
                        "episode_id": int(episode_ids[env_idx]),
                        "env_id": env_idx,
                        "length_steps": int(episode_lengths[env_idx]),
                        "return_env": float(episode_returns[env_idx]),
                        "return_rnd": float(episode_rnd[env_idx]),
                        "return_novel": float(episode_novel[env_idx]),
                        "return_bayes": float(episode_bayes[env_idx]),
                        "return_total_with_intrinsic": float(episode_total[env_idx]),
                    }
                )
                episode_ids[env_idx] += 1
                episode_returns[env_idx] = 0.0
                episode_rnd[env_idx] = 0.0
                episode_novel[env_idx] = 0.0
                episode_bayes[env_idx] = 0.0
                episode_total[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                agent.reset_episode(env_idx)
                # select next init savestate if available
                maybe_state = curriculum.sample_state()
                env = envs.envs[env_idx]
                if maybe_state and maybe_state.exists():
                    env.init_state_name = maybe_state.stem
                    env.init_state_path = Path(env.state_dir) / f"{env.init_state_name}.state"
                # save new savestates if requested
                if args.save_state_every and (episode_ids[env_idx] % args.save_state_every == 0):
                    save_path = save_state_dir / f"ep{episode_ids[env_idx]:05d}_env{env_idx}.state"
                    env._save_emulator_state(save_path)
                    curriculum.add_state(save_path)
                # record video if requested (env 0 only)
                if args.record_video_every and env_idx == 0 and episode_ids[env_idx] % args.record_video_every == 0:
                    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                    out_path = Path(args.video_dir) / f"train_ep{episode_ids[env_idx]:05d}.mp4"
                    if frames_for_video:
                        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
                        for art in artifacts:
                            print(f"[video] saved {art['path']} ({art['frames']} frames)")
                    frames_for_video.clear()
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

            # Log coarse positional / stat info for visuals
            info = infos[env_idx] if isinstance(infos, (list, tuple)) else {}
            stats = info.get("stats", {}) if isinstance(info, dict) else {}
            hist = stats.get("action_hist", None)
            entropy = None
            if hist is not None:
                hist = np.array(hist, dtype=np.float32)
                total = hist.sum()
                if total > 0:
                    p = hist / total
                    entropy = float(-(p * np.log(p + 1e-8)).sum())
            pos_writer.writerow(
                {
                    "step_global": step,
                    "env": env_idx,
                    "episode": int(episode_ids[env_idx]),
                    "map_id": stats.get("map_id", 0),
                    "coord": stats.get("coord", 0),
                    "badge": stats.get("badge", 0),
                    "deaths": stats.get("deaths", 0),
                    "reward": float(rewards[env_idx]),
                    "entropy": entropy if entropy is not None else "",
                }
            )
            positions_file.flush()

        if full_reset:
            obs, _ = envs.reset(seed=args.seed + step + 1)
            obs = obs_to_dict(obs)
            frame_stacks = [deque(maxlen=args.frame_stack) for _ in range(args.num_envs)]
            for i in range(args.num_envs):
                for _ in range(args.frame_stack):
                    frame_stacks[i].append(obs["screen"][i])
            states = agent.reset_hidden(args.num_envs)
            dones = np.zeros(args.num_envs, dtype=bool)
            continue

        next_frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        next_obs_tensors = to_torch_batch(next_obs, device=device)
        next_structured, _ = agent.flag_encoder(next_obs_tensors)

        if args.mode == "train":
            agent.add_transition(
                frame_batch,
                structured,
                actions,
                total_reward,
                {"env": r_env, "rnd": intrinsic["rnd"], "novel": intrinsic["novel"], "bayes": intrinsic["bayes"]},
                torch.as_tensor(done_flags, device=device, dtype=torch.float32),
                next_frame_batch,
                next_structured,
                states,
            )
            if step >= args.learning_starts and step % args.train_every == 0:
                metrics = agent.learn(args.batch_size)
                if metrics and step % max(1, args.log_interval) == 0:
                    loss = metrics.get("loss", 0.0)
                    rnd_loss = metrics.get("rnd_loss", 0.0)
                    print(f"step {step}: loss={loss:.4f} rnd_loss={rnd_loss:.4f}")

        states = new_states
        obs = next_obs

        for env_idx in range(args.num_envs):
            reward_logger.log_step(
                step_global=step,
                env_id=env_idx,
                episode_id=int(episode_ids[env_idx]),
                rewards={
                    "env": rewards[env_idx],
                    "rnd": intrinsic["rnd"][env_idx],
                    "novel": intrinsic["novel"][env_idx],
                    "bayes": intrinsic["bayes"][env_idx],
                    "total": total_reward[env_idx],
                },
                weights={"rnd": args.rnd_weight, "novel": args.novelty_weight, "bayes": args.bayes_weight},
                milestone_flags=milestone_flags[env_idx].detach().cpu().tolist(),
                map_id=int(obs["map_id"][env_idx]) if "map_id" in obs else None,
            )
            if env_idx in bayes_updates:
                for name, stats in bayes_updates[env_idx].items():
                    reward_logger.log_progress(name, stats["alpha"], stats["beta"], stats["alarm_score"])
                    # optional: save savestate on Bayes update
                    if args.save_state_on_bayes:
                        env = envs.envs[env_idx]
                        save_path = save_state_dir / f"bayes_ep{episode_ids[env_idx]:05d}_env{env_idx}_{name}.state"
                        env._save_emulator_state(save_path)
                        curriculum.add_state(save_path)

            # collect frames for optional video (env 0 only)
            if args.record_video_every and env_idx == 0:
                frame_rgb = np.repeat(obs["screen"][env_idx], repeats=3, axis=-1)
                frames_for_video.append(frame_rgb)

        perf.step(args.num_envs)
        if step % max(1, args.log_interval) == 0 and step > 0:
            stats = perf.stats()
            print(f"step {step} | SPS {stats['sps']:.1f} | episodes/hr {stats['episodes_per_hour']:.1f}")

    reward_logger.close()
    envs.close()


if __name__ == "__main__":
    main()
