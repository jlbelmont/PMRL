"""
Multi-env DQN training loop for the slim hierarchical agent.

Follows the contracts in DESIGN_SLIM_MODEL_V2.md and REWARD_LOGGING_SPEC.md.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import pufferlib
    import pufferlib.emulation
    import pufferlib.vector
except ImportError:  # pragma: no cover - optional dependency
    pufferlib = None  # type: ignore[assignment]

from pokemonred_puffer.environment import RedGymEnv

from .agent import AgentConfig, SlimHierarchicalDQN
from .flags import FlagEncoder
from .logging_utils import PerformanceTracker, RewardLogger


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
    return env_cfg  # plain dict for OmegaConf.create later


def make_env(env_cfg: Dict, seed: int, wrap_with_puffer: bool = False):
    cfg = OmegaConf.create(env_cfg)
    env = RedGymEnv(cfg)
    env.reset(seed=seed)
    if wrap_with_puffer and pufferlib is not None:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env)  # type: ignore[attr-defined]
    return env


def make_puffer_env_creator(env_cfg: Dict) -> callable:
    def env_creator():
        cfg = OmegaConf.create(env_cfg)
        env = RedGymEnv(cfg)
        return pufferlib.emulation.GymnasiumPufferEnv(env=env)  # type: ignore[attr-defined]

    return env_creator


def obs_to_dict(obs) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, np.ndarray) and obs.dtype.names:
        return {k: obs[k] for k in obs.dtype.names}
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

    envs = None
    driver = "gym"
    if pufferlib is not None:
        try:
            env_creator = make_puffer_env_creator(env_cfg)
            envs = pufferlib.vector.make(env_creator, num_envs=args.num_envs)  # type: ignore[attr-defined]
            driver = "pufferlib"
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[train_slim] pufferlib vector make failed ({exc}), falling back to gym SyncVectorEnv")
            envs = None

    if envs is None:
        def env_fn(idx: int):
            return make_env(env_cfg, seed=args.seed + idx, wrap_with_puffer=False)

        envs = gym.vector.SyncVectorEnv([lambda idx=i: env_fn(idx) for i in range(args.num_envs)])
        driver = "gym"

    obs, _ = envs.reset(seed=args.seed)
    obs = obs_to_dict(obs)
    if driver == "pufferlib":
        try:
            num_actions = envs.single_action_space.n
        except AttributeError:
            num_actions = envs.action_space.n
    else:
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
    agent = SlimHierarchicalDQN(agent_cfg, device=device)
    agent.flag_encoder = flag_encoder
    reward_logger = RewardLogger(log_dir=log_dir)
    perf = PerformanceTracker()

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
        if np.any(done_flags) and hasattr(envs, "reset_done"):
            try:
                reset_obs_batch, _ = envs.reset_done(done_flags)
                reset_obs_batch = obs_to_dict(reset_obs_batch)
            except Exception:
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
                if reset_obs_batch is not None:
                    for key in next_obs:
                        next_obs[key][env_idx] = reset_obs_batch[key][env_idx]
                    frame_stacks[env_idx].clear()
                    for _ in range(args.frame_stack):
                        frame_stacks[env_idx].append(reset_obs_batch["screen"][env_idx])
                    new_states[env_idx] = agent.network.initial_state(batch_size=1, device=device)
                    dones[env_idx] = False
                elif driver == "gym":
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
                    full_reset = True
            else:
                frame_stacks[env_idx].append(next_obs["screen"][env_idx])

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
        states = new_states
        obs = next_obs

        if step >= args.learning_starts and step % args.train_every == 0:
            metrics = agent.learn(args.batch_size)
            if metrics:
                loss = metrics.get("loss", 0.0)
                rnd_loss = metrics.get("rnd_loss", 0.0)
                print(f"step {step}: loss={loss:.4f} rnd_loss={rnd_loss:.4f}")

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

        perf.step(args.num_envs)
        if step % 500 == 0 and step > 0:
            stats = perf.stats()
            print(f"step {step} | SPS {stats['sps']:.1f} | episodes/hr {stats['episodes_per_hour']:.1f}")

    reward_logger.close()
    envs.close()


if __name__ == "__main__":
    main()
