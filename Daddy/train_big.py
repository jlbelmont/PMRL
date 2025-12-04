"""
Scaled-up training entrypoint for the hierarchical DQN agent.

Defaults to a “large” model preset and higher parallelism for swarming runs.
Follows logging requirements in REWARD_LOGGING_SPEC.md with rich terminal feedback.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from .agent import AgentConfig, SlimHierarchicalDQN
from .env_utils import SafeRedEnv, SimpleVectorEnv
from .flags import FlagEncoder
from .logging_utils import PerformanceTracker, RewardLogger
from .curriculum import CurriculumManager
from .video_utils import maybe_save_video
from pokemonred_puffer.environment import RedGymEnv


def _cleanup_shared_memory():
    try:
        RedGymEnv.env_id.close()
        RedGymEnv.env_id.unlink()
    except Exception:
        pass


atexit.register(_cleanup_shared_memory)


MODEL_PRESETS = {
    "slim": {"cnn_channels": (16, 32, 64), "gru": 128, "lstm": 128, "ssm": 128, "structured": 128},
    # doubled dimensions for large swarming runs
    "large": {"cnn_channels": (64, 128, 256), "gru": 512, "lstm": 512, "ssm": 512, "structured": 512},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaled training for hierarchical DQN.")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-update", type=int, default=4_000)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-interval", type=int, default=5_000)
    p.add_argument("--eval-interval", type=int, default=0, help="Optional eval every N steps (0 to disable).")
    p.add_argument("--run-name", type=str, default="big_run")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rnd-weight", type=float, default=0.1)
    p.add_argument("--novelty-weight", type=float, default=0.05)
    p.add_argument("--bayes-weight", type=float, default=0.05)
    p.add_argument("--save-state-dir", type=str, default="Daddy/savestates")
    p.add_argument("--save-state-every", type=int, default=0)
    p.add_argument("--save-state-on-bayes", action="store_true")
    p.add_argument("--record-video-every", type=int, default=0)
    p.add_argument("--video-interval", type=int, default=0, help="Alias for record-video-every.")
    p.add_argument("--video-dir", type=str, default="runs/videos")
    p.add_argument("--save-interval", type=int, default=0, help="Save model checkpoint every N steps (0 to disable).")
    p.add_argument(
        "--prune-interval",
        type=int,
        default=0,
        help="If >0, prune mastered savestates every N steps (uses CurriculumManager.promote_or_demote). Off by default.",
    )
    p.add_argument("--model-size", choices=list(MODEL_PRESETS.keys()), default="large")
    p.add_argument("--headless", action="store_true", help="Force headless emulator")
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.set_defaults(headless=True)
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
    env_cfg["two_bit"] = False
    env_cfg["use_rubinstein_reward"] = True
    assets_root = repo_root / "archive" / "pokemonred_puffer_assets"
    rom_path = assets_root / "red.gb"
    state_dir = assets_root / "pyboy_states"
    if not rom_path.exists():
        rom_path = repo_root / "pokemonred_puffer" / "red.gb"
    if not state_dir.exists():
        state_dir = repo_root / "pokemonred_puffer" / "pyboy_states"
    env_cfg["gb_path"] = str(rom_path)
    env_cfg["state_dir"] = str(state_dir)
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
    """
    Convert a list of frame deques into (B, C, H, W) ready for the network.
    Ensures channel-first layout, merges the temporal dimension into channels,
    and guards against tiny spatial sizes by repeating/padding to at least 36x36.
    """
    batches: List[np.ndarray] = []
    for frames in frame_stacks:
        arr = np.stack(list(frames), axis=0)  # (T, ...)

        # Squeeze stray singleton dims that sometimes appear from wrappers
        while arr.ndim > 4 and (arr.shape[1] == 1 or arr.shape[-1] == 1):
            if arr.shape[1] == 1:
                arr = np.squeeze(arr, axis=1)
            elif arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            else:
                break

        # Normalize to channel-first (T, C, H, W)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))  # channel-last -> channel-first
        elif arr.ndim == 3:  # (T, H, W)
            arr = arr[:, None, ...]

        if arr.ndim != 4:
            raise ValueError(f"Unexpected frame shape {arr.shape}")

        _, _, h, w = arr.shape

        # Repeat to reach minimum spatial size for conv kernels
        min_hw = 36
        rep_h = (min_hw + h - 1) // h
        rep_w = (min_hw + w - 1) // w
        if rep_h > 1 or rep_w > 1:
            arr = np.repeat(np.repeat(arr, rep_h, axis=2), rep_w, axis=3)
            h, w = arr.shape[2], arr.shape[3]

        # Pad if still undersized
        pad_h = max(0, min_hw - h)
        pad_w = max(0, min_hw - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        # Merge time into channels
        t, c, h, w = arr.shape
        merged = arr.reshape(t * c, h, w).astype(np.float32)
        # Final safety: ensure conv kernels always have room
        if merged.shape[1] < min_hw or merged.shape[2] < min_hw:
            pad_h = max(0, min_hw - merged.shape[1])
            pad_w = max(0, min_hw - merged.shape[2])
            merged = np.pad(merged, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        batches.append(merged)

    return np.stack(batches, axis=0)  # (B, C, H, W)


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    preset = MODEL_PRESETS[args.model_size]

    # log directory and files
    log_dir = Path(args.log_dir) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    reward_logger = RewardLogger(log_dir=log_dir)
    perf = PerformanceTracker()

    # positions.csv for visuals
    positions_path = log_dir / "positions.csv"
    positions_file = open(positions_path, "w", newline="")
    atexit.register(positions_file.close)
    pos_writer = csv.DictWriter(
        positions_file,
        fieldnames=["step_global", "env", "episode", "map_id", "coord", "badge", "deaths", "reward", "entropy"],
    )
    pos_writer.writeheader()

    env_cfg = load_env_config(headless=args.headless)
    state_dir = Path(env_cfg["state_dir"])
    initial_states = sorted(state_dir.glob("*.state"))
    save_state_dir = Path(args.save_state_dir) / args.run_name
    save_state_dir.mkdir(parents=True, exist_ok=True)
    curriculum = CurriculumManager(savestates=initial_states)

    def env_fn(idx: int):
        return make_env(env_cfg, seed=args.seed + idx)

    envs = SimpleVectorEnv([lambda idx=i: env_fn(idx) for i in range(args.num_envs)])
    obs, _ = envs.reset(seed=args.seed)
    obs = obs_to_dict(obs)

    num_actions = envs.single_action_space.n
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
    agent = SlimHierarchicalDQN(
        agent_cfg,
        device=device,
        cnn_channels=preset["cnn_channels"],
        gru_size=preset["gru"],
        lstm_size=preset["lstm"],
        ssm_size=preset["ssm"],
        structured_hidden=preset["structured"],
    )
    agent.flag_encoder = flag_encoder
    agent.epsilon_decay = (1.0 - agent.epsilon_min) / max(1, args.total_steps)

    states = agent.reset_hidden(args.num_envs)
    dones = np.zeros(args.num_envs, dtype=bool)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_rnd = np.zeros(args.num_envs, dtype=np.float32)
    episode_novel = np.zeros(args.num_envs, dtype=np.float32)
    episode_bayes = np.zeros(args.num_envs, dtype=np.float32)
    episode_total = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_ids = np.zeros(args.num_envs, dtype=np.int64)
    last_badges = np.zeros(args.num_envs, dtype=np.int32)
    last_map = np.full(args.num_envs, -1, dtype=np.int32)

    # rolling stats for terminal logging
    recent_lengths: Deque[float] = deque(maxlen=200)
    recent_env: Deque[float] = deque(maxlen=200)
    recent_total: Deque[float] = deque(maxlen=200)
    recent_r_env: Deque[float] = deque(maxlen=200)
    recent_r_rnd: Deque[float] = deque(maxlen=200)
    recent_r_novel: Deque[float] = deque(maxlen=200)
    recent_r_bayes: Deque[float] = deque(maxlen=200)

    start_time = time.time()
    last_log = start_time
    last_checkpoint = start_time
    video_every = args.video_interval if args.video_interval else args.record_video_every
    frames_for_video: List[np.ndarray] = []

    for step in range(args.total_steps):
        frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        obs_tensors = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
        structured, milestone_flags = agent.flag_encoder(obs_tensors)
        done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)
        # collect frames for env0 for optional video
        if video_every:
            frames_for_video.append(obs["screen"][0])

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
        rnd_np = intrinsic["rnd"].detach().cpu().numpy()
        novel_np = intrinsic["novel"].detach().cpu().numpy()
        bayes_np = intrinsic["bayes"].detach().cpu().numpy()
        total_np = total_reward.detach().cpu().numpy()
        episode_rnd += rnd_np
        episode_novel += novel_np
        episode_bayes += bayes_np
        episode_total += total_np

        perf.step(args.num_envs)

        for env_idx, done in enumerate(done_flags):
            # log per-step for visuals
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
            # save savestate on map transitions (lightweight curriculum trigger)
            raw_map = stats.get("map_id", -1)
            try:
                map_id = int(raw_map)
            except Exception:
                map_id = -1
            if map_id >= 0 and map_id != last_map[env_idx]:
                map_path = save_state_dir / f"map{map_id:03d}_step{step+1}_env{env_idx}.state"
                envs.envs[env_idx]._save_emulator_state(map_path)
                curriculum.add_state(map_path)
                last_map[env_idx] = map_id

            if done:
                perf.episode()
                bayes_updates = agent.update_bayes_from_milestones(
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
                        "milestones_reached": ";".join(
                            [k for k, v in bayes_updates.items() if v.get("updated", False)]
                        ),
                    }
                )
                # log posterior progress if available
                for k, v in bayes_updates.items():
                    if {"alpha", "beta"} <= set(v.keys()):
                        reward_logger.log_progress(k, v["alpha"], v["beta"], v.get("alarm_score", 0.0))

                recent_lengths.append(float(episode_lengths[env_idx]))
                recent_env.append(float(episode_returns[env_idx]))
                recent_total.append(float(episode_total[env_idx]))
                recent_r_env.append(float(np.mean(r_env.cpu().numpy())))
                recent_r_rnd.append(float(np.mean(rnd_np)))
                recent_r_novel.append(float(np.mean(novel_np)))
                recent_r_bayes.append(float(np.mean(bayes_np)))

                # save savestate on badge milestones
                badge_count = stats.get("badge", 0)
                if badge_count > last_badges[env_idx]:
                    save_path = save_state_dir / f"badge{badge_count:02d}_step{step+1}_env{env_idx}.state"
                    envs.envs[env_idx]._save_emulator_state(save_path)
                    curriculum.add_state(save_path)
                    last_badges[env_idx] = badge_count

                logging.info(
                    "[EP %d env=%d] len=%d R_env=%.2f R_total=%.2f badges=%s map=%s",
                    episode_ids[env_idx],
                    env_idx,
                    episode_lengths[env_idx],
                    episode_returns[env_idx],
                    episode_total[env_idx],
                    badge_count,
                    stats.get("map_id", -1),
                )
                episode_ids[env_idx] += 1
                episode_returns[env_idx] = 0.0
                episode_rnd[env_idx] = 0.0
                episode_novel[env_idx] = 0.0
                episode_bayes[env_idx] = 0.0
                episode_total[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                agent.reset_episode(env_idx)
                # curriculum: rotate through savestates if available
                maybe_state = curriculum.sample_state()
                env = envs.envs[env_idx]
                if maybe_state and maybe_state.exists():
                    env.init_state_name = maybe_state.stem
                    env.init_state_path = Path(env.state_dir) / f"{env.init_state_name}.state"
                if args.save_state_every and (episode_ids[env_idx] % args.save_state_every == 0):
                    save_path = save_state_dir / f"ep{episode_ids[env_idx]:05d}_env{env_idx}.state"
                    env._save_emulator_state(save_path)
                    curriculum.add_state(save_path)
                if video_every and env_idx == 0 and episode_ids[env_idx] % video_every == 0:
                    Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                    out_path = Path(args.video_dir) / f"train_ep{episode_ids[env_idx]:05d}.mp4"
                    if frames_for_video:
                        artifacts = maybe_save_video(frames_for_video, "mp4", out_path, fps=15)
                        for art in artifacts:
                            logging.info("[video] saved %s (%d frames)", art["path"], art["frames"])
                    frames_for_video.clear()
                elif env_idx == 0:
                    # reset buffer even if no video recorded to avoid mixing episodes
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

        # actor critic switch
        obs = next_obs
        states = new_states

        next_frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        next_obs_tensors = {k: torch.as_tensor(v, device=device) for k, v in next_obs.items()}
        next_structured, _ = agent.flag_encoder(next_obs_tensors)

        # store transition
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
            agent.learn(args.batch_size)

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) * args.num_envs / max(elapsed, 1e-6)
            avg_len = np.mean(recent_lengths) if recent_lengths else 0.0
            avg_env = np.mean(recent_env) if recent_env else 0.0
            avg_total = np.mean(recent_total) if recent_total else 0.0
            mean_r_env = np.mean(recent_r_env) if recent_r_env else 0.0
            mean_r_rnd = np.mean(recent_r_rnd) if recent_r_rnd else 0.0
            mean_r_novel = np.mean(recent_r_novel) if recent_r_novel else 0.0
            mean_r_bayes = np.mean(recent_r_bayes) if recent_r_bayes else 0.0
            logging.info(
                "[step %d/%d] SPS=%.1f eps=%.3f avg_len=%.1f avg_env=%.2f avg_total=%.2f | r_env=%.3f r_rnd=%.3f r_novel=%.3f r_bayes=%.3f | episodes=%d envs=%d | %s",
                step + 1,
                args.total_steps,
                sps,
                agent.epsilon,
                avg_len,
                avg_env,
                avg_total,
                mean_r_env,
                mean_r_rnd,
                mean_r_novel,
                mean_r_bayes,
                int(np.sum(episode_ids)),
                args.num_envs,
                curriculum.summary(),
            )
            last_log = time.time()

        if args.eval_interval and (step + 1) % args.eval_interval == 0:
            logging.info("[eval] placeholder: eval_interval set but eval loop not implemented.")

        if args.save_interval and (step + 1) % args.save_interval == 0:
            ckpt_path = log_dir / f"checkpoint_step{step+1}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            logging.info("[ckpt] saved %s", ckpt_path)

        if args.prune_interval and (step + 1) % args.prune_interval == 0:
            before = curriculum.summary()
            curriculum.promote_or_demote()
            after = curriculum.summary()
            logging.info("[prune] %s -> %s", before, after)

    # flush logs
    reward_logger.flush_steps()
    reward_logger.flush_episodes()
    positions_file.flush()


if __name__ == "__main__":
    main()
