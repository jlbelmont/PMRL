"""
Lightweight rollout script to sanity-check env integration, WRAM flags, and video export.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from .agent import AgentConfig, SlimHierarchicalDQN
from .env_utils import SafeRedEnv, SimpleVectorEnv
from .flags import FlagEncoder
from .video_utils import maybe_save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug rollout for the slim agent.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", choices=["none", "mp4", "gif", "both"], default="none")
    parser.add_argument("--video-dir", type=str, default="runs/videos")
    parser.add_argument("--headless", action="store_true", help="Run PyBoy headless")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser.parse_args()


def load_env_config(headless: bool) -> Dict:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "pokemonred_puffer" / "config.yaml"
    config = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    env_cfg["headless"] = headless
    env_cfg["save_video"] = False
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
    init_state_file = Path(env_cfg["state_dir"]) / f"{init_state}.state"
    if not init_state_file.exists():
        candidates = sorted(Path(env_cfg["state_dir"]).glob("*.state"))
        if candidates:
            env_cfg["init_state"] = candidates[0].stem
    return env_cfg


def make_env(env_cfg: Dict, seed: int) -> SafeRedEnv:
    cfg = OmegaConf.create(env_cfg)
    env = SafeRedEnv(cfg)
    env.reset(seed=seed)
    return env


def stack_frames(frame_stacks: List[Deque[np.ndarray]]) -> np.ndarray:
    """
    Convert a list of frame deques into (B, C, H, W) ready for the network.
    Mirrors the stack_frames used in train_big/train_slim:
      * squeezes stray singleton dims that appear from wrappers,
      * normalizes to channel-first,
      * guarantees at least 8x8 spatial size by repeat/pad,
      * merges time into channels.
    """
    batches: List[np.ndarray] = []
    for frames in frame_stacks:
        arr = np.stack(list(frames), axis=0)  # (T, ...)

        # Squeeze extra singleton dims (sometimes (T,1,H,W,1), etc.)
        while arr.ndim > 4 and (arr.shape[1] == 1 or arr.shape[-1] == 1):
            if arr.shape[1] == 1:
                arr = np.squeeze(arr, axis=1)
            elif arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            else:
                break

        # Normalize to channel-first (T, C, H, W)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 3:  # (T, H, W)
            arr = arr[:, None, ...]
        elif arr.ndim == 2:  # (H, W)
            arr = arr[None, None, ...]

        if arr.ndim != 4:
            raise ValueError(f"Unexpected frame shape {arr.shape}")

        _, _, h, w = arr.shape

        # Repeat to reach minimum spatial size for conv kernels
        rep_h = (8 + h - 1) // h
        rep_w = (8 + w - 1) // w
        if rep_h > 1 or rep_w > 1:
            arr = np.repeat(np.repeat(arr, rep_h, axis=2), rep_w, axis=3)
            h, w = arr.shape[2], arr.shape[3]

        # Pad if still undersized
        pad_h = max(0, 8 - h)
        pad_w = max(0, 8 - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        # Merge time into channels
        t, c, h, w = arr.shape
        merged = arr.reshape(t * c, h, w).astype(np.float32)
        # Final safety: ensure spatial dims still meet conv expectations
        mh, mw = merged.shape[1], merged.shape[2]
        pad_h = max(0, 8 - mh)
        pad_w = max(0, 8 - mw)
        if pad_h or pad_w:
            merged = np.pad(merged, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        batches.append(merged)

    return np.stack(batches, axis=0)


def to_torch_batch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_config(headless=args.headless)

    def env_fn(idx: int):
        return make_env(env_cfg, seed=args.seed + idx)

    envs = SimpleVectorEnv([lambda idx=i: env_fn(idx) for i in range(args.num_envs)])
    obs, _ = envs.reset(seed=args.seed)
    num_actions = envs.single_action_space.n

    frame_stacks: List[Deque[np.ndarray]] = [deque(maxlen=args.frame_stack) for _ in range(args.num_envs)]
    for i in range(args.num_envs):
        for _ in range(args.frame_stack):
            frame_stacks[i].append(obs["screen"][i])

    flag_encoder = FlagEncoder()
    agent_cfg = AgentConfig(
        num_actions=num_actions,
        frame_stack=args.frame_stack,
        structured_dim=flag_encoder.output_dim,
        replay_capacity=1,
    )
    agent = SlimHierarchicalDQN(agent_cfg, device=device)
    agent.flag_encoder = flag_encoder
    states = agent.reset_hidden(args.num_envs)
    dones = np.zeros(args.num_envs, dtype=bool)
    frames_for_video: List[np.ndarray] = []

    for step in range(args.steps):
        frame_batch = torch.as_tensor(stack_frames(frame_stacks), device=device)
        # Final tensor-level safety to avoid tiny spatial dims slipping through
        if frame_batch.shape[-2] < 8 or frame_batch.shape[-1] < 8:
            pad_h = max(0, 8 - frame_batch.shape[-2])
            pad_w = max(0, 8 - frame_batch.shape[-1])
            frame_batch = torch.nn.functional.pad(frame_batch, (0, pad_w, 0, pad_h), mode="replicate")
        structured, milestone_flags = agent.flag_encoder(to_torch_batch(obs, device=device))
        done_mask = torch.as_tensor(dones, device=device, dtype=torch.float32)
        actions, new_states, aux = agent.act(frame_batch, structured, states, done_mask)
        obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
        dones = np.logical_or(terminated, truncated)

        for env_idx, done in enumerate(dones):
            if done:
                reset_obs, _ = envs.envs[env_idx].reset(seed=args.seed + env_idx)
                for key in obs:
                    obs[key][env_idx] = reset_obs[key]
                frame_stacks[env_idx].clear()
                for _ in range(args.frame_stack):
                    frame_stacks[env_idx].append(reset_obs["screen"])
                new_states[env_idx] = agent.network.initial_state(batch_size=1, device=device)
            else:
                frame_stacks[env_idx].append(obs["screen"][env_idx])

        states = new_states
        if args.video != "none" and args.num_envs == 1:
            # Expand grayscale channel for quick inspection
            frame_rgb = np.repeat(obs["screen"][0], repeats=3, axis=-1)
            frames_for_video.append(frame_rgb)

        if step % 50 == 0:
            print(
                f"step {step} | reward {np.mean(rewards):.3f} | rnd {agent.rnd.intrinsic_reward(aux['ssm']).mean().item():.3f}"
            )

    if args.video != "none" and frames_for_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.video_dir) / f"debug_rollout.{args.video if args.video != 'both' else 'mp4'}"
        artifacts = maybe_save_video(frames_for_video, args.video if args.video != "none" else "mp4", out_path, fps=15)
        for art in artifacts:
            print(f"Saved video: {art['path']} ({art['frames']} frames)")

    envs.close()


if __name__ == "__main__":
    main()
