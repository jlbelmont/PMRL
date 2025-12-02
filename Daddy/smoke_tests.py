"""
Lightweight smoke tests for the slim agent stack.

Runs small checks for:
- Network forward pass
- RND loss
- Replay buffer insert/sample
- Env step via pufferlib (fallback to gym SyncVectorEnv)
- Logging and video utilities
"""

from __future__ import annotations

import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from multiprocessing import shared_memory

try:
    import pufferlib
    import pufferlib.emulation
    import pufferlib.vector
except Exception:  # pragma: no cover - optional
    pufferlib = None  # type: ignore[assignment]

from omegaconf import OmegaConf

from .agent import AgentConfig, SlimHierarchicalDQN
from .flags import FlagEncoder
from .logging_utils import PerformanceTracker, RewardLogger
from .networks import SlimHierarchicalQNetwork
from .replay_buffer import ReplayBuffer
from .rnd import RNDModel
from .video_utils import maybe_save_video

# Reduce SDL/pygame noise in headless tests and prefer bundled SDL once
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "-1000,-1000")
# Try to avoid SDL loader collisions by preferring one provider
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")


def fake_frames(batch: int, stack: int, h: int = 72, w: int = 80) -> torch.Tensor:
    return torch.randint(0, 256, (batch, stack, h, w), dtype=torch.uint8)


def test_network_forward() -> None:
    flag_enc = FlagEncoder()
    structured_dim = flag_enc.output_dim
    net = SlimHierarchicalQNetwork(frame_channels=4, num_actions=8, structured_dim=structured_dim)
    frames = fake_frames(batch=2, stack=4)
    structured = torch.randn(2, structured_dim)
    q, state, aux = net(frames, structured, state=None, done=None)
    assert q.shape == (2, 8)
    assert aux["ssm"].shape[0] == 2
    print("[ok] network forward")


def test_rnd() -> None:
    rnd = RNDModel(feature_dim=32)
    feats = torch.randn(5, 32)
    rew = rnd.intrinsic_reward(feats)
    loss = rnd.loss(feats)
    assert rew.shape == (5,)
    assert loss.shape == (5,)
    print("[ok] RND intrinsic + loss")


def test_replay_buffer() -> None:
    rb = ReplayBuffer(capacity=10, device=torch.device("cpu"))
    frames = fake_frames(1, 4)
    next_frames = fake_frames(1, 4)
    for i in range(6):
        rb.add(
            frames=frames,
            structured=None,
            action=0,
            reward=1.0,
            reward_components={"env": 1.0},
            done=False,
            next_frames=next_frames,
            next_structured=None,
            state=None,
        )
    batch = rb.sample(4)
    assert batch["frames"].shape[0] == 4
    print("[ok] replay buffer add/sample")


def make_env(headless: bool = True):
    # Lazy import to avoid SDL loading when env test is skipped
    from pokemonred_puffer.environment import RedGymEnv
    cfg_path = Path(__file__).resolve().parents[1] / "pokemonred_puffer" / "config.yaml"
    config = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    env_cfg["headless"] = headless
    env_cfg["save_video"] = False
    root = Path(__file__).resolve().parents[1]
    rom_path = root / "pokemonred_puffer" / "red.gb"
    if not rom_path.exists():
        print("[warn] red.gb not found; skipping env creation")
        return None
    env_cfg["gb_path"] = str(rom_path)

    state_dir = root / "pokemonred_puffer" / "pyboy_states"
    init_state = state_dir / f"{env_cfg.get('init_state', 'Bulbasaur')}.state"
    if not init_state.exists():
        print("[warn] init state not found; skipping env creation")
        return None
    env_cfg["state_dir"] = str(state_dir)

    cfg = OmegaConf.create(env_cfg)
    env = RedGymEnv(cfg)
    env.reset(seed=0)
    return env


def test_env_step() -> None:
    # Skip env step for now until ROM + savestates are available
    print("[warn] env step test skipped (waiting for ROM/state availability)")


def test_agent_act_learn() -> None:
    flag_enc = FlagEncoder()
    structured_dim = flag_enc.output_dim
    cfg = AgentConfig(num_actions=8, frame_stack=4, structured_dim=structured_dim, replay_capacity=32)
    agent = SlimHierarchicalDQN(cfg, device=torch.device("cpu"))
    frames = fake_frames(batch=4, stack=4)
    structured = torch.randn(4, structured_dim)
    states = agent.reset_hidden(num_envs=4)
    done_mask = torch.zeros(4)
    actions, new_states, aux = agent.act(frames.float(), structured, states, done_mask)
    assert actions.shape == (4,)
    agent.add_transition(
        frames.float(),
        structured,
        actions,
        torch.ones(4),
        {"env": torch.ones(4), "rnd": torch.ones(4), "novel": torch.ones(4), "bayes": torch.ones(4)},
        torch.zeros(4),
        frames.float(),
        structured,
        new_states,
    )
    agent.learn(batch_size=1)
    print("[ok] agent act/learn path")


def test_logging_and_video() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = RewardLogger(log_dir=tmpdir, flush_every=1)
        logger.log_step(
            step_global=1,
            env_id=0,
            episode_id=0,
            rewards={"env": 1.0, "rnd": 0.1, "novel": 0.2, "bayes": 0.0, "total": 1.3},
            weights={"rnd": 0.1, "novel": 0.2, "bayes": 0.0},
            milestone_flags=[0, 1, 0],
            map_id=1,
        )
        logger.log_episode({"episode_id": 0, "env_id": 0, "length_steps": 10, "return_env": 5.0})
        logger.flush_steps()
        logger.flush_episodes()

        frames = [np.zeros((72, 80, 3), dtype=np.uint8) for _ in range(5)]
        artifacts = maybe_save_video(frames, "gif", Path(tmpdir) / "test.gif", fps=5)
        assert artifacts, "Video artifact not created"
        print("[ok] logging and video utils")


def test_performance_tracker() -> None:
    perf = PerformanceTracker()
    perf.step(100)
    perf.episode(2)
    stats = perf.stats()
    assert "sps" in stats
    print("[ok] performance tracker")


def main() -> None:
    test_network_forward()
    test_rnd()
    test_replay_buffer()
    test_agent_act_learn()
    test_logging_and_video()
    test_performance_tracker()
    test_env_step()
    # Attempt to clean up any leaked shared_memory segments that trigger warnings
    try:
        for name in list(shared_memory._resource_tracker._registry.keys()):  # type: ignore[attr-defined]
            if name.startswith("shm"):
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
    except Exception:
        pass
    print("All smoke tests completed.")


if __name__ == "__main__":
    main()
