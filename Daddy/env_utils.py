"""
Env utilities for the slim agent.

Provides a SafeRedEnv wrapper that tolerates missing savestates by creating
an initial one on first use, and helpers for saving/loading savestates.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from pokemonred_puffer.environment import RedGymEnv
import numpy as np


class SafeRedEnv(RedGymEnv):
    """
    Drop-in RedGymEnv that can run without pre-existing savestates.
    If the configured init_state is missing, it creates a fresh savestate
    from the default game boot so resets never crash.
    """

    def __init__(self, env_config: DictConfig):
        super().__init__(env_config)
        self._ensure_init_state()

    def _ensure_init_state(self) -> None:
        if not self.init_state_path.exists():
            self.init_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.init_state_path, "wb") as f:
                self.pyboy.save_state(f)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._ensure_init_state()
        return super().reset(seed=seed, options=options)

    def save_state(self, path: Path) -> None:
        """
        Save the current emulator state to a .state file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)


def stack_obs_list(obs_list):
    """
    Stack a list of dict observations into batched dict of np.ndarrays.
    """
    out = {}
    for key in obs_list[0].keys():
        items = [o[key] for o in obs_list]
        arrs = []
        for it in items:
            a = np.array(it)
            if a.shape == ():  # scalar -> (1,)
                a = np.expand_dims(a, 0)
            arrs.append(a)
        out[key] = np.stack(arrs, axis=0)
    return out


class SimpleVectorEnv:
    """
    Minimal synchronous vector env that stacks dict observations.
    Avoids gym.vector stacking issues with scalar boxes.
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed: Optional[int] = None):
        obs_list, infos = [], []
        for idx, env in enumerate(self.envs):
            o, info = env.reset(seed=None if seed is None else seed + idx)
            obs_list.append(o)
            infos.append(info)
        return stack_obs_list(obs_list), infos

    def step(self, actions):
        obs_list, rews, terms, truncs, infos = [], [], [], [], []
        for env, act in zip(self.envs, actions):
            o, r, t, tr, info = env.step(int(act))
            obs_list.append(o)
            rews.append(r)
            terms.append(t)
            truncs.append(tr)
            infos.append(info)
        return (
            stack_obs_list(obs_list),
            np.array(rews),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()
