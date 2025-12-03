from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_returns(run_dir: Path) -> Optional[Path]:
    ep_path = run_dir / "episodes.csv"
    ev_path = run_dir / "events.csv"
    if ep_path.exists():
        ep = pd.read_csv(ep_path)
        if ep.empty:
            return None
        plt.figure(figsize=(10, 5))
        if "return_total_with_intrinsic" in ep:
            plt.plot(ep["episode_id"], ep["return_total_with_intrinsic"], label="total_with_intrinsic")
        if "return_env" in ep:
            plt.plot(ep["episode_id"], ep["return_env"], label="env")
        if "length_steps" in ep:
            plt.plot(ep["episode_id"], ep["length_steps"], label="length_steps", alpha=0.4)
        plt.xlabel("episode")
        plt.ylabel("value")
        plt.legend()
        plt.title(run_dir.name)
    elif ev_path.exists():
        ev = pd.read_csv(ev_path)
        if ev.empty:
            return None
        plt.figure(figsize=(10, 5))
        plt.plot(ev["step_global"], ev["r_total"].rolling(window=500, min_periods=1).mean(), label="r_total (rolling mean)")
        plt.xlabel("step_global")
        plt.ylabel("r_total")
        plt.legend()
        plt.title(f"{run_dir.name} (from events)")
    else:
        return None
    out = run_dir / "plot_returns.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_bayes(run_dir: Path) -> Optional[Path]:
    prog_path = run_dir / "progress.csv"
    if not prog_path.exists():
        return None
    prog = pd.read_csv(prog_path)
    if prog.empty or "milestone" not in prog:
        return None
    plt.figure(figsize=(10, 5))
    for m in prog["milestone"].unique():
        sub = prog[prog["milestone"] == m]
        mean = sub["alpha"] / (sub["alpha"] + sub["beta"])
        plt.plot(sub.index, mean, label=m)
    plt.xlabel("update")
    plt.ylabel("posterior mean")
    plt.legend()
    plt.title(f"Bayes progress: {run_dir.name}")
    out = run_dir / "plot_bayes.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_map_freq(run_dir: Path) -> Optional[Path]:
    """Simple map frequency chart from positions.csv (if present)."""
    pos_path = run_dir / "positions.csv"
    if not pos_path.exists():
        return None
    pos = pd.read_csv(pos_path)
    if pos.empty or "map_id" not in pos:
        return None
    counts = pos["map_id"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    counts.plot(kind="bar")
    plt.xlabel("map_id")
    plt.ylabel("count")
    plt.title(f"Map visit frequency: {run_dir.name}")
    out = run_dir / "plot_map_freq.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_action_entropy(run_dir: Path) -> Optional[Path]:
    """If positions.csv has entropy column, plot it; otherwise skip."""
    pos_path = run_dir / "positions.csv"
    if not pos_path.exists():
        return None
    pos = pd.read_csv(pos_path)
    if "entropy" not in pos:
        return None
    plt.figure(figsize=(10, 4))
    plt.plot(pos["step_global"], pos["entropy"])
    plt.xlabel("step_global")
    plt.ylabel("action entropy")
    plt.title(f"Entropy timeline: {run_dir.name}")
    out = run_dir / "plot_entropy.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_episode_path(run_dir: Path, episode_id: int | None = None) -> Optional[Path]:
    """
    For a single episode (if recorded in positions.csv), plot map_id and coord over time.
    This is a lightweight placeholder until full map coordinates are logged.
    """
    pos_path = run_dir / "positions.csv"
    if not pos_path.exists():
        return None
    pos = pd.read_csv(pos_path)
    if pos.empty:
        return None
    if episode_id is not None and "episode" in pos:
        pos = pos[pos["episode"] == episode_id]
        if pos.empty:
            return None
    plt.figure(figsize=(10, 4))
    if "map_id" in pos:
        plt.plot(pos["step_global"], pos["map_id"], label="map_id", alpha=0.7)
    if "coord" in pos:
        plt.plot(pos["step_global"], pos["coord"], label="coord", alpha=0.7)
    plt.xlabel("step_global")
    plt.ylabel("value")
    plt.legend()
    plt.title(f"Episode path (coarse): {run_dir.name}")
    out = run_dir / "plot_episode_path.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out
