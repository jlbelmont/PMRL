"""
Plot training and Bayes progress for the most recent run.

Usage (from repo root):
  source .venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/pokemonred_puffer:$PYTHONPATH"
  python scripts/plot_latest.py            # auto-picks newest run in runs/
  python scripts/plot_latest.py --run runs/sanity_small   # specify a run dir

Outputs:
  <run_dir>/plot_returns.png      (episode returns and lengths)
  <run_dir>/plot_bayes.png        (posterior mean per milestone, if progress.csv exists)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_run(base: Path) -> Optional[Path]:
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / "episodes.csv").exists():
            mtime = (d / "episodes.csv").stat().st_mtime
            candidates.append((mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def plot_returns(run_dir: Path) -> Optional[Path]:
    ep_path = run_dir / "episodes.csv"
    ev_path = run_dir / "events.csv"
    if ep_path.exists():
        ep = pd.read_csv(ep_path)
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
            print(f"[warn] events.csv empty in {run_dir}")
            return None
        plt.figure(figsize=(10, 5))
        plt.plot(ev["step_global"], ev["r_total"].rolling(window=500, min_periods=1).mean(), label="r_total (rolling mean)")
        plt.xlabel("step_global")
        plt.ylabel("r_total")
        plt.legend()
        plt.title(f"{run_dir.name} (from events)")
    else:
        print(f"[warn] no episodes.csv or events.csv in {run_dir}")
        return None
    out = run_dir / "plot_returns.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[ok] wrote {out}")
    return out


def plot_bayes(run_dir: Path) -> Optional[Path]:
    prog_path = run_dir / "progress.csv"
    if not prog_path.exists():
        print(f"[warn] no progress.csv in {run_dir}")
        return None
    prog = pd.read_csv(prog_path)
    if prog.empty or "milestone" not in prog:
        print(f"[warn] empty or missing milestone column in {prog_path}")
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
    print(f"[ok] wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="Run directory (default: latest in runs/)")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run(Path("runs"))
        if run_dir is None:
            print("[err] no runs found (expected runs/<run>/episodes.csv)")
            return
    if not run_dir.exists():
        print(f"[err] run dir not found: {run_dir}")
        return

    plot_returns(run_dir)
    plot_bayes(run_dir)


if __name__ == "__main__":
    main()
