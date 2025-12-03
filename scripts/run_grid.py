"""
Automate multiple training runs with different model sizes and plot the results.

Usage:
  source .venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/pokemonred_puffer:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=""
  python scripts/run_grid.py --runs small medium --total-steps 200000 --num-envs 4

You can override model sizes per run with the flags below (cnn, gru, lstm, ssm, structured).
Results are logged under runs/<run_name>/ and plots are written to runs/plots.png.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["small", "medium", "big"], help="Names for the runs.")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--save-state-every", type=int, default=0)
    p.add_argument("--record-video-every", type=int, default=0)
    p.add_argument("--video-dir", type=str, default="runs/videos")
    # model sizes
    p.add_argument("--cnn", nargs=3, type=int, default=[32, 64, 64], help="CNN channels triplet.")
    p.add_argument("--gru", type=int, default=128)
    p.add_argument("--lstm", type=int, default=128)
    p.add_argument("--ssm", type=int, default=128)
    p.add_argument("--structured-hidden", type=int, default=128)
    return p.parse_args()


def run_train(run_name: str, args: argparse.Namespace) -> None:
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "-m",
        "Daddy.train_slim",
        f"--log-dir={log_dir}",
        f"--total-steps={args.total_steps}",
        f"--num-envs={args.num_envs}",
        f"--frame-stack={args.frame_stack}",
        f"--device={args.device}",
        f"--log-interval={args.log_interval}",
        f"--save-state-every={args.save_state_every}",
        f"--record-video-every={args.record_video_every}",
        f"--video-dir={args.video_dir}",
        f"--cnn-channels={','.join(map(str, args.cnn))}",
        f"--gru-size={args.gru}",
        f"--lstm-size={args.lstm}",
        f"--ssm-size={args.ssm}",
        f"--structured-hidden={args.structured_hidden}",
    ]
    if args.headless:
        cmd.append("--headless")
    print(f"[run_grid] launching {run_name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def plot_runs(run_names: List[str]) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for run in run_names:
        ep_path = Path("runs") / run / "episodes.csv"
        if not ep_path.exists():
            continue
        df = pd.read_csv(ep_path)
        plt.plot(df["episode_id"], df["return_total_with_intrinsic"], label=run)
    plt.xlabel("episode")
    plt.ylabel("return_total_with_intrinsic")
    plt.legend()
    plt.title("Episode returns per run")
    out = Path("runs") / "plots.png"
    out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out)
    print(f"[run_grid] saved plot to {out}")


def main() -> None:
    args = parse_args()
    for run in args.runs:
        run_train(run, args)
    plot_runs(args.runs)


if __name__ == "__main__":
    main()
