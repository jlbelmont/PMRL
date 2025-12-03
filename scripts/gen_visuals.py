"""
Generate a bundle of visuals (returns, Bayes, map freq, entropy, coarse paths)
for a given run directory.

Usage:
  source .venv/bin/activate
  export PYTHONPATH="$PWD:$PWD/pokemonred_puffer:$PYTHONPATH"
  python scripts/gen_visuals.py --run runs/short10k
  # or let it auto-pick latest run in runs/
  python scripts/gen_visuals.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Daddy.visuals import plot_action_entropy, plot_bayes, plot_episode_path, plot_map_freq, plot_returns


def find_latest_run(base: Path) -> Path | None:
    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / "episodes.csv").exists() or (d / "events.csv").exists():
            mtime = max(
                (d / "episodes.csv").stat().st_mtime if (d / "episodes.csv").exists() else 0,
                (d / "events.csv").stat().st_mtime if (d / "events.csv").exists() else 0,
            )
            candidates.append((mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="Run directory (default: latest in runs/)")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else find_latest_run(Path("runs"))
    if run_dir is None or not run_dir.exists():
        print("[err] no run found")
        return

    print(f"[info] generating visuals for {run_dir}")
    plot_returns(run_dir)
    plot_bayes(run_dir)
    plot_map_freq(run_dir)
    plot_action_entropy(run_dir)
    plot_episode_path(run_dir)


if __name__ == "__main__":
    main()
