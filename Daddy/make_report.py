"""
Generate a LaTeX report for a training run.

Usage:
    python -m Daddy.make_report --run-dir runs/slim --output-tex docs/report.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .networks import SlimHierarchicalQNetwork


def _load_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _load_rewards(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps_path = run_dir / "steps.csv"
    if not steps_path.exists():
        return np.array([]), np.array([]), np.array([])
    import pandas as pd  # type: ignore

    df = pd.read_csv(steps_path)
    steps = df["step_global"].values
    r_env = df["r_env"].values
    r_total = df["r_total"].values
    return steps, r_env, r_total


def _plot_rewards(run_dir: Path, fig_dir: Path) -> str:
    steps, r_env, r_total = _load_rewards(run_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "rewards_best_run.pdf"
    if steps.size == 0:
        plt.figure()
        plt.text(0.5, 0.5, "No reward data", ha="center", va="center")
    else:
        plt.figure(figsize=(7, 4))
        plt.plot(steps, r_env, label="Env reward", alpha=0.7)
        plt.plot(steps, r_total, label="Total reward", alpha=0.7)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def _model_diagram(fig_dir: Path) -> str:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "model_diagram.pdf"
    plt.figure(figsize=(8, 2))
    stages = ["Frames", "CNN", "GRU", "LSTM", "SSM", "Q-head"]
    x = np.arange(len(stages))
    for i, s in enumerate(stages):
        plt.gca().add_patch(plt.Rectangle((i, 0.25), 0.8, 0.5, edgecolor="black", facecolor="#c7d6ff"))
        plt.text(i + 0.4, 0.5, s, ha="center", va="center")
    plt.xlim(-0.5, len(stages) - 0.2)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def _model_table(config: Dict) -> str:
    rows = [
        ("CNN channels", config.get("cnn_channels", "32,64,64")),
        ("GRU size", config.get("gru_size", 128)),
        ("LSTM size", config.get("lstm_size", 128)),
        ("SSM size", config.get("ssm_size", 128)),
        ("Structured dim", config.get("structured_hidden", 128)),
    ]
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Parameter & Value \\\\", "\\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _config_table(config: Dict) -> str:
    rows = [
        ("Total steps", config.get("total_steps", "")),
        ("Num envs", config.get("num_envs", "")),
        ("Batch size", config.get("batch_size", "")),
        ("Buffer size", config.get("buffer_size", "")),
        ("Learning rate", config.get("lr", config.get("learning_rate", ""))),
        ("Frame stack", config.get("frame_stack", "")),
        ("Replay capacity", config.get("replay_capacity", "")),
    ]
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Config & Value \\\\", "\\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _summary_text(steps: np.ndarray, r_env: np.ndarray, r_total: np.ndarray) -> str:
    if steps.size == 0:
        return "No reward data available."
    best_idx = np.argmax(r_total)
    return (
        f"Best total reward {r_total[best_idx]:.2f} at step {int(steps[best_idx])}. "
        f"Final env reward {r_env[-1]:.2f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX report for a run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (logs).")
    parser.add_argument("--output-tex", type=str, required=True, help="Output .tex path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_tex = Path(args.output_tex)
    fig_dir = output_tex.parent / "figures"

    config = _load_config(run_dir)
    steps, r_env, r_total = _load_rewards(run_dir)
    reward_plot = _plot_rewards(run_dir, fig_dir)
    model_fig = _model_diagram(fig_dir)
    model_table = _model_table(config)
    config_table = _config_table(config)
    summary_text = _summary_text(steps, r_env, r_total)

    template_path = Path(__file__).resolve().parents[1] / "docs" / "report_template.tex"
    tex = template_path.read_text()
    model_rel = Path(model_fig).relative_to(output_tex.parent)
    reward_rel = Path(reward_plot).relative_to(output_tex.parent)
    tex = tex.replace("<<MODEL_DIAGRAM>>", str(model_rel))
    tex = tex.replace("<<MODEL_TABLE>>", model_table)
    tex = tex.replace("<<CONFIG_TABLE>>", config_table)
    tex = tex.replace("<<REWARD_PLOT>>", str(reward_rel))
    tex = tex.replace("<<SUMMARY_TEXT>>", summary_text)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex)
    print(f"[report] wrote {output_tex}")


if __name__ == "__main__":
    main()
