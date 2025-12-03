"""
Lightweight visualization helpers for the slim agent.

Functions read logged CSVs in a run directory:
- episodes.csv / events.csv      : returns, lengths, r_total
- progress.csv                   : Bayes posteriors (alpha/beta per milestone)
- positions.csv (new, optional)  : per-step map/coord/reward snapshots
"""

from .plots import (
    plot_returns,
    plot_bayes,
    plot_map_freq,
    plot_action_entropy,
    plot_episode_path,
)

__all__ = [
    "plot_returns",
    "plot_bayes",
    "plot_map_freq",
    "plot_action_entropy",
    "plot_episode_path",
]
