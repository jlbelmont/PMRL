"""
Orchestrator for auto-resubmitting SLURM jobs.
Ported from cluster_rl_next for the slim hierarchical agent.
"""

from .run_controller import main as run_controller_main

__all__ = ["run_controller_main"]

