"""
SLURM job controller for auto-resubmitting training jobs.

Ported from cluster_rl_next/orchestrator/run_controller.py for the slim agent.

Usage:
    # Start training (auto-resubmits every 4 hours)
    python -m Daddy.cluster.orchestrator.run_controller start \\
        --config Daddy/cluster/cluster_config.json \\
        --partition gpu-linuxlab \\
        --ddp-world-size 4

    # Check status
    python -m Daddy.cluster.orchestrator.run_controller status \\
        --config Daddy/cluster/cluster_config.json

    # Stop training
    python -m Daddy.cluster.orchestrator.run_controller stop \\
        --config Daddy/cluster/cluster_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .slurm_templates import render_train_job


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _load_config(path: str) -> Dict[str, Any]:
    """Load JSON config file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Automate chained SLURM runs for the slim Pokémon Red agent."
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status"],
        help="Controller action."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to cluster config JSON."
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--mem-per-gpu", default="20G")
    parser.add_argument("--hours", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--partition", default="gpu-linuxlab")
    parser.add_argument("--account", default="engr-class-any")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between queue checks."
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Optional limit on chained jobs (0 = unlimited)."
    )
    parser.add_argument(
        "--ddp-world-size",
        type=int,
        default=1,
        help="When >1, submit that many jobs for distributed training."
    )
    parser.add_argument(
        "--ddp-store-path",
        default=None,
        help="Optional path for DDP rendezvous file."
    )
    return parser.parse_args()


def _state_paths(run_dir: str) -> Dict[str, str]:
    """Get paths for controller state files."""
    ctrl_dir = os.path.join(run_dir, "controller")
    _ensure_dir(ctrl_dir)
    return {
        "state": os.path.join(ctrl_dir, "state.json"),
        "stop": os.path.join(ctrl_dir, "stop"),
        "slurm_dir": os.path.join(ctrl_dir, "slurm_scripts"),
    }


def _load_state(path: str) -> Dict[str, Any]:
    """Load controller state from disk."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_state(path: str, data: Dict[str, Any]) -> None:
    """Atomically save controller state."""
    _ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _submit_job(
    config: Dict[str, Any],
    run_name: str,
    args,
    slurm_dir: str
) -> str:
    """Submit a single SLURM job and return the job ID."""
    script = render_train_job(
        config=config.get("training", {}),
        run_name=run_name,
        gpus=args.gpus,
        mem_per_gpu=args.mem_per_gpu,
        hours=args.hours,
        cpus=args.cpus,
        partition=args.partition,
        account=args.account,
    )
    _ensure_dir(slurm_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.join(slurm_dir, f"train_{timestamp}.sbatch")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)
    
    proc = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
        check=True
    )
    output = proc.stdout.strip()
    parts = output.split()
    job_id = parts[-1] if parts else output
    print(f"[controller] submitted job {job_id}")
    return job_id


def _submit_ddp_jobs(
    config: Dict[str, Any],
    run_name: str,
    args,
    slurm_dir: str
) -> list[str]:
    """Submit multiple SLURM jobs for distributed training."""
    _ensure_dir(slurm_dir)
    job_ids: list[str] = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    for rank in range(int(args.ddp_world_size)):
        store_path = args.ddp_store_path or f"runs/{run_name}/ddp_store"
        script = render_train_job(
            config=config.get("training", {}),
            run_name=run_name,
            gpus=args.gpus,
            mem_per_gpu=args.mem_per_gpu,
            hours=args.hours,
            cpus=args.cpus,
            partition=args.partition,
            account=args.account,
            rank=rank,
            world_size=int(args.ddp_world_size),
            ddp_store_path=store_path,
        )
        script_path = os.path.join(slurm_dir, f"train_rank{rank}_{timestamp}.sbatch")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script)
        
        proc = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        output = proc.stdout.strip()
        parts = output.split()
        job_id = parts[-1] if parts else output
        print(f"[controller] submitted rank {rank} as job {job_id}")
        job_ids.append(job_id)
        time.sleep(0.2)  # Small delay between submissions
    
    return job_ids


def _job_active(job_id: str) -> bool:
    """Check if a SLURM job is still running or pending."""
    if not job_id:
        return False
    proc = subprocess.run(
        ["squeue", "-j", str(job_id), "-h"],
        capture_output=True,
        text=True
    )
    return proc.returncode == 0 and proc.stdout.strip() != ""


def _cancel_job(job_id: str) -> None:
    """Cancel a SLURM job."""
    if not job_id:
        return
    subprocess.run(["scancel", str(job_id)], capture_output=True, text=True)


def _print_status(state_path: str) -> None:
    """Print current controller status."""
    state = _load_state(state_path)
    if not state:
        print("No controller state found.")
        return
    
    job_id = state.get("job_id")
    rounds = state.get("rounds", 0)
    last_launch = state.get("last_launch")
    job_ids = state.get("job_ids")
    
    if job_ids:
        # DDP mode - multiple jobs
        active_map = {jid: _job_active(jid) for jid in job_ids}
        any_active = any(active_map.values())
        print(json.dumps({
            "job_ids": job_ids,
            "rounds": rounds,
            "last_launch": last_launch,
            "any_active": any_active,
            "active_map": active_map
        }, indent=2))
        return
    
    # Single job mode
    active = _job_active(job_id)
    print(json.dumps({
        "job_id": job_id,
        "rounds": rounds,
        "last_launch": last_launch,
        "active": active
    }, indent=2))


def _start_controller(
    config: Dict[str, Any],
    config_path: str,
    args,
    paths: Dict[str, str]
) -> None:
    """Main controller loop - submits and monitors jobs."""
    run_name = config.get("run_name", "slim_run")
    run_dir = config.get("run_dir", f"runs/{run_name}")
    _ensure_dir(run_dir)
    
    slurm_dir = paths["slurm_dir"]
    rounds = 0
    stop_path = paths["stop"]
    
    print(f"[controller] Starting training run: {run_name}")
    print(f"[controller] Run directory: {run_dir}")
    print(f"[controller] DDP world size: {args.ddp_world_size}")
    
    while True:
        # Check for stop signal
        if os.path.exists(stop_path):
            print("[controller] stop file detected, exiting.")
            break
        
        if int(args.ddp_world_size) > 1:
            # Submit multiple jobs for distributed training
            job_ids = _submit_ddp_jobs(config, run_name, args, slurm_dir)
            state = {
                "job_ids": job_ids,
                "rounds": rounds,
                "last_launch": datetime.utcnow().isoformat()
            }
            _save_state(paths["state"], state)
            
            # Monitor all jobs
            while True:
                statuses = {jid: _job_active(jid) for jid in job_ids}
                any_active = any(statuses.values())
                all_active = all(statuses.values()) if job_ids else False
                
                if not any_active:
                    # All jobs finished
                    break
                
                if os.path.exists(stop_path):
                    print("[controller] stop requested, cancelling jobs.")
                    for jid in job_ids:
                        _cancel_job(jid)
                    break
                
                if any_active and not all_active:
                    # One or more ranks died; cancel remaining to allow clean resubmission
                    print("[controller] detected rank dropout; cancelling remaining ranks to resubmit as a group.")
                    for jid, active in statuses.items():
                        if active:
                            _cancel_job(jid)
                    break
                
                time.sleep(max(10, args.poll_interval))
        else:
            # Single job mode
            job_id = _submit_job(config, run_name, args, slurm_dir)
            state = {
                "job_id": job_id,
                "rounds": rounds,
                "last_launch": datetime.utcnow().isoformat()
            }
            _save_state(paths["state"], state)
            
            # Wait for job to complete
            while _job_active(job_id):
                if os.path.exists(stop_path):
                    print("[controller] stop requested, cancelling job.")
                    _cancel_job(job_id)
                    break
                time.sleep(max(10, args.poll_interval))
        
        # Check if we should stop
        if os.path.exists(stop_path):
            break
        
        rounds += 1
        if args.max_rounds and rounds >= args.max_rounds:
            print("[controller] max rounds reached, exiting.")
            break
        
        print("[controller] job(s) completed, resubmitting after short delay.")
        time.sleep(15)


def _stop_controller(paths: Dict[str, str]) -> None:
    """Signal the controller to stop and cancel active jobs."""
    # Create stop file
    with open(paths["stop"], "w", encoding="utf-8") as fh:
        fh.write("stop")
    
    # Load state and cancel any active jobs
    state = _load_state(paths["state"])
    job_id = state.get("job_id")
    job_ids = state.get("job_ids")
    
    if job_ids:
        # DDP mode
        cancelled_any = False
        for jid in job_ids:
            if _job_active(jid):
                print(f"[controller] cancelling active job {jid}")
                _cancel_job(jid)
                cancelled_any = True
        if not cancelled_any:
            print("[controller] no active DDP jobs to cancel.")
        return
    
    # Single job mode
    if job_id and _job_active(job_id):
        print(f"[controller] cancelling active job {job_id}")
        _cancel_job(job_id)
    else:
        print("[controller] no active job to cancel.")


def main():
    args = _parse_args()
    config = _load_config(args.config)
    
    run_name = config.get("run_name", "slim_run")
    run_dir = config.get("run_dir", f"runs/{run_name}")
    paths = _state_paths(run_dir)
    
    if args.command == "status":
        _print_status(paths["state"])
        return
    
    if args.command == "stop":
        _stop_controller(paths)
        return
    
    if args.command == "start":
        # Remove any existing stop file
        if os.path.exists(paths["stop"]):
            os.remove(paths["stop"])
        
        try:
            _start_controller(config, args.config, args, paths)
        except KeyboardInterrupt:
            print("\n[controller] interrupted, shutting down.")
            _stop_controller(paths)
        return
    
    print("Unknown command", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

