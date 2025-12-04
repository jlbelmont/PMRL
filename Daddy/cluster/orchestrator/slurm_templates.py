"""
SLURM job script templates for the slim hierarchical agent.
Generates sbatch scripts that run Daddy.train_slim with appropriate arguments.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, Optional


def render_train_job(
    *,
    config: Dict[str, Any],
    run_name: str,
    gpus: int = 1,
    mem_per_gpu: str = "20G",
    hours: int = 4,
    partition: str = "gpu-linuxlab",
    cpus: int = 8,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    ddp_store_path: Optional[str] = None,
    account: str = "engr-class-any",
    exclude_nodes: str = "r40a-02,r40a-03",
) -> str:
    """
    Render a SLURM batch script for training the slim agent.
    
    Args:
        config: Dictionary with training parameters (total_steps, num_envs, etc.)
        run_name: Name for this training run (used for log directories)
        gpus: Number of GPUs to request
        mem_per_gpu: Memory per GPU (e.g., "20G")
        hours: Wall-clock time limit
        partition: SLURM partition
        cpus: Number of CPU cores
        rank: DDP rank (if using distributed training)
        world_size: DDP world size (if using distributed training)
        ddp_store_path: Path for DDP rendezvous file
        account: SLURM account
        exclude_nodes: Nodes to exclude (comma-separated)
    """
    time_str = f"{hours:02d}:00:00"
    
    # Calculate max runtime (leave 15 min buffer before SLURM kills us)
    max_runtime_seconds = (hours * 3600) - 900  # 15 min buffer
    
    # Build train_slim command-line args from config
    train_args = _build_train_args(config, run_name, max_runtime_seconds)
    
    # DDP environment setup (for multi-job training)
    ddp_block = ""
    rank_val = 0 if rank is None else int(rank)
    if world_size is not None and world_size > 1:
        store = ddp_store_path or f"runs/{run_name}/ddp_store"
        ddp_block = f"""
    # DDP environment
    export WORLD_SIZE={int(world_size)}
    export RANK={rank_val}
    export LOCAL_RANK=0
    DDP_STORE_PATH="{store}"
    if [[ ! "$DDP_STORE_PATH" = /* ]]; then
        DDP_STORE_PATH="$PWD/$DDP_STORE_PATH"
    fi
    mkdir -p "$(dirname "$DDP_STORE_PATH")"
    rm -f "$DDP_STORE_PATH"
    export DDP_STORE="file://$DDP_STORE_PATH"
    export SLIM_RANK={rank_val}
    export SLIM_WORLD_SIZE={int(world_size)}
    """
    else:
        # Single job mode - still set SLIM_RANK for consistency
        ddp_block = f"""
    # Single job mode
    export SLIM_RANK={rank_val}
    export SLIM_WORLD_SIZE=1
    """
    
    script = f"""
    #!/bin/bash
    #SBATCH -p {partition}
    #SBATCH -A {account}
    #SBATCH -J slim_{run_name}
    #SBATCH -c {cpus}
    #SBATCH --gres=gpu:{gpus}
    #SBATCH --mem-per-gpu={mem_per_gpu}
    #SBATCH --time={time_str}
    #SBATCH --output=runs/{run_name}/logs/slurm-%j.out
    #SBATCH --error=runs/{run_name}/logs/slurm-%j.err
    #SBATCH -x {exclude_nodes}

    # Navigate to repo root
    cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
    
    # Create directories
    mkdir -p "runs/{run_name}/logs"
    mkdir -p "runs/{run_name}/checkpoints"
    
    # Environment setup
    export OMP_NUM_THREADS={cpus}
    export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
    export PATH="$HOME/.local/bin:$PATH"
    
    # Conda activation (cluster-specific)
    export CONDA_ENVS_DIRS="${{HOME}}/conda_envs/"
    export CONDA_PKGS_DIRS="${{HOME}}/conda_pkgs/"
    source ~/.bashrc 2>/dev/null || true
    
    # Try venv first, fall back to conda
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    elif command -v conda &> /dev/null; then
        conda activate pokemon_slim 2>/dev/null || true
    fi
    {ddp_block}
    echo "Starting slim agent training run {run_name} at $(date)"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURMD_NODENAME"
    
    # Run training
    python -m Daddy.train_slim {train_args}
    
    echo "Training completed at $(date)"
    """
    return dedent(script).strip() + "\n"


def _build_train_args(config: Dict[str, Any], run_name: str, max_runtime_seconds: int = 13500) -> str:
    """
    Convert config dictionary to command-line arguments for train_slim.py.
    
    Args:
        config: Training configuration dictionary
        run_name: Name of the training run
        max_runtime_seconds: Max runtime before graceful exit (default 13500 = 3h45m, safe for 4h SLURM jobs)
    """
    args = []
    
    # Core training params
    if "total_steps" in config:
        args.append(f"--total-steps {config['total_steps']}")
    if "batch_size" in config:
        args.append(f"--batch-size {config['batch_size']}")
    if "buffer_size" in config:
        args.append(f"--buffer-size {config['buffer_size']}")
    if "learning_starts" in config:
        args.append(f"--learning-starts {config['learning_starts']}")
    if "train_every" in config:
        args.append(f"--train-every {config['train_every']}")
    if "target_update" in config:
        args.append(f"--target-update {config['target_update']}")
    if "num_envs" in config:
        args.append(f"--num-envs {config['num_envs']}")
    if "frame_stack" in config:
        args.append(f"--frame-stack {config['frame_stack']}")
    if "seed" in config:
        args.append(f"--seed {config['seed']}")
    
    # Intrinsic reward weights
    if "rnd_weight" in config:
        args.append(f"--rnd-weight {config['rnd_weight']}")
    if "novelty_weight" in config:
        args.append(f"--novelty-weight {config['novelty_weight']}")
    if "bayes_weight" in config:
        args.append(f"--bayes-weight {config['bayes_weight']}")
    
    # Device
    if "device" in config:
        args.append(f"--device {config['device']}")
    
    # Model architecture
    if "cnn_channels" in config:
        args.append(f"--cnn-channels {config['cnn_channels']}")
    if "gru_size" in config:
        args.append(f"--gru-size {config['gru_size']}")
    if "lstm_size" in config:
        args.append(f"--lstm-size {config['lstm_size']}")
    if "ssm_size" in config:
        args.append(f"--ssm-size {config['ssm_size']}")
    if "structured_hidden" in config:
        args.append(f"--structured-hidden {config['structured_hidden']}")
    
    # Logging
    log_dir = config.get("log_dir", f"runs/{run_name}")
    args.append(f"--log-dir {log_dir}")
    if "log_interval" in config:
        args.append(f"--log-interval {config['log_interval']}")
    
    # Video recording (typically disabled on cluster)
    if config.get("record_video_every", 0) > 0:
        args.append(f"--record-video-every {config['record_video_every']}")
        if "video_dir" in config:
            args.append(f"--video-dir {config['video_dir']}")
    
    # Headless mode (always true on cluster)
    args.append("--headless")
    
    # Mode
    mode = config.get("mode", "train")
    args.append(f"--mode {mode}")
    
    # Checkpointing for auto-resubmission (critical for SLURM)
    checkpoint_dir = config.get("checkpoint_dir", f"runs/{run_name}/checkpoints")
    args.append(f"--checkpoint-dir {checkpoint_dir}")
    
    checkpoint_interval = config.get("checkpoint_interval", 10000)
    args.append(f"--checkpoint-interval {checkpoint_interval}")
    
    # Max runtime - exit gracefully before SLURM kills us
    args.append(f"--max-runtime-seconds {max_runtime_seconds}")
    
    # Always resume from latest checkpoint (for auto-resubmission)
    args.append("--resume")
    
    # Max checkpoints to keep
    max_checkpoints = config.get("max_checkpoints", 6)
    args.append(f"--max-checkpoints {max_checkpoints}")
    
    # Streaming (enabled via config)
    if config.get("stream", False):
        args.append("--stream")
        stream_port = config.get("stream_port", 9999)
        args.append(f"--stream-port {stream_port}")
        stream_interval = config.get("stream_interval", 60)
        args.append(f"--stream-interval {stream_interval}")
        stream_top_k = config.get("stream_top_k", 2)
        args.append(f"--stream-top-k {stream_top_k}")
        # Shared candidates path for multi-job coordination
        candidates_path = config.get("stream_candidates_path", f"runs/{run_name}/stream_candidates.json")
        args.append(f"--stream-candidates-path {candidates_path}")
    
    # Job rank (for DDP and streaming coordination) - will be replaced by SLURM template
    args.append("--job-rank $SLIM_RANK")
    
    return " \\\n        ".join(args)

