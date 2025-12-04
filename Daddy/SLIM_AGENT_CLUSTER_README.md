# Slim Pokémon Red RL Agent — Cluster & Distributed Training Guide

Compatible with `pokemonred_puffer`, deployable on WashU Academic Compute Cluster.  
Supports distributed training across cluster + laptops with automatic weight syncing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [SSH Key Setup](#3-ssh-key-setup)
4. [Cluster Training](#4-cluster-training)
5. [Laptop Contribution](#5-laptop-contribution)
6. [Live Streaming](#6-live-streaming)
7. [Weight Synchronization](#7-weight-synchronization)
8. [Configuration Reference](#8-configuration-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

This system enables distributed Pokémon Red RL training across:

- **WashU ENGR Cluster**: 4 concurrent SLURM jobs (max per student)
- **Your Laptop**: Additional actor environments
- **Friend's Laptop**: More actors under their cluster quota

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLUSTER                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Job 0   │ │  Job 1   │ │  Job 2   │ │  Job 3   │           │
│  │ GPU+CPU  │ │ GPU+CPU  │ │ GPU+CPU  │ │ GPU+CPU  │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │                   │
│       └────────────┴─────┬──────┴────────────┘                   │
│                          │                                       │
│              ┌───────────┴───────────┐                          │
│              │    shared_weights/    │ ◄── Polyak averaging     │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │ rsync (SSH key)
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ Your Mac │       │Friend's │       │ Viewer  │
   │ (actors) │       │   Mac   │       │(stream) │
   └──────────┘       └─────────┘       └─────────┘
```

### Cluster Limits (per student)

| Resource | Limit |
|----------|-------|
| Concurrent jobs | 4 |
| CPUs per job | 8 |
| GPU per job | 1 |
| GPU RAM | 20GB |
| Wall time | 4 hours |

The orchestrator automatically resubmits jobs every 4 hours.

---

## 2. Quick Start

### First-Time Setup

```bash
# 1. Setup SSH key (one time)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id a.a.baggio@shell.engr.wustl.edu

# 2. Clone/sync repo to cluster
ssh a.a.baggio@shell.engr.wustl.edu
cd /project/scratch01/compiling/a.a.baggio/PokemonRedExperiments
git pull  # or clone if first time

# 3. Activate environment
source .venv/bin/activate
```

### Start Training on Cluster

```bash
# On cluster login node
python -m Daddy.cluster.orchestrator.run_controller start \
    --config Daddy/cluster/cluster_config.json \
    --partition gpu-linuxlab \
    --ddp-world-size 4

# This submits 4 jobs that auto-resubmit every 4 hours
```

### View Training on Your Laptop

```bash
# Terminal 1: Start viewer
python -m Daddy.streaming.tunnel_server --port 9999

# Terminal 2: Create SSH tunnel
ssh -R 9999:localhost:9999 a.a.baggio@shell.engr.wustl.edu
# Keep this terminal open
```

### Contribute Your Laptop's Compute

```bash
# Sync weights first
./Daddy/cluster/sync_weights.sh pull

# Run actors (auto-detects hardware)
python -m Daddy.run_actors_only \
    --shared-dir ~/shared_weights \
    --num-envs auto \
    --autotune
```

---

## 3. SSH Key Setup

SSH keys enable passwordless authentication, required for automatic weight syncing.

### Generate SSH Key

```bash
# Generate a new key (press Enter for all prompts)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# This creates:
#   ~/.ssh/id_ed25519      (private key - never share!)
#   ~/.ssh/id_ed25519.pub  (public key - share this)
```

### Copy Key to Cluster

```bash
# This will prompt for your cluster password ONE time
ssh-copy-id a.a.baggio@shell.engr.wustl.edu

# Replace 'a.a.baggio' with your username
```

### Test Connection

```bash
# Should connect without password prompt
ssh a.a.baggio@shell.engr.wustl.edu "echo 'SSH key works!'"
```

### Interactive Setup Helper

```bash
# Run the setup helper
./Daddy/cluster/sync_weights.sh setup
```

### For Your Friend

Your friend needs to:

1. Generate their own SSH key: `ssh-keygen -t ed25519`
2. Copy it to cluster: `ssh-copy-id their.username@shell.engr.wustl.edu`
3. Update `sync_weights.sh` with their username or use environment variables:

```bash
export SYNC_CLUSTER_USER="their.username"
./Daddy/cluster/sync_weights.sh pull
```

---

## 4. Cluster Training

### Start the Orchestrator

The orchestrator submits SLURM jobs and auto-resubmits them every 4 hours.

```bash
# SSH to cluster
ssh a.a.baggio@shell.engr.wustl.edu

# Navigate to repo
cd /project/scratch01/compiling/a.a.baggio/PokemonRedExperiments
source .venv/bin/activate

# Start orchestrator (stays running, resubmits jobs)
python -m Daddy.cluster.orchestrator.run_controller start \
    --config Daddy/cluster/cluster_config.json \
    --partition gpu-linuxlab \
    --ddp-world-size 4
```

### Check Status

```bash
python -m Daddy.cluster.orchestrator.run_controller status \
    --config Daddy/cluster/cluster_config.json
```

### Stop Training

```bash
python -m Daddy.cluster.orchestrator.run_controller stop \
    --config Daddy/cluster/cluster_config.json
```

### Monitor Jobs

```bash
# See your running jobs
squeue -u $USER

# Watch job output
tail -f runs/slim_run/logs/slurm-*.out
```

---

## 5. Laptop Contribution

Your laptop can contribute additional training compute by running actor environments.

### Setup

```bash
# 1. Profile your hardware (one time)
python -m Daddy.autotune --user-id $(hostname)

# 2. Start weight sync (in background or separate terminal)
./Daddy/cluster/sync_weights.sh pull "" ~/shared_weights 5

# 3. Run actors
python -m Daddy.run_actors_only \
    --shared-dir ~/shared_weights \
    --num-envs auto \
    --contribute
```

### Options

| Flag | Description |
|------|-------------|
| `--num-envs auto` | Auto-detect optimal number |
| `--num-envs 4` | Run exactly 4 environments |
| `--autotune` | Run hardware benchmark first |
| `--contribute` | Push weights back to shared dir |
| `--stream` | Enable streaming to viewer |
| `--device cpu` | Force CPU (useful for MacBooks) |

### Resource Usage

The autotune system recommends settings based on:

- CPU cores (leaves 1 for system)
- RAM (reserves 4GB for system)
- GPU/MPS availability

Typical recommendations:

| Machine | Recommended Envs |
|---------|-----------------|
| MacBook Air M1 (8GB) | 2-3 |
| MacBook Pro M1 (16GB) | 4-6 |
| MacBook Pro M2 (32GB) | 6-8 |
| Desktop with GPU | 8-16 |

---

## 6. Live Streaming

Watch the best agents play in real-time on your laptop.

### Start Viewer

```bash
# Start the viewer server
python -m Daddy.streaming.tunnel_server --port 9999

# Options:
#   --top-k 2      Show top 2 agents (default)
#   --width 800    Window width
#   --height 600   Window height
#   --headless     Print stats only (no window)
```

### Create SSH Tunnel

In a separate terminal:

```bash
# Create reverse tunnel from cluster to laptop
ssh -R 9999:localhost:9999 a.a.baggio@shell.engr.wustl.edu

# Keep this terminal open!
# The tunnel routes cluster → localhost:9999 → your laptop
```

### How It Works

1. Cluster jobs track agent progress (badges + story flags + reward)
2. Only top-K agents send frames to save bandwidth
3. Frames go through SSH reverse tunnel to your laptop
4. Viewer displays frames in a pygame window

If your laptop closes or sleeps, streaming stops but training continues.

---

## 7. Weight Synchronization

### Manual Sync

```bash
# Pull weights from cluster
./Daddy/cluster/sync_weights.sh pull

# Push weights to cluster  
./Daddy/cluster/sync_weights.sh push

# Bidirectional sync
./Daddy/cluster/sync_weights.sh bidirectional
```

### Continuous Sync

```bash
# Pull every 5 seconds
./Daddy/cluster/sync_weights.sh pull "" ~/shared_weights 5

# Run in background
./Daddy/cluster/sync_weights.sh pull "" ~/shared_weights 5 &
```

### macOS LaunchAgent (Auto-Start)

Create a LaunchAgent to start syncing automatically:

```bash
cat > ~/Library/LaunchAgents/com.pokemon.sync.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pokemon.sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd ~/PMRL-main && ./Daddy/cluster/sync_weights.sh pull "" ~/shared_weights 5</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Load the agent
launchctl load ~/Library/LaunchAgents/com.pokemon.sync.plist

# Unload when done
launchctl unload ~/Library/LaunchAgents/com.pokemon.sync.plist
```

---

## 8. Configuration Reference

### Cluster Config: `Daddy/cluster/cluster_config.json`

```json
{
  "run_name": "slim_run",
  "run_dir": "runs/slim_run",
  
  "training": {
    "total_steps": 5000000,
    "num_envs": 8,
    "batch_size": 64,
    "device": "cuda",
    "stream": true,
    "stream_port": 9999
  },
  
  "collaboration": {
    "enabled": true,
    "shared_dir": "/project/scratch01/.../shared_weights"
  }
}
```

### Key Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `total_steps` | Total training steps | 5,000,000 |
| `num_envs` | Environments per job | 8 |
| `checkpoint_interval` | Steps between checkpoints | 10,000 |
| `max_runtime_seconds` | Exit before SLURM kills (4h = 14400) | 13,500 |
| `stream_port` | SSH tunnel port | 9999 |
| `stream_top_k` | Number of agents to stream | 2 |

---

## 9. Troubleshooting

### SSH Connection Failed

```bash
# Check if key exists
ls -la ~/.ssh/id_ed25519

# Test connection with verbose output
ssh -v a.a.baggio@shell.engr.wustl.edu

# Re-copy key if needed
ssh-copy-id a.a.baggio@shell.engr.wustl.edu
```

### Jobs Not Starting

```bash
# Check queue
squeue -u $USER

# Check for errors
cat runs/slim_run/logs/slurm-*.err

# Check partition availability
sinfo -p gpu-linuxlab
```

### Streaming Not Working

1. Check viewer is running: `python -m Daddy.streaming.tunnel_server --port 9999`
2. Check tunnel is open: `ssh -R 9999:localhost:9999 user@cluster`
3. Check training has `stream: true` in config
4. Check firewall isn't blocking port 9999

### Weight Sync Failing

```bash
# Test SSH connection
ssh a.a.baggio@shell.engr.wustl.edu "echo 'works'"

# Check rsync is installed
which rsync

# Test rsync manually
rsync -avz user@cluster:/path/to/test.txt ./
```

### Out of Memory on Laptop

```bash
# Reduce environments
python -m Daddy.run_actors_only --num-envs 2

# Or run autotune to get recommendations
python -m Daddy.autotune --user-id $(hostname)
```

---

## Files Reference

```
Daddy/
├── train_slim.py              # Main training script
├── run_actors_only.py         # Laptop actor-only mode
├── autotune.py                # Hardware detection
├── streaming/
│   ├── tunnel_server.py       # Laptop viewer
│   ├── tunnel_client.py       # Cluster frame sender
│   └── best_selector.py       # Top-K agent selection
└── cluster/
    ├── cluster_config.json    # Training configuration
    ├── sync_weights.sh        # Weight sync script
    └── orchestrator/
        ├── run_controller.py  # SLURM job manager
        └── slurm_templates.py # Job script generator
```
