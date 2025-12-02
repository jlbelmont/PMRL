
# Slim Pokémon Red RL Agent — Cluster‑Ready & Copilot‑Ready Guide  
### Compatible with `pokemonred_puffer`, deployable on WashU Academic Compute Cluster  
### Includes full Codex/Copilot prompts + SLURM job templates

---

# 1. Overview

This document extends the slim‑agent design spec with **cluster‑adaptation instructions** so the entire `Daddy/` codebase can run on the **WashU Academic Compute Cluster** (ENGR Cluster) using:

- `qlogin` (CPU interactive)
- `qlogin-gpu` (GPU interactive)
- **SLURM batch jobs** via `sbatch`
- Academic Jupyter GPU Notebook sessions  

Reference: *Academic Compute Cluster Guide* fileciteturn0file0

The goal is to make the slim agent fully runnable in all cluster modes.

---

# 2. Required Repository Structure

```
pokemonred_puffer/            # DO NOT MODIFY
Final/epsilon/                # Old model (leave intact)
Daddy/                        # New slim agent lives entirely here
  DESIGN_SLIM_MODEL.md
  README.md
  __init__.py
  agent.py
  networks.py
  rnd.py
  flags.py
  bayes_quests.py
  curriculum.py
  replay_buffer.py
  logging_utils.py
  video_utils.py
  train_slim.py
  debug_rollout.py
  cluster/
    slurm_gpu.job
    slurm_cpu.job
    env_setup.sh
```

---

# 3. Cluster Adaptation Requirements

The slim agent must:

### ✔ Work in **interactive GPU sessions**:
```
qlogin-gpu -g 10
```

### ✔ Work in **batch jobs**:
```
sbatch Daddy/cluster/slurm_gpu.job
```

### ✔ Allow **custom conda environments** stored in user's home directory  
(Required due to ENG cluster quota system)  
Use in your `.bashrc` or per-job scripts:

```bash
export CONDA_ENVS_DIRS="$HOME/conda_envs/"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs/"
```

### ✔ Run inside **Academic Jupyter GPU Notebook**  
(Where dependencies are installed with `pip install --user`)

### ✔ Respect ENG cluster job limits:
- Max GPU RAM: **20GB**
- Max time: **4 hours per job**
- Jobs per user: **4**
- Max CPUs per job: **8**

---

# 4. SLURM Job Templates

## 4.1 GPU Job — `Daddy/cluster/slurm_gpu.job`

```bash
#!/bin/bash
#SBATCH -p gpu-linuxlab
#SBATCH -G 1
#SBATCH --gres=gpumem:10G
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH -J slim-dqn-gpu
#SBATCH -o slim_gpu_%j.out
#SBATCH -e slim_gpu_%j.err

export CONDA_ENVS_DIRS="$HOME/conda_envs/"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs/"
source ~/.bashrc
conda activate pokemon_slim

python -m Daddy.train_slim --config configs/slim.yaml
```

---

## 4.2 CPU Job — `Daddy/cluster/slurm_cpu.job`

```bash
#!/bin/bash
#SBATCH -p linuxlab
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH -J slim-dqn-cpu
#SBATCH -o slim_cpu_%j.out
#SBATCH -e slim_cpu_%j.err

export CONDA_ENVS_DIRS="$HOME/conda_envs/"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs/"
source ~/.bashrc
conda activate pokemon_slim_cpu

python -m Daddy.train_slim --config configs/slim_cpu.yaml
```

---

# 5. Environment Setup Script — `Daddy/cluster/env_setup.sh`

```bash
#!/bin/bash

export CONDA_ENVS_DIRS="$HOME/conda_envs/"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs/"

module load anaconda
conda create -y -n pokemon_slim python=3.10
conda activate pokemon_slim

pip install torch torchvision
pip install gym numpy pillow matplotlib tqdm
pip install opencv-python imageio wandb

echo "Environment ready."
```

---

# 6. Codex / Copilot Instruction Block (Cluster‑Aware)

Paste the following into Copilot Chat **inside a file inside `Daddy/`**:

```
Read Daddy/DESIGN_SLIM_MODEL.md and this README.
Treat them as the complete specification for the new slim Pokémon Red RL agent.

Rules:
1. DO NOT modify pokemonred_puffer/ or Final/epsilon/.
2. All new code must live only in Daddy/.
3. All modules must be cluster-safe: 
   - Avoid popup windows
   - Allow --no-video or headless modes
   - Save all logs & videos locally
4. Add full support for SLURM execution, conda activation, and GPU RAM requests.

Start by generating:
- Daddy/networks.py (slim CNN + GRU/LSTM + Q-head)
- Daddy/agent.py (SlimDQN with WRAM flags, RND hooks, and recurrent state)
```

Then run follow-ups:

```
Implement rnd.py following the spec and ensure it's lightweight enough for GPU RAM limits.
```

```
Implement flags.py to decode WRAM flags from the puffer env and prepare cluster-safe feature tensors.
```

```
Implement train_slim.py so it can run in SLURM and qlogin sessions without modification.
```

---

# 7. Cluster Run Instructions

## 7.1 Interactive GPU Session  
(from login node)

```
ssh washukey@shell.engr.wustl.edu
qlogin-gpu -g 10
conda activate pokemon_slim
python -m Daddy.train_slim
```

## 7.2 Submit GPU Batch Job

```
sbatch Daddy/cluster/slurm_gpu.job
```

## 7.3 Monitor Jobs

```
squeue -u $USER
```

---

# 8. Debugging on the Cluster

Debug rollout:

```
python Daddy/debug_rollout.py --steps 200
```

Verify WRAM flag decoding:

```
python Daddy/debug_rollout.py --flags
```

Render a short video (cluster-safe):

```
python Daddy/debug_rollout.py --video --max-frames 500
```

---

# 9. GitHub Setup

Initialize repo:

```bash
git init
git add .
git commit -m "Initial slim agent + cluster support"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Now your slim agent is fully cluster-ready and GitHub-ready.

---
