#!/bin/bash

export CONDA_ENVS_DIRS="$HOME/conda_envs/"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs/"

module load anaconda
conda create -y -n pokemon_slim python=3.10
conda activate pokemon_slim

pip install torch torchvision
pip install gymnasium numpy pillow matplotlib tqdm
pip install opencv-python imageio wandb omegaconf

echo "Environment ready."
