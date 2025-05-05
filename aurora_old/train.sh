#!/bin/bash -eux
#SBATCH --job-name=aur
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH -p sorcery
#SBATCH -A meinel-mlai

# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate aurora
set +eu
which python
nvidia-smi
torchrun --nnodes=1 --nproc_per_node=2 main.py 
