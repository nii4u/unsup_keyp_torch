#!/bin/bash
#SBATCH --output=/scratch/lloyd/job_logs/example/job-%j.out
#SBATCH --mem-per-gpu=11G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=80:00:00
#SBATCH --qos=dinesh-high
#SBATCH --partition=dinesh-compute

hostname
nvidia-smi

python train.py

exit 0
