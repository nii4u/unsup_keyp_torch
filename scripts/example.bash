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


python train.py --data_dir /scratch/lloyd/data/fetch_reach_25 --no_first --keyp_pred --num_epochs 625 --num_keypoints 16 --exp_name train_600_Epoch


exit 0
