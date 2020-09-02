#!/bin/bash
#SBATCH --output=/scratch/lloyd/job_logs/example/job-%j.out
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --qos=dinesh-high
#SBATCH --partition=dinesh-compute

hostname
nvidia-smi


python test_viz.py --data_dir /scratch/lloyd/data/fetch_reach_25 --no_first --keyp_pred --num_steps 150 --pretrained_path exp_data/demo_lloyd_s_0/2020-09-01-19-44-28/checkpoints --vids_path demo_lloyd --num_keypoints 16 --ckpt 25
#python test_viz.py --data_dir data/fetch_reach_25 --pretrained_path /home/lloyd/git-repos/unsup_keyp_torch/exp_data/lloyd_train_5/20/checkpoints --vids_path vids/lloyd_train_20 --actiion_dim 6 --ckpt 550



exit 0
