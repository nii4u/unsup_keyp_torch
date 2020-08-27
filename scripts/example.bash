#!/bin/bash
#SBATCH --output=/job_logs/job-%j.out
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --qos=dinesh-high
#SBATCH --partition=dinesh-compute

hostname
nvidia-smi

#python train_keyp_pred.py --data_dir data/fetch_reach_25 --exp_name demo_lloyd_thurs

python train.py --data_dir /scratch/lloyd/data/fetch_reach_25 --no_first --num_keypoints 16 --num_epochs 20 --exp_name lloyd_train_20_wed

#python test_viz.py --data_dir /scratch/lloyd/data/fetch_reach_25 --no_first --keyp_pred --timesteps 150 --pretrained_path exp_data/fetch_reach_16kp_track_s_0/2020-06-02-03-52-29/checkpoints --vids_path fetch_reach_25_16kp_TODAY --num_keypoints 16 --action_dim 4 --ckpt 600

#python train.py --data_dir /scratch/lloyd/data/fetch_reach_25 --num_epochs 5 --exp_name lloyd_train_5
#python test_viz.py --data_dir /scratch/lloyd/data/fetch_reach_25 --pretrained_path exp_data/lloyd_train_5/2020-08-26-12-20-41/checkpoints --vids_path outpu_video_1 --action_dim 4 --ckpt 5


#python test_viz.py --data_dir /scratch/lloyd/data/fetch_reach_25 --pretrained_path /home/lloyd/git-repos/unsup_keyp_torch/exp_data/lloyd_train_20/2020-08-25-17-06-15/checkpoints --vids_path vids/lloyd_train_20 --action_dim 4 
#python test_viz.py --data_dir /scratch/lloyd/data/fetch_reach_25 --no_first --keyp_pred --timesteps 150 --pretrained_path exp_data/lloyd_train_20/2020-08-25-17-06-15/checkpoints --vids_path vids/lloyd_train_20 --num_keypoints 16 --action_dim 4 --ckpt 20


exit 0
