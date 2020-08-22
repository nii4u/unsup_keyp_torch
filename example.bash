#!/bin/bash
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --qos=low
#SBATCH --partition=compute

hostname
nvidia-smi

#python train_keyp_pred.py --data_dir data/fetch_reach_25 --exp_name demo_lloyd_thurs

train.py --data_dir/fetech_reach --num_epochs 20 --exp_name lloyd_train_20
#python test_viz.py --data_dir data/fetch_reach_25 --no_first --keyp_pred --timesteps 150 --pretrained_path exp_data/fetch_reach_16kp_track_s_0/2020-06-02-03-52-29/checkpoints --vids_path fetch_reach_25_16kp_TODAY --num_keypoints 16 --action_dim 4 --ckpt 600

exit 0
