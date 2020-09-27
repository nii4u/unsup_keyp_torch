#!/bin/bash
#SBATCH --output=/scratch/lloyd/job_logs/example/job-%j.out
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=11G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=80:00:00
#SBATCH --qos=dinesh-high
#SBATCH --partition=dinesh-compute
#SBATCH --array=0
args="$*"
nvidia-smi -L



source /scratch/srinath/venvs/tf1-dev/bin/activate
cd ~/git-repos/unsup_keyp_torch/

python train_keyp_inverse_forward.py --heatmap_reg 5e-2 --num_keypoints 16 --seed $(($SLURM_ARRAY_TASK_ID * 5)) $args

PYTHON_EXIT_CODE=$?

exit $PYTHON_EXIT_CODE
