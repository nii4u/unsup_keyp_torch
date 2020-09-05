#!/bin/bash
#SBATCH --output=/scratch/lloyd/job_logs/example/job-%j.out
#SBATCH --mem-per-gpu=11G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --qos=dinesh-high
#SBATCH --partition=dinesh-compute

args="$*"

source /scratch/srinath/venvs/tf1-dev/bin/activate
cd ~/git-repos/unsup_keyp_torch/

python test_viz.py $args
PYTHON_EXIT_CODE=$?

exit $PYTHON_EXIT_CODE
