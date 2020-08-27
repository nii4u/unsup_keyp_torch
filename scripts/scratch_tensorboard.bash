#!/bin/bash
#SBATCH --output=/scratch/lloyd/git-repo/unsup_keyp_torch/job_logs/tb/job-tensorboard-%j.out
#SBATCH --qos=viz
#SBATCH --partition=viz
#SBATCH --cores=1

PORT_MAP=/tmp/tensorboard_port_map

TB_PORT=$(cat $PORT_MAP | grep "$SLURM_JOBID," | cut -d',' -f2)
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=`echo $@ |  sed s/" "/,/g | cut -d "," -f1-`

echo "Go to http://$IP_ADDRESS:$TB_PORT"
echo $TB_FOLDER

source /scratch/lloyd/venvs/tf1-dev/bin/activate
tensorboard --logdir $TB_FOLDER --port $TB_PORT

