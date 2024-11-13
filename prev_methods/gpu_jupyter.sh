#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1

#source ~/.bashrc
#module purge
#module load python3/intel/3.8.6
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11

#port=$(shuf -i 10000-65500 -n 1)
port=8881
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

jupyter lab --no-browser --port $port --notebook-dir /scratch/ab9738/dfdl_imputation --NotebookApp.token='' --NotebookApp.password=''