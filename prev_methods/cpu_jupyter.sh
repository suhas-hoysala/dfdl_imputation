#!/bin/bash

#SBATCH --job-name=jupyter_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=9:00:00

#source ~/.bashrc
#module purge
#module load python3/intel/3.8.6

port=8881
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

jupyter lab --no-browser --port $port --notebook-dir /scratch/ab9738/dfdl_imputation --NotebookApp.token='' --NotebookApp.password=''