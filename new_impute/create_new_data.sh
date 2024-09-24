#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=create_new_data
#SBATCH --mail-type=END
#SBATCH --mail-user=yz5944@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

RUNDIR=$SCRATCH/transfer_learning/run-${SLURM_JOB_ID}
mkdir -p $RUNDIR
cd $RUNDIR

source /home/${USER}/.bashrc
source activate bio2

export OMP_NUM_THREADS=12
time python create_new_data.py
