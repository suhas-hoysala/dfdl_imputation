#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=denoise
#SBATCH --mail-type=END
#SBATCH --mail-user=yz5944@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load python/intel/3.10.14

RUNDIR=$SCRATCH/transfer_learning/run-${SLURM_JOB_ID}
mkdir -p $RUNDIR

cp $SCRATCH/${USER}/bio/dfdl_imputation/GENIE3/transfer_learning.py $RUNDIR/
cp -r $SCRATCH/${USER}/bio/dfdl_imputation/SERGIO/data_sets $RUNDIR/

cd $RUNDIR

source /home/${USER}/.bashrc
source activate bio2 

export OMP_NUM_THREADS=12

time python denoise.py

# Save the results
cp -r ./results $SCRATCH/yz5944/bio/dfdl_imputation/new_impute/results/run-${SLURM_JOB_ID}
