#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=create_new_data
#SBATCH --mail-type=END
#SBATCH --mail-user=yz5944@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load python/3.10.14

# Set up the run directory
RUNDIR=$SCRATCH/create_new_data/run-${SLURM_JOB_ID}
mkdir -p $RUNDIR

# # Copy the Python script and necessary data to the run directory
# cp $SCRATCH/${USER}/bio/dfdl_imputation/GENIE3/create_new_data.py $RUNDIR/
# cp -r $SCRATCH/${USER}/bio/dfdl_imputation/GENIE3/data_sets $RUNDIR/

# Change to the run directory
cd $RUNDIR

source /home/${USER}/.bashrc
source activate bio2 

time mpirun python create_new_data.py

# Save the results
cp -r ./results $SCRATCH/yz5944/bio/dfdl_imputation/new_impute/results/run-${SLURM_JOB_ID}
