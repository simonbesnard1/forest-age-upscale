#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=5-00:00:00

# Set name of job shown in squeue
#SBATCH --job-name xval-forest-age-MLP

# Request CPU resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80

# Memory usage
#SBATCH --mem=50GB

# Email notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=fail

#Output job
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/cross_validation/log/%j_MLP.out

CONDA_BASE=$(conda info --base) 
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate age_upscale
python /home/besnard/projects/forest-age-upscale/scripts/cross_validation/run_xval_study_MLP.py

exit
