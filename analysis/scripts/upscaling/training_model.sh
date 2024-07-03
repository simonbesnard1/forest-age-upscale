#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=8-00:00:00

# Set name of job shown in squeue
#SBATCH --job-name training_model
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/training_model_%A.out
#SBATCH --error=/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/training_model_%A.err

# Request CPU resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20

# Memory usage
#SBATCH --mem=50G

# Email notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=fail

CONDA_BASE=$(conda info --base) 
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate age_upscale
python /home/besnard/projects/forest-age-upscale/scripts/upscaling/training_model.py

exit_status=$?

# Define output and error file paths
output_file="/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/training_model_${SLURM_ARRAY_JOB_ID}.out"
error_file="/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/training_model_${SLURM_ARRAY_JOB_ID}}.err"

# Delete output and error files if the job was successful
if [ $exit_status -eq 0 ]; then
    rm -f $output_file $error_file
fi

exit
