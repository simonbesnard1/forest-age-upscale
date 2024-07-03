#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=03-00:00:00

# Set name of job shown in squeue
#SBATCH --job-name create_age_agb_management_data
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/management_analysis/log/create_age_agb_management_data_%A_%a.out
#SBATCH --error=/home/besnard/projects/forest-age-upscale/scripts/management_analysis/log/create_age_agb_management_data_%A_%a.err

# Request CPU resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Memory usage
#SBATCH --mem=500G

# Email notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=fail

CONDA_BASE=$(conda info --base) 
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate age_upscale

# Define output and error file paths
output_file="/home/besnard/projects/forest-age-upscale/scripts/management_analysis/log/create_age_agb_management_data_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
error_file="/home/besnard/projects/forest-age-upscale/scripts/management_analysis/log/create_age_agb_management_data_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

python /home/besnard/projects/forest-age-upscale/scripts/management_analysis/create_age_agb_management_data.py

# Run the Python script
exit_status=$?

# Delete output and error files if the job was successful
if [ $exit_status -eq 0 ]; then
    rm -f $output_file $error_file
fi

exit $exit_status


