#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=00-02:00:00

# Set name of job shown in squeue
#SBATCH --job-name management_calc
#SBATCH --array=0-99%1
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/management_partition/log/management_calc_%A_%a.out
#SBATCH --error=/home/besnard/projects/forest-age-upscale/scripts/management_partition/log/management_calc_%A_%a.err

# Request CPU resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Memory usage
#SBATCH --mem=10G

# Email notifications
#SBATCH --mail-type=END
#SBATCH --mail-type=fail

CONDA_BASE=$(conda info --base) 
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate age_upscale
python /home/besnard/projects/forest-age-upscale/scripts/management_partition/management_calc.py $SLURM_ARRAY_TASK_ID

exit_status=$?

# Define output and error file paths
output_file="/home/besnard/projects/forest-age-upscale/scripts/management_partition/log/management_calc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
error_file="/home/besnard/projects/forest-age-upscale/scripts/management_partition/log/management_calc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Delete output and error files if the job was successful
if [ $exit_status -eq 0 ]; then
    rm -f $output_file $error_file
fi

exit
