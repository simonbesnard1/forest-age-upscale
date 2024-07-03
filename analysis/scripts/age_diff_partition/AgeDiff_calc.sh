#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=06-00:00:00

# Set name of job shown in squeue
#SBATCH --job-name AgeDiff_calc
#SBATCH --array=1137-1139%3
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/age_diff_partition/log/AgeDiff_calc_%A_%a.out
#SBATCH --error=/home/besnard/projects/forest-age-upscale/scripts/age_diff_partition/log/AgeDiff_calc_%A_%a.err

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

#module use /cluster/eb/testing/modules/all/
#module load GDAL/3.5.0-foss-2022a

python /home/besnard/projects/forest-age-upscale/scripts/age_diff_partition/AgeDiff_calc.py $SLURM_ARRAY_TASK_ID

exit_status=$?

# Define output and error file paths
output_file="/home/besnard/projects/forest-age-upscale/scripts/age_diff_partition/log/AgeDiff_calc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
error_file="/home/besnard/projects/forest-age-upscale/scripts/age_diff_partition/log/AgeDiff_calc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Delete output and error files if the job was successful
if [ $exit_status -eq 0 ]; then
    rm -f $output_file $error_file
fi

exit
