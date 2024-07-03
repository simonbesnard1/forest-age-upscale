#!/bin/bash

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=30-00:00:00

# Set name of job shown in squeue
#SBATCH --job-name xval-forest-age-autoML

# Request CPU resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20

# Memory usage
#SBATCH --mem=80G

# Email notifications
#SBATCH --mail-user=besnard@gfz-potsdam.de
#SBATCH --mail-type=END
#SBATCH --mail-type=fail

#Output job
#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/cross_validation/log/%j_run_xval_study_autoML.out

source activate age_upscale
python /home/besnard/projects/forest-age-upscale/scripts/cross_validation/run_xval_study_autoML.py

exit
