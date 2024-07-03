#!/bin/bash

#SBATCH --output=/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/batch_upscaling_calc_%j.out
#SBATCH --error=/home/besnard/projects/forest-age-upscale/scripts/upscaling/log/batch_upscaling_calc_%j.err
#SBATCH --time=10-00:00:00

# Function to check if a job array has completed
check_jobs_completed() {
    local job_id=$1
    while true; do
        # Check if the job is still in the queue
        if ! squeue -j $job_id > /dev/null 2>&1; then
            # Job is no longer in the queue, it has either completed or failed
            return 0
        fi
        sleep 60 # Wait for 60 seconds before checking again
    done
}


# Total number of jobs
TOTAL_JOBS=5202
MAX_JOBS_PER_BATCH=1000

# Path to the SLURM submission script
JOB_SCRIPT="/home/besnard/projects/forest-age-upscale/scripts/upscaling/run_upscaling_100m.sh"

# Extract directory from JOB_SCRIPT path
SCRIPT_DIR=$(dirname "$JOB_SCRIPT")

# Submit jobs in batches
for (( i=0; i<TOTAL_JOBS; i+=MAX_JOBS_PER_BATCH )); do
    END=$((i+MAX_JOBS_PER_BATCH-1))
    if [ $END -ge $TOTAL_JOBS ]; then
        END=$((TOTAL_JOBS-1))
    fi
    
    # Modify the job script for the current batch
    TEMP_SCRIPT="$SCRIPT_DIR/temp_job_script_$i.sh"
    sed "s/#SBATCH --array=.*/#SBATCH --array=$i-$END%20/" "$JOB_SCRIPT" > "$TEMP_SCRIPT"

    # Submit the job and capture the job ID
    job_id=$(sbatch "$TEMP_SCRIPT" | awk '{print $4}')

    # Wait for the job array to complete
    check_jobs_completed $job_id

    # Clean up the temporary script for this batch
    rm "$TEMP_SCRIPT"
done


