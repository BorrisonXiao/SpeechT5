#!/bin/bash

# --- Configuration ---
START_STEP=30000
END_STEP=80000
INTERVAL=2000
VERSION="v1.4" # The version directory (e.g., 'exp/asr/v1.4/')
SLURM_TEMPLATE="./inference_asr.sh"
LOGS_DIR="logs/inference_asr"

# Ensure the logs directory exists
mkdir -p "$LOGS_DIR"

echo "Starting SLURM job submission sweep:"
echo "Version Directory: ${VERSION}"
echo "Step Range: ${START_STEP} to ${END_STEP} (Interval: ${INTERVAL})"
echo "--------------------------------------------------------"

CHECKPOINT_DIR="exp/asr/${VERSION}/"
# Loop through the steps
for ((STEP = ${START_STEP}; STEP <= ${END_STEP}; STEP += ${INTERVAL})); do

    # Find checkpoint file matching the step pattern
    checkpoint_file=$(find "$CHECKPOINT_DIR" -name "checkpoint_*_${STEP}.pt" -print -quit)

    if [[ -z "$checkpoint_file" ]]; then
        echo "WARNING: No checkpoint found for step ${STEP}. Skipping."
        continue
    fi

    # Extract base filename without extension for tag
    checkpoint_base=$(basename "$checkpoint_file" .pt)
    tag="${VERSION}-${checkpoint_base}"

    # Construct job name and output paths
    JOB_NAME="${VERSION}_${STEP}"
    OUTPUT_FILE="${LOGS_DIR}/${JOB_NAME}.out"
    # ERROR_FILE="${LOGS_DIR}/${JOB_NAME}.err"

    echo "Submitting job ${JOB_NAME} (Step: ${STEP})..."

    # Submit the job to SLURM
    # We export the dynamic STEP and VERSION variables for use in the template script
    sbatch --job-name="${JOB_NAME}" \
        --output="${OUTPUT_FILE}" \
        "${SLURM_TEMPLATE}"  --CHECKPOINT_PATH ${checkpoint_file} --tag ${tag}

    # Optional: Add a small sleep to avoid overwhelming the job scheduler
    sleep 0.2
done

echo "--------------------------------------------------------"
echo "Job submission complete. Check logs in the '${LOGS_DIR}' directory."
