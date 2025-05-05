#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/california/california_data_pruning_singlePR.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0


# Define the specific pruning_rate values
PRUNING_RATES=(-0.3 -0.5 -0.7 -0.9 0.3 0.5 0.7 0.9)
for PRUNING_RATE in "${PRUNING_RATES[@]}"
#for PRUNING_RATE in $(seq -0.9 0.1 0.9)
do
    # Skip PRUNING_RATE=0
    if [[ "$PRUNING_RATE" == "0.0" ]]; then
        continue
    fi

    echo "Running with pruning_rate=${PRUNING_RATE} on GPU=${GPU_ID}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python "$SCRIPT_PATH" \
        --iterations 1000 \
        --pruning_method 0 \
        --wandb_name avg_loss \
        --wandb_project california_housing_pruning \
        --seeds 50 \
        --pruning_rate "$PRUNING_RATE" &

    # Update GPU_ID for the next process
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
done

# Wait for all background processes to complete
wait
echo "All tasks completed."