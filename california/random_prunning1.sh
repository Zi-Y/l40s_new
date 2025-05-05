#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/california/california_random_pruning_singlePR.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0

#PRUNING_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.7 0.9)
PRUNING_RATES=(0.3 0.5 0.7 0.9)

for PRUNING_RATE in "${PRUNING_RATES[@]}"
do
    echo "Running with pruning_rate=${PRUNING_RATE} on GPU=${GPU_ID}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT_PATH" --wandb_project california_housing_pruning --wandb_name random --seeds 500 --pruning_rate "$PRUNING_RATE" &

    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
done


wait
echo "All tasks completed."
