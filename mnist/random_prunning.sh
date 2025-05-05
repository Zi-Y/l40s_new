#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/mnist/mnist_random_pruning_singlePR.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0

# Loop through pruning_rate values from 0.1 to 0.9, excluding 0.6
for PRUNING_RATE in $(seq 0.0 0.1 0.9)
do
#    if [[ "$PRUNING_RATE" == "0.6" ]]; then
#        continue
#    fi

    echo "Running with pruning_rate=${PRUNING_RATE} on GPU=${GPU_ID}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT_PATH" --wandb_project m-classification-cnn --seeds 50 --pruning_rate "$PRUNING_RATE" &

    # Update GPU_ID for the next process
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
done

# Wait for all background processes to complete
wait
echo "All tasks completed."
