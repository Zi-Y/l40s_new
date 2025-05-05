#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/california/california_data_pruning_singlePR_new.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0

# Define the specific pruning_rate values
PRUNING_RATES=(-0.3 -0.5 -0.7 -0.9 0.3 0.5 0.7 0.9)
#PRUNING_RATES=(0.3 0.5 0.7 0.9)

# Loop through pruning methods
for PRUNING_METHOD in 0 2 3 4
do
    # Loop through pruning rates
    for PRUNING_RATE in "${PRUNING_RATES[@]}"
    do
        # Update wandb_name based on PRUNING_METHOD
        if [ "$PRUNING_METHOD" -eq 0 ]; then
            WANDB_NAME="avg_loss"
        elif [ "$PRUNING_METHOD" -eq 2 ]; then
            WANDB_NAME="S2L"
        elif [ "$PRUNING_METHOD" -eq 3 ]; then
            WANDB_NAME="avg_loss_NoWeight"
        elif [ "$PRUNING_METHOD" -eq 4 ]; then
            WANDB_NAME="S2L_NoWeight"
        else
            WANDB_NAME="error"
        fi

        echo "Running with pruning_method=${PRUNING_METHOD}, pruning_rate=${PRUNING_RATE}, wandb_name=${WANDB_NAME}, on GPU=${GPU_ID}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python "$SCRIPT_PATH" \
            --iterations 1000 \
            --pruning_method "$PRUNING_METHOD" \
            --wandb_name "$WANDB_NAME" \
            --wandb_project california_housing_pruning_full \
            --seeds 500 \
            --pruning_rate "$PRUNING_RATE" &

        # Update GPU_ID for the next process
        GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
    done
done

# Wait for all background processes to complete
wait
echo "All tasks completed."
#/home/local/zi/miniconda