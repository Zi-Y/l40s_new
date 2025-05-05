#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/mnist/mnist_data_pruning_singlePR.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0

# Define the specific pruning_rate values
#PRUNING_RATES=(-0.3 -0.5 -0.7 -0.9 0.3 0.5 0.7 0.9)
#PRUNING_RATES=(0.3 0.5 0.7 0.9)
PRUNING_RATES=(0.3)

# Loop through pruning rates
for PRUNING_RATE in "${PRUNING_RATES[@]}"
do
    # Loop through metric_seed from 0 to 100
    for METRIC_SEED in {0..150}
    do
        PRUNING_METHOD=0
        WANDB_NAME="avg_loss_each_seed"

        echo "Running with pruning_method=${PRUNING_METHOD}, pruning_rate=${PRUNING_RATE}, metric_seed=${METRIC_SEED}, wandb_name=${WANDB_NAME}, on GPU=${GPU_ID}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python "$SCRIPT_PATH" \
            --iterations 500 \
            --pruning_method "$PRUNING_METHOD" \
            --wandb_name "$WANDB_NAME" \
            --wandb_project m-classification-test \
            --cal_metric_for_each_seed 1 \
            --seeds 1 \
            --pruning_rate "$PRUNING_RATE" \
            --metric_seed "$METRIC_SEED" &

        # Update GPU_ID for the next process
        GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
    done
done

# Wait for all background processes to complete
wait
echo "All tasks completed."