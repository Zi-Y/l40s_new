#!/bin/bash

# Define the script path and the base command
SCRIPT_PATH="/home/zi/research_project/mnist/mnist_data_pruning_singlePR.py"

# Define the number of GPUs available
NUM_GPUS=4
GPU_ID=0

# Define the specific pruning_rate values
#PRUNING_RATES=(-0.3 -0.5 -0.7 -0.9 0.3 0.5 0.7 0.9)
PRUNING_RATES=(0.3 0.5 0.7 0.9)

# Loop through pruning methods
for PRUNING_METHOD in 0 2
do
    # Loop through pruning rates
    for PRUNING_RATE in "${PRUNING_RATES[@]}"
    do
        # Update pruning_rate if pruning_method is 2
        if [ "$PRUNING_METHOD" -eq 2 ]; then
            PRUNING_RATE="-$(echo $PRUNING_RATE | sed 's/^-//')" # Ensure the pruning rate is negative
            WANDB_NAME="TTDS_avg_metric"
        else
            WANDB_NAME="avg_loss_avg_metric"
        fi

        echo "Running with pruning_method=${PRUNING_METHOD}, pruning_rate=${PRUNING_RATE}, wandb_name=${WANDB_NAME}, on GPU=${GPU_ID}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python "$SCRIPT_PATH" \
            --iterations 1000 \
            --pruning_method "$PRUNING_METHOD" \
            --wandb_name "$WANDB_NAME" \
            --use_avg_metrics_with_number_seeds 100 \
            --wandb_project m-classification-cnn \
            --cal_metric_for_each_seed 0 \
            --seeds 20 \
            --pruning_rate "$PRUNING_RATE" &

        # Update GPU_ID for the next process
        GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
    done
done

# Wait for all background processes to complete
wait
echo "All tasks completed."
