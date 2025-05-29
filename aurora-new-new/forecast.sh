#!/bin/bash

# Initialize conda:
export OMP_NUM_THREADS=4

# Set variable to choose between with TP or without TP (true for with TP, false for without TP)
WITH_TP=False

if [ "$WITH_TP" = true ]; then
    MODEL="standard_Aurora_with_tp"
    VARIABLE_WEIGHTS="finetuning_with_tp"
    DATASET="cerra_with_tp"
else
    MODEL="standard_Aurora"
    VARIABLE_WEIGHTS="finetuning"
    DATASET="cerra"
fi

# AISC cluster
DATASET_PATH="/data/shared/ekapex/cerra_full.zarr"
CHECKPOINT_PATH="/data/shared/ekapex/trained_models/Aurora_large_with_cerra_full_stats/aurora-Aurora-finetuning-step219000-best.ckpt"
OUTPUT_DIR="/data/shared/ekapex/forecasts/Aurora_large_with_cerra_full_stats/forecast_Aurora-step219000.zarr"

# Chair cluster A40
DATASET_PATH="/mnt/ssd/datasets/cerra_full_derived.zarr"
CHECKPOINT_PATH="/mnt/ssd/trained_aurora_models/Aurora-step219000.ckpt"
OUTPUT_DIR="/mnt/ssd/datasets/forecast_Aurora-step219000.zarr"

rm -rf "$OUTPUT_DIR"

echo "Running with model_name=$MODEL..."
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=25793 \
  main.py \
    task=forecast \
    task.distributed=False \
    task.model_name=Aurora \
    task.use_activation_checkpointing=False \
    task.use_torch_compile=False \
    task.checkpoint_path=$CHECKPOINT_PATH \
    task.output_dir=$OUTPUT_DIR \
    task.use_wb2_format=True \
    task.lead_times=[6] \
    model=$MODEL \
    model.patch_size=8 \
    model.use_lora=False \
    variable_weights=$VARIABLE_WEIGHTS \
    dataset=$DATASET \
    dataset.common.use_dummy_slt=False \
    dataset.common.data_path=$DATASET_PATH \
    dataset.val.start_time="2021-01-01T00:00:00" \
    dataset.val.end_time="2021-12-31T21:00:00" \
    dataloader.num_workers=4 \
    logging.use_wandb=False \
