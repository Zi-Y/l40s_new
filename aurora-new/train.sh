#!/bin/bash

# Initialize conda:
wandb login xxx
export OMP_NUM_THREADS=4

output_dir="./trained_models_on_cerra/Aurora_large"
mkdir -p "$output_dir"

# Set variable to choose between with TP or without TP (true for with TP, false for without TP)
WITH_TP=false

if [ "$WITH_TP" = true ]; then
    MODEL="standard_Aurora_with_tp"
    VARIABLE_WEIGHTS="finetuning_with_tp"
    DATASET="cerra_with_tp"
else
    MODEL="standard_Aurora"
    VARIABLE_WEIGHTS="finetuning"
    DATASET="cerra"
fi

echo "Running with model.patch_size=8..."
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=25793 main.py \
    task.distributed=True \
    task.model_name=Aurora \
    task.phase=finetuning \
    task.total_steps=100000 \
    task.rollout_steps=1 \
    model=$MODEL \
    variable_weights=$VARIABLE_WEIGHTS \
    optimizer.weight_decay=0 \
    optimizer.constant_lr=2e-4 \
    lr_scheduler.warmup_steps=1000 \
    dataset=$DATASET \
    dataset.common.use_dummy_slt=False \
    dataset.common.data_path=xxx \
    dataloader.num_workers=4 \
    logging.project_name=aurora_large_experiments_full_cerra \
    checkpoint.ckpt_dir="$output_dir"