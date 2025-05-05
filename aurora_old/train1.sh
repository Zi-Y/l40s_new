#!/bin/bash

wandb login 4329d2d253401e552885447250c4068faf9a99a9

export OMP_NUM_THREADS=4

patch_sizes=(4 6 8 16)

for patch_size in "${patch_sizes[@]}"
do
    output_dir="/mnt/ssd/trained_models_on_cerra/Aurora_training_Patch_size_${patch_size}"
    mkdir -p "$output_dir"

    echo "Running with model.patch_size=$patch_size..."
    torchrun --nnodes=1 --nproc_per_node=4 main.py \
        task.distributed=True \
        dataset=cerra_debug \
        task.phase=finetuning \
        dataloader.num_workers=4 \
        model.patch_size=$patch_size \
        optimizer.weight_decay=0 \
        optimizer.constant_lr=2e-4 \
        lr_scheduler.warmup_steps=1000 \
        logging.project_name=aurora_experiments_patch_size \
        task.total_steps=30000 \
        checkpoint.ckpt_epoch=1 \
        checkpoint.ckpt_dir="$output_dir"

    echo "Finished running with model.patch_size=$patch_size."
done

echo "All runs completed."