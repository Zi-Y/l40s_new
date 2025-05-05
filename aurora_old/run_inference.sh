#!/bin/bash

# 运行4个进程，分别使用4个GPU
for i in {1..4}; do
    export CUDA_VISIBLE_DEVICES=$((i-1))
    nohup /home/zi/miniconda3/bin/conda run -n aurora python /home/zi/research_project/aurora/main.py \
        task.distributed=False \
        dataset=cerra_infer${i} \
        task.task=inference \
        task.model_name=Aurora \
        dataloader.num_workers=4 \
        model.patch_size=12 \
        logging.project_name=aurora_test \
        task.total_steps=50000000 \
        logging.use_wandb=False \
        checkpoint.ckpt_dir=/mnt/ssd/trained_aurora_models_on_cerra/Aurora_large_training_Patch_size_12_lr_1e-4_Era5_normalization \
        checkpoint.ckpt_file=aurora-Aurora-finetuning-85-100000.ckpt \
        > log_cerra_infer${i}.txt 2>&1 &
done

echo "All processes started!"