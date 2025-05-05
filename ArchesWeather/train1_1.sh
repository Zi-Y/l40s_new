#!/bin/bash

# Initialize conda:
wandb login 4329d2d253401e552885447250c4068faf9a99a9
# 设置环境变量
export OMP_NUM_THREADS=4

# 定义 patch_size 列表
patch_sizes=(3)

# 循环运行指定的 patch_size
for patch_size in "${patch_sizes[@]}"
do
    # 创建新的文件夹，文件夹名称为 Aurora_training_Patch_size<patch_size>
    output_dir="/mnt/ssd/zi/4xl40s_ag_2_graphcast_seed${patch_size}"
    mkdir -p "$output_dir"

    # 运行程序
    echo "Running with seed=$patch_size..."
    python train_hydra.py \
        cluster=example-slurm \
        module=4xl40s_ag_1_graphcast_seed3 \
        dataloader=era5-w \
        module.seed=$patch_size \
        module.accumulate_grad_batches=2 \
        module.path_save_base="$output_dir" \
        max_steps=320000 \

    echo "Finished running with model.patch_size=$patch_size."
done

echo "All runs completed."