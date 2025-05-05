#!/bin/bash

# Initialize conda:
wandb login 4329d2d253401e552885447250c4068faf9a99a9
# 设置环境变量
export OMP_NUM_THREADS=4

output_dir="/mnt/ssd/zi/4xl40s_ag_1_graphcast_seed3_new"
mkdir -p "$output_dir"

# 运行程序
python train_hydra.py \
    cluster=example-slurm \
    module=4xl40s_ag_1_graphcast_seed3 \
    dataloader=era5-w \
    module.seed=3 \
    module.path_save_base="$output_dir" \
    max_steps=320000 \


# 定义 patch_size 列表
patch_sizes=(0.7 0.5 0.3)

# 循环运行指定的 patch_size
for patch_size in "${patch_sizes[@]}"
do
    new_value=$(awk "BEGIN {print $patch_size * 100}")
    # 创建新的文件夹，文件夹名称为 Aurora_training_Patch_size<patch_size>
    output_dir="/mnt/ssd/zi/4xl40s_ag_1_graphcast_seed3_keep_${new_value}"
    mkdir -p "$output_dir"

    # 运行程序
    echo "Running with seed=$patch_size..."
    python train_hydra.py \
        cluster=example-slurm \
        module=4xl40s_ag_1_graphcast_seed3 \
        dataloader=era5-w \
        module.seed=3 \
        module.use_infobatch=True \
        module.path_save_base="$output_dir" \
        module.info_batch.prune_easy=-50 \
        module.info_batch.info_batch_ratio=$patch_size \
        max_steps=320000 \

    echo "Finished running with model.patch_size=$patch_size."
done

echo "All runs completed."