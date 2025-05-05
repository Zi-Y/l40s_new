#!/usr/bin/env bash

# 进入脚本发生任何错误时，及时退出
set -e

# 指定你的 Python 脚本路径
PYTHON_SCRIPT="/home/zi/research_project/aurora/main.py"

# 通用参数设置（与每个进程都相同的参数）
COMMON_ARGS=" \
    task.distributed=False \
    task.task=inference \
    task.model_name=Aurora \
    dataloader.num_workers=4 \
    model.patch_size=12 \
    logging.project_name=aurora_test \
    task.total_steps=50000000 \
    logging.use_wandb=False \
    checkpoint.ckpt_dir=/mnt/ssd/zi/Aurora_large_training_Patch_size_8_lr_1e-4_Era5_normalization \
    checkpoint.ckpt_file=aurora-Aurora-finetuning-58-68788.ckpt \
"

# 分别指定 4 个不同的 dataset
DATASETS=("cerra_infer1" "cerra_infer2" "cerra_infer3" "cerra_infer4")

# 在 4 个 GPU 上并行执行
for i in {0..3}
do
  CUDA_VISIBLE_DEVICES=$i nohup \
    python ${PYTHON_SCRIPT} ${COMMON_ARGS} dataset=${DATASETS[$i]} \
    > "infer_gpu_${i}.log" 2>&1 &
  echo "Started inference on GPU $i with dataset=${DATASETS[$i]}"
done

# 等待所有子进程结束
wait
echo "All inference processes have completed."