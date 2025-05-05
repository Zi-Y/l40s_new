#!/bin/bash

# === 配置部分 ===
SCRIPT_PATH="/home/local/zi/research_project/quality_air/air_data_pruning_singlePR_new.py"  # 脚本路径
NUM_GPUS=4                     # GPU 数量
MAX_JOBS=20                   # 最大并发进程数

#PRUNING_RATES=(-0.3 -0.5 -0.7 -0.9 0.3 0.5 0.7 0.9)
PRUNING_RATES=(0.0)

count=0  # 用于计算当前是第几个任务，从而分配 GPU
for PRUNING_METHOD in -1
do
    for PRUNING_RATE in "${PRUNING_RATES[@]}"
    do
        # 这里将训练 seed 改为示例区间，请根据需要修改
        for TRAINING_SEED in {0..2000}
        do
            # 根据 PRUNING_METHOD 设置 WANDB_NAME
            if [ "$PRUNING_METHOD" -eq 0 ]; then
                WANDB_NAME="avg_loss"
            elif [ "$PRUNING_METHOD" -eq 2 ]; then
                WANDB_NAME="S2L"
            elif [ "$PRUNING_METHOD" -eq 3 ]; then
                WANDB_NAME="avg_loss_NoWeight"
            elif [ "$PRUNING_METHOD" -eq 4 ]; then
                WANDB_NAME="S2L_NoWeight"
            elif [ "$PRUNING_METHOD" -eq -1 ]; then
                WANDB_NAME="random"
            else
                WANDB_NAME="error"
            fi

            # 若后台进程已经达到/超过最大并发限制，则阻塞等待
            while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]
            do
                sleep 1
            done

            # 分配 GPU (轮流分配到 0,1,2,3)
            GPU_ID=$((count % NUM_GPUS))

            echo "Running with: pruning_method=${PRUNING_METHOD}, pruning_rate=${PRUNING_RATE}, training_seed=${TRAINING_SEED}, wandb_name=${WANDB_NAME}, GPU=${GPU_ID}"

            CUDA_VISIBLE_DEVICES="$GPU_ID" \
            python "$SCRIPT_PATH" \
                --iterations 1000 \
                --pruning_method "$PRUNING_METHOD" \
                --wandb_name "$WANDB_NAME" \
                --wandb_project air_quality_full \
                --seeds 1 \
                --pruning_rate "$PRUNING_RATE" \
                --training_seed "$TRAINING_SEED" &

            # 计数器 +1
            count=$((count+1))

        done
    done
done

# 等待所有后台进程结束
wait
echo "All tasks completed."