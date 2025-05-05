#!/usr/bin/env bash

SCRIPT_PATH="/home/zi/research_project/mnist/mnist_data_pruning_singlePR.py"
NUM_GPUS=4
GPU_ID=0

# 无论何时都需要加 --pruning_rate
PRUNING_RATES=(0.3 0.5 0.7 0.9)
METHODS=(0 2)

# 统一封装执行逻辑
run_experiment() {
    local method="$1"
    local rate="$2"
    local wandb_name="$3"

    echo "Running with pruning_method=${method}, pruning_rate=${rate}, wandb_name=${wandb_name}, on GPU=${GPU_ID}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python "${SCRIPT_PATH}" \
        --iterations 500 \
        --pruning_method "${method}" \
        --pruning_rate "${rate}" \
        --wandb_name "${wandb_name}" \
        --use_avg_metrics_with_number_seeds 1 \
        --wandb_project m-classification-cnn \
        --cal_metric_for_each_seed 0 \
        --seeds 400 &

    # 更新 GPU_ID，以循环使用不同的 GPU
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
}

# 对三个不同的 pruning_method 分别进行测试
for method in "${METHODS[@]}"; do
    for rate in "${PRUNING_RATES[@]}"; do
        case $method in
            # pruning_method=2，需要将 rate 改为负数
            2)
                neg_rate="-${rate#-}"  # 去掉已有的 '-'，再加上 '-'
                run_experiment "$method" "$neg_rate" "TTDS_single_metric"
                ;;
            # pruning_method=-1，随机剪枝
            -1)
                run_experiment "$method" "$rate" "random_pruning"
                ;;
            # pruning_method=0 (或其它)，正常剪枝
            *)
                run_experiment "$method" "$rate" "avg_loss_single_metric"
                ;;
        esac
    done
done

# random pruning
PRUNING_RATES=(0.0 0.3 0.5 0.7 0.9)
METHODS=(-1)

# 对三个不同的 pruning_method 分别进行测试
for method in "${METHODS[@]}"; do
    for rate in "${PRUNING_RATES[@]}"; do
        case $method in
            # pruning_method=2，需要将 rate 改为负数
            2)
                neg_rate="-${rate#-}"  # 去掉已有的 '-'，再加上 '-'
                run_experiment "$method" "$neg_rate" "TTDS_single_metric"
                ;;
            # pruning_method=-1，随机剪枝
            -1)
                run_experiment "$method" "$rate" "random_pruning"
                ;;
            # pruning_method=0 (或其它)，正常剪枝
            *)
                run_experiment "$method" "$rate" "avg_loss_single_metric"
                ;;
        esac
    done
done

wait
echo "All tasks completed."