#!/bin/bash

# 定义 GPU 的索引数组
gpus=(0 1 2 3)

# 定义 subset_rate 数组
subset_rates=(0.3 0.5 0.7 0.9)

# 定义 pruning_methods 数组
#pruning_methods=(-1 2)
pruning_methods=(3)

# 定义其他固定参数
DATA_PATH="./data"
DATASET="cifar100"
ARCH="resnet18"
EPOCHS=200
LEARNING_RATE=0.1
BATCH_SIZE=128
SAVE_PATH="./checkpoint/pruned-dataset"

# 自定义 xx 和 yy 的值
WIN_XX="win10"  # 替换为你希望的值
EP_YY="ep30"    # 替换为你希望的值

# 动态生成 MASK_PATH 和 SCORE_PATH
MASK_PATH="./checkpoint/generated_mask/data_mask_${WIN_XX}_${EP_YY}.npy"
SCORE_PATH="./checkpoint/generated_mask/score_${WIN_XX}_${EP_YY}.npy"

# 遍历 pruning_methods、subset_rate 和 manualSeed，并为每个 GPU 分配一个任务
for method in ${pruning_methods[@]}; do
  for seed in {42..45}; do
    for i in ${!subset_rates[@]}; do
      SUBSET_RATE=${subset_rates[$i]}
      GPU=${gpus[$i]}
      echo "Running pruning_method=${method}, subset_rate=${SUBSET_RATE}, manualSeed=${seed} on GPU ${GPU}"
      CUDA_VISIBLE_DEVICES=${GPU} nohup \
      python train_subset.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --batch-size ${BATCH_SIZE} \
        --save_path ${SAVE_PATH} \
        --subset_rate ${SUBSET_RATE} \
        --mask_path ${MASK_PATH} \
        --score_path ${SCORE_PATH} \
        --pruning_methods ${method} \
        --manualSeed ${seed} > output_method_${method}_subset_${SUBSET_RATE}_seed_${seed}.log 2>&1 &
    done
  done
done
# 等待所有后台任务完成
wait
# 最终提示
echo "All tasks submitted."
