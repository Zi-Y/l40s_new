#!/usr/bin/env bash

##########################################
# Usage: ./run.sh <gpu_list_string>
# e.g.   ./run.sh 0,1
#        ./run.sh 0,1,2,3
##########################################

## 如果不传 GPU 列表就提示错误并退出
#if [ $# -lt 1 ]; then
#  echo "Usage: $0 <gpu_list_string> (e.g. '0,1' or '0,1,2,3')"
#  exit 1
#fi
#
## 将命令行参数第 1 个当作 GPU 列表字符串，如 "0,1,2,3"
#gpu_list_string=$1
#
## 将 gpu_list_string 分割成数组 gpus
#IFS=',' read -r -a gpus <<< "$gpu_list_string"
#
## 判断数组长度是否在 [2,4] 之间
#num_gpus=${#gpus[@]}
#if [ "$num_gpus" -lt 2 ] || [ "$num_gpus" -gt 4 ]; then
#  echo "Error: The number of GPUs must be 2~4, but got: $num_gpus"
#  exit 1
#fi

# 在此示例中，演示用固定 gpus 列表
gpus=(0 1 2 3)
#gpus=(1 2 3)
#gpus=(0)
num_gpus=${#gpus[@]}
echo "We will use these GPUs: ${gpus[*]}"
echo "Number of GPUs: $num_gpus"

############################
# 1. 定义需要遍历的参数
############################
# 子集比例
# subset_rates=(0.3 0.5 0.7 0.9)
subset_rates=(0.0)

# 剪枝方法
# pruning_methods=(-1 2 3 4 5)
pruning_methods=(-1)

# 遍历的随机种子
#seeds=(42 43 44 45)
seeds=(52 53 54 55)
#seeds=(42 43)
#seeds=(46 47 48 49 50 51 52 53 57 58 59 60)
#seeds=(46 47 48 49 50 51 52 53)

# 固定参数
DATA_PATH="./data"
DATASET="cifar100"
ARCH="resnet18"
LEARNING_RATE=0.1
BATCH_SIZE=128
SAVE_PATH="./checkpoint/pruned-dataset"

# 自定义 xx 和 yy 的值
WIN_XX="win10"  # 替换为你希望的值
EP_YY="ep30"    # 替换为你希望的值

# 动态生成 MASK_PATH 和 SCORE_PATH（默认方法）
MASK_PATH="./checkpoint/generated_mask/data_mask_${WIN_XX}_${EP_YY}.npy"
SCORE_PATH="./checkpoint/generated_mask/score_${WIN_XX}_${EP_YY}.npy"

############################
# 2. 收集所有任务并计算“cost”
############################
# cost = 1 - subset_rate
task_lines=""

for method in "${pruning_methods[@]}"; do
  for seed in "${seeds[@]}"; do
    for srate in "${subset_rates[@]}"; do
      cost=$(awk -v sr="$srate" 'BEGIN{print 1 - sr}')
      # 每行: cost method seed srate
      task_lines+="${cost} ${method} ${seed} ${srate}\n"
    done
  done
done

############################
# 3. 任务按 cost 从大到小排序
############################
sorted_tasks=$(echo -e "$task_lines" | sort -k1,1nr)

############################
# 4. 用一个数组记录 GPU 负载
############################
gpu_loads=()
for ((i=0; i<"$num_gpus"; i++)); do
  gpu_loads+=(0)
done

# 找到最空闲 GPU 的函数
get_min_gpu() {
  local min_load=9999999
  local min_idx=0
  for i in "${!gpu_loads[@]}"; do
    cmp=$(echo "${gpu_loads[$i]} < $min_load" | bc -l)
    if [ "$cmp" -eq 1 ]; then
      min_load=${gpu_loads[$i]}
      min_idx=$i
    fi
  done
  echo "$min_idx"
}

############################
# 5. 分配任务 & 启动进程
############################
echo "===== Starting All Tasks with Balanced GPU Allocation ====="
echo "Total tasks:" $(echo -e "$sorted_tasks" | wc -l)
echo "-----------------------------------------------------------"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  cost=$(echo "$line" | awk '{print $1}')
  method=$(echo "$line" | awk '{print $2}')
  seed=$(echo "$line" | awk '{print $3}')
  srate=$(echo "$line" | awk '{print $4}')

  # 动态计算 epochs = 200 / (1 - subset_rate)，取整
  dynamic_epochs=$(awk -v sr="$srate" 'BEGIN {print int(200 / (1 - sr))}')

  # 找到最空闲 GPU
  gpu_idx=$(get_min_gpu)
  gpu_id=${gpus[$gpu_idx]}

  # 更新该 GPU 的负载
  new_load=$(echo "${gpu_loads[$gpu_idx]} + $cost" | bc -l)
  gpu_loads[$gpu_idx]=$new_load

  echo "分配任务: method=$method, subset_rate=$srate, seed=$seed, cost=$cost -> GPU=$gpu_id (负载=${new_load}), epochs=${dynamic_epochs}"

  # 当 pruning_method = 4 时，使用新的 MASK_PATH 与 SCORE_PATH
  if [ "$method" -eq 4 ]; then
    local_mask_path="./checkpoint/generated_mask/data_loss_mask_loss_${WIN_XX}_${EP_YY}.npy"
    local_score_path="./checkpoint/generated_mask/score_loss_loss_${WIN_XX}_${EP_YY}.npy"
  else
    local_mask_path="$MASK_PATH"
    local_score_path="$SCORE_PATH"
  fi

  # 启动训练脚本
  CUDA_VISIBLE_DEVICES="$gpu_id" nohup python train_subset.py \
    --data_path "${DATA_PATH}" \
    --dataset "${DATASET}" \
    --arch "${ARCH}" \
    --epochs "${dynamic_epochs}" \
    --learning_rate "${LEARNING_RATE}" \
    --batch-size "${BATCH_SIZE}" \
    --save_path "${SAVE_PATH}" \
    --subset_rate "${srate}" \
    --mask_path "${local_mask_path}" \
    --score_path "${local_score_path}" \
    --pruning_methods "${method}" \
    --manualSeed "${seed}" \
    > "output_method_${method}_subset_${srate}_seed_${seed}.log" 2>&1 &

done <<< "$sorted_tasks"

# 等待所有并行进程完成
wait

echo "All tasks have been submitted and completed."