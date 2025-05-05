#!/bin/bash

MAX_TASKS_PER_GPU=3

# 定义可用的GPU列表，根据实际情况修改
#GPUS=(0 1 2 3)
GPUS=(0 1 2 3)

# 为每个GPU初始化一个数组存放其正在运行的任务pid
for gpu in "${GPUS[@]}"; do
    eval "pids_$gpu=()"
done

# 遍历 seed (0~50)，pruning_rate (0.0, 0.1, 0.3, 0.5, 0.7, 0.9) 和 pruning_method (0,1,2,3) 的所有组合
for seed in {0..50}; do
#    for pruning_rate in 0.0 0.1 0.3 0.5 0.7 0.9; do
    for pruning_rate in 0.0 0.1 0.3 0.5 0.7 0.9; do
#        for pruning_method in 0 1 2 3; do
        for pruning_method in 0 4; do
            # 新增条件：如果 pruning_rate 为 0.0 且 pruning_method 不为 0，则跳过该组合
            if [ "$pruning_rate" = "0.0" ] && [ "$pruning_method" != "0" ]; then
                continue
            fi

            if [ "$pruning_rate" != "0.0" ] && [ "$pruning_method" = "0" ]; then
                continue
            fi

            # 等待直到至少有一个GPU的正在运行任务数少于 MAX_TASKS_PER_GPU
            while true; do
                available_gpu=""
                for gpu in "${GPUS[@]}"; do
                    eval "arr=(\"\${pids_$gpu[@]}\")"
                    new_arr=()
                    # 清理已结束的任务
                    for pid in "${arr[@]}"; do
                        if kill -0 "$pid" 2>/dev/null; then
                            new_arr+=("$pid")
                        fi
                    done
                    # 更新该GPU的pid数组
                    eval "pids_$gpu=(\"\${new_arr[@]}\")"
                    # 检查该GPU是否有空余资源
                    if [ "${#new_arr[@]}" -lt "$MAX_TASKS_PER_GPU" ]; then
                        available_gpu=$gpu
                        break
                    fi
                done
                if [ -n "$available_gpu" ]; then
                    break
                fi
                sleep 2
            done

            # 指定当前GPU
            export CUDA_VISIBLE_DEVICES=$available_gpu
            echo "启动任务：GPU $available_gpu | seed $seed | pruning_rate $pruning_rate | pruning_method $pruning_method"
            /home/zi/miniconda3/bin/conda run -n regression python /home/zi/research_project/iTransformer/run.py \
                --is_training 1 \
                --root_path /mnt/ssd/iTransformer_datasets/weather/ \
                --data_path weather.csv \
                --model_id weather_96_96 \
                --model iTransformer \
                --data custom \
                --features M \
                --seq_len 96 \
                --pred_len 96 \
                --e_layers 3 \
                --enc_in 21 \
                --dec_in 21 \
                --c_out 21 \
                --d_model 512\
                --d_ff 512\
                --itr 1 \
                --num_workers 8 \
                --train_iterations 12000 \
                --pruning_method ${pruning_method} \
                --pruning_rate ${pruning_rate} \
                --seed ${seed} &

            # 记录该GPU运行的任务PID
            pid=$!
            eval "pids_$available_gpu+=(\"$pid\")"
            sleep 5
        done
    done
done

# 等待所有后台任务结束
wait
echo "所有任务执行完毕！"