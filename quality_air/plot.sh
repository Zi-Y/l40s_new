#!/bin/bash

# 定义 pruning_method 的所有值
#pruning_methods=(-1 1 3 4 101 103 104)
pruning_methods=(1 3 4 101 103 104 201 203 204 301 303 304)

# 遍历所有 pruning_method 并并行运行命令
for method in "${pruning_methods[@]}"; do
    echo "Running with pruning_method=$method"
    /home/local/zi/miniconda3/bin/conda run -n regression --no-capture-output \
        python /home/local/zi/research_project/quality_air/air_data_distribution_train.py --pruning_method "$method" &
done

# 等待所有后台进程完成
wait
