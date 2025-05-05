import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
import os
from datetime import datetime
import time

from omegaconf import OmegaConf
import signal

import numpy as np
import warnings
import matplotlib.pyplot as plt
from evaluation.deterministic_metrics import headline_wrmse
import re, os
from typing import List, Tuple
import json

warnings.filterwarnings('ignore')

all_results_np = 0
# /hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results
info_results_np = 0

for seed in range(1,7):
    saved_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
    file_names = f'3xA100_graphcast_seed{seed}_300000.txt'

    # read txt file
    file_path = saved_path + file_names
    # 创建一个列表来存储读取的数值
    values = []
    # 读取txt文件中的值
    with open(file_path, 'r') as f:
        for line in f:
            # 每行格式为 'key: value'，我们只提取value
            value = float(line.strip().split(': ')[1])
            values.append(value)

    # 将读取的值转换为NumPy数组
    numpy_array = np.array(values)

    # 将数组转换为1行多列的矩阵（根据原始数据的长度调整列数）
    numpy_matrix = numpy_array.reshape(1, -1)

    if seed == 1:
        all_results_np = numpy_matrix
    else:
        all_results_np = np.vstack((all_results_np, numpy_matrix))

all_results_np_mean = np.mean(all_results_np, axis=0)
print(all_results_np_mean)
all_results_np_std = np.std(all_results_np, axis=0)

for seed in range(1, 13):

    saved_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
    if seed == 1:
        file_names = f'3xA100_seed1_infobatch_297000.txt'
    elif seed == 2:
        file_names = f'3xA100_graphcast_seed2_infobatch_294000.txt'
    elif seed == 3:
        continue
        file_names = f'3xA100_graphcast_seed3_infobatch_200000.txt'
    elif seed == 4:
        file_names = f'3xA100_graphcast_seed3_infobatch_220000.txt'
    elif seed == 5:
        file_names = f'3xA100_graphcast_seed3_infobatch_240000.txt'
    elif seed == 6:
        file_names = f'3xA100_graphcast_seed3_infobatch_260000.txt'
    elif seed == 7:
        file_names = f'3xA100_graphcast_seed3_infobatch_280000.txt'
    elif seed == 8:
        file_names = f'3xA100_graphcast_seed3_infobatch_300000.txt'
    elif seed == 9:
        file_names = f'3xA100_graphcast_seed3_infobatch_320000.txt'
    elif seed == 10:
        file_names = f'3xA100_graphcast_seed3_infobatch_prune_easy_200000.txt'
    elif seed == 11:
        file_names = f'3xA100_graphcast_seed3_infobatch_prune_easy_220000.txt'
    elif seed == 12:
        file_names = f'3xA100_graphcast_seed3_infobatch_prune_easy_240000.txt'
    else:
        pass

    # read txt file
    file_path = saved_path + file_names
    # 创建一个列表来存储读取的数值
    values = []
    # 读取txt文件中的值
    with open(file_path, 'r') as f:
        for line in f:
            # 每行格式为 'key: value'，我们只提取value
            value = float(line.strip().split(': ')[1])
            values.append(value)

    # 将读取的值转换为NumPy数组
    numpy_array = np.array(values)

    # 将数组转换为1行多列的矩阵（根据原始数据的长度调整列数）
    numpy_matrix = numpy_array.reshape(1, -1)
    print(file_names)
    print(numpy_matrix[0,:6])
    print('\n')

    # if seed == 1:
    #     all_results_np = numpy_matrix
    # else:
    #     all_results_np = np.vstack((all_results_np, numpy_matrix))

print("all normal training results：")
all_results_np_mean = np.mean(all_results_np, axis=0)
print(all_results_np_mean)
all_results_np_std = np.std(all_results_np, axis=0)
print('test')




# x_vals_easy = []
# y_vals_easy = []
# x_vals_hard = []
# y_vals_hard = []


def read_values_from_file(file_path, variable):
    """
    从文件中读取指定变量的值。
    """
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            if variable in line:
                value = float(line.strip().split(': ')[1])
                values.append(value)
    return values


def collect_training_loss(saved_path: str, variable: str, x_limit_low: int, x_limit_high: int) -> Tuple[List[int], List[float]]:
    """
    根据文件名模式和变量提取数据。
    """
    parent_folder = os.path.basename(os.path.dirname(saved_path))

    # # for test
    # if '/' in saved_path:
    #     return [],[]

    saved_path = f'/hpi/fs00/share/ekapex/zi/{parent_folder}/infobatch_loss_values/'
    if '3xA100' in saved_path or '3xV100' in saved_path:
        num_device = 3
    elif '4xV100' in saved_path or '4xA100' in saved_path:
        num_device = 4
    else:
        raise ValueError("error in num_device")

    x_vals, y_vals = [], []
    total_num_epochs = len(os.listdir(saved_path)) // num_device
    # 最后的一个epoch没有遍历所有元素，因此不计入统计
    # total_num_epochs = total_num_epochs - 1

    '''
    first 4 columns are for surface variables
    the next 13*6 columns are for upper-air variables
    the same variable first,
    variable1_level1, variable1_level2, ..., variable1_leve13,
    variable2_level1, variable2_level2, ..., variable6_level13
    the -4 position saved the ID
    the -3 position saved the occurrence times
    the one before the last columns is global training step
    the last columns is for total weighted loss
    '''
    # variables = dict(
    #     state_level=['geopotential', 'u_component_of_wind', 'v_component_of_wind',
    #                  'temperature', 'specific_humidity', 'vertical_velocity'],
    #     state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind',
    #                    '2m_temperature', 'mean_sea_level_pressure'])
    #
    # pressure_levels = np.array([50, 100, 150, 200, 250,
    #                             300, 400, 500, 600, 700,
    #                             850, 925, 1000])
    # metrics = dict(
    #     T2m=surface_wrmse[..., 2, 0],
    #     SP=surface_wrmse[..., 3, 0],
    #     U10=surface_wrmse[..., 0, 0],
    #     V10=surface_wrmse[..., 1, 0],
    #     Z500=level_wrmse[..., 0, 7],
    #     T850=level_wrmse[..., 3, 10],
    #     Q700=1000 * level_wrmse[..., 4, 9],
    #     U850=level_wrmse[..., 1, 10],
    #     V850=level_wrmse[..., 2, 10])
    name_dict = dict(
        T2m=2,
        SP=3,
        U10=0,
        V10=1,
        Z500=4 + 7,
        T850=4 + 13*3 +10,
        Q700=4 + (4 * 13) + 9,
        U850=4 + (1 * 13) + 10,
        V850=4 + (2 * 13) + 10,
        loss=85)
    name_index = name_dict[variable]
    loss_epoch_np = 0
    for epoch_index, epoch in enumerate(range(0, total_num_epochs)):
        loss_epoch_np_cache = 0

        for device_id in range(num_device):
            if (device_id == 0) and (epoch_index == 0):
                loss_epoch_np = np.load(saved_path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy")[:, [84, name_index]]
                # Find the rows in matrix1 that are all zeros
                zero_rows = np.all(loss_epoch_np == 0, axis=1)
                loss_epoch_np = loss_epoch_np[~zero_rows]

            else:
                loss_epoch_np_cache = np.load(saved_path + "/Epoch_" + str(epoch) + "_device_" + str(device_id) + ".npy")[:, [84, name_index]]
                zero_rows = np.all(loss_epoch_np_cache == 0, axis=1)
                loss_epoch_np_cache = loss_epoch_np_cache[~zero_rows]

                loss_epoch_np = np.vstack((loss_epoch_np, loss_epoch_np_cache))

        if epoch % 20 == 0:
            print(f'parent_folder: {parent_folder}, epoch {epoch}')

    # 遍历每个 step 的范围
    step = 20000
    max_iteration = int(loss_epoch_np[:, 0].max())
    for start in range(0, max_iteration + 1, step):
        end = start + step - 1
        if x_limit_low <= end <= x_limit_high:
            mask = (loss_epoch_np[:, 0] >= start) & (loss_epoch_np[:, 0] <= end)  # 筛选出这一组的数据
            if np.any(mask):  # 如果这一组有数据
                avg_loss = loss_epoch_np[mask, 1].mean()
                x_vals.append(end+1)# 计算平均 loss
                # 计算平方根 与eval时计算方式一样
                y_vals.append(avg_loss ** 0.5)

    return x_vals, y_vals


def collect_data(saved_path: str, variable: str, x_limit_low: int, x_limit_high: int) -> Tuple[List[int], List[float]]:
    """
    根据文件名模式和变量提取数据。
    """
    parent_folder = os.path.basename(os.path.dirname(saved_path))

    file_dict_path = os.path.join(os.path.dirname(saved_path.rstrip('/')), 'saved_dict', parent_folder + '_' + variable + '.json')
    if os.path.exists(file_dict_path):
        with open(file_dict_path, 'r') as f:
            dict_results = json.load(f)

        return dict_results['x'], dict_results['y']
    else:
        x_vals, y_vals = [], []
        results_x, results_y = [], []
        matching_files = [f for f in os.listdir(saved_path) if f.endswith(".txt")]

        for filename in matching_files:
            file_path = os.path.join(saved_path, filename)
            match = re.search(r'_(\d+)\.txt$', filename)
            if match:
                x_val = int(match.group(1))
                if x_limit_low <= x_val <= x_limit_high:
                    values = read_values_from_file(file_path, variable)
                    if values:
                        x_vals.append(x_val)
                        y_vals.append(values[0])

        # 对数据进行排序
        sorted_pairs = sorted(zip(x_vals, y_vals))

        # 解包排序结果
        x_vals_sorted = [pair[0] for pair in sorted_pairs]
        y_vals_sorted = [pair[1] for pair in sorted_pairs]
        results_x.append(x_vals_sorted)
        results_y.append(y_vals_sorted)

        R1, R2 = collect_training_loss(saved_path, variable, x_limit_low, x_limit_high)

        results_x.append(R1)
        results_y.append(R2)

        results_dict = {'x': results_x, 'y': results_y}

        # 保存嵌套列表到 JSON 文件
        dict_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/saved_dict/'
        with open(dict_path + parent_folder + '_' + variable +'.json', 'w') as f:
            json.dump(results_dict, f)

        return results_x, results_y


# eval_variable = 'Z500'

variable_list = ['Z500', 'T850', 'Q700', 'U850', 'V850', 'T2m', 'SP', 'U10', 'V10']

# variable_list = ['T2m', 'SP', 'Z500', 'T850']
variable_list = ['loss']
# variable_list = ['Z500', 'T850', 'T2m', 'SP']
# variable_list = ['U10', 'SP', 'Z500', 'T850']
# 3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_right_40000


# 4xV100_ag_1_graphcast_no_GC_seed3_infobatch_prune_easy_20000


avg_list = []

x_limit_low = 120000
# x_limit_low = 210000
x_limit_low = 0
x_limit_high = 320000

for eval_val_index, eval_variable in enumerate(variable_list):

    for plot_4xV100 in [False, True]:
    # for plot_4xV100 in [False]:
        for plot_remove_all in [True, False]:
        # for plot_remove_all in [True]:
            # plot_remove_all = True
            plot_3xA100 = not plot_4xV100
            saved_path = '/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'

            x_vals_easy = []
            y_vals_easy = []
            x_vals_hard = []
            y_vals_hard = []
            x_vals_only_easy = []
            y_vals_only_easy = []
            x_vals_only_hard = []
            y_vals_only_hard = []
            x_vals_only_hard_BI_s24_3xA100 = []
            y_vals_only_hard_BI_s24_3xA100 = []
            x_vals_only_hard_BI_s26_3xA100 = []
            y_vals_only_hard_BI_s26_3xA100 = []

            x_vals_300k_no_pruning_6seeds_3xA100 = []
            y_vals_300k_no_pruning_6seeds_3xA100 = []
            x_vals_only_hard_r10_static_3xA100 = []
            y_vals_only_hard_r10_static_3xA100 = []
            x_vals_only_easy_r10_static_3xA100 = []
            y_vals_only_easy_r10_static_3xA100 = []
            x_vals_only_easy_r30_static_3xA100 = []
            y_vals_only_easy_r30_static_3xA100 = []
            x_vals_only_hard_r30_static_3xA100 = []
            y_vals_only_hard_r30_static_3xA100 = []
            x_vals_only_hard_r50_static_3xA100 = []
            y_vals_only_hard_r50_static_3xA100 = []
            x_vals_only_easy_r50_static_3xA100 = []
            y_vals_only_easy_r50_static_3xA100 = []
            x_vals_only_hard_r70_static_3xA100 = []
            y_vals_only_hard_r70_static_3xA100 = []

            x_vals_seed3_S2L_k350_r30_static_3xA100 = []
            y_vals_seed3_S2L_k350_r30_static_3xA100 = []
            x_vals_seed3_S2L_k350_r50_static_3xA100 = []
            y_vals_seed3_S2L_k350_r50_static_3xA100 = []
            x_vals_seed3_S2L_k350_r70_static_3xA100 = []
            y_vals_seed3_S2L_k350_r70_static_3xA100 = []

            x_vals_only_easy_r70_static_3xA100 = []
            y_vals_only_easy_r70_static_3xA100 = []

            x_vals_only_hard_BI_lr35_4xV100 = []
            y_vals_only_hard_BI_lr35_4xV100 = []

            x_vals_only_hard_BI_lr4_4xA100 = []
            y_vals_only_hard_BI_lr4_4xA100 = []


            x_vals_only_hard_r30_static_4xV100 = []
            y_vals_only_hard_r30_static_4xV100 = []
            x_vals_only_easy_r30_static_4xV100 = []
            y_vals_only_easy_r30_static_4xV100 = []
            x_vals_normal = []
            y_vals_normal = []


            if plot_4xV100:
                key_worlds1 = "4xV100_ag_1_graphcast_no_GC_seed3_infobatch_prune_"
                key_worlds2 = "4xV100_ag_1_graphcast_no_GC_seed3_"
            elif plot_3xA100:
                key_worlds1 = "3xA100_ag_1_graphcast_seed3_infobatch_prune_"
                key_worlds2 = "3xA100_ag_1_graphcast_seed3_"

            matching_files_info_batch = [f for f in os.listdir(saved_path) if key_worlds1 in f
                                         and f.endswith(".txt")]
            matching_files_normal = [f for f in os.listdir(saved_path) if key_worlds2 in f
                                     and f.endswith(".txt") and (f.__len__() < 50)]

            for filename in matching_files_info_batch:

                # Extract the x value from the filename (the number after the last underscore)
                file_path = saved_path + filename
                match = re.search(r'_(\d+)\.txt$', file_path)
                if match:
                    x_val = int(match.group(1))
                    if x_val < x_limit_low:
                        continue
                    if x_val > x_limit_high:
                        continue

                    with open(file_path, 'r') as f:
                        for line in f:
                            # 每行格式为 'key: value'，我们只提取value
                            if eval_variable in line.strip().split(': ')[0]:
                                value = float(line.strip().split(': ')[1])
                                # values.append(value)
                                break

                    # Extract the Z500 value from the content
                    # z500_match = re.search(r'Z500:\s*([0-9.]+)', file_content[filename])
                    # if z500_match:
                    #     y_val = float(z500_match.group(1))
                    # y_vals_easy.append(values[0])

                    # if len(filename) > 50:
                    if 'easy' in filename:
                        # if 'ra0' in filename:
                        if 'raALL0' in filename:
                            x_vals_only_easy.append(x_val)
                            y_vals_only_easy.append(value)
                        else:
                            if 'ra0' not in filename:
                                x_vals_easy.append(x_val)
                                y_vals_easy.append(value)
                    else:
                        if 'ra0' in filename:
                            x_vals_only_hard.append(x_val)
                            y_vals_only_hard.append(value)
                        else:
                            x_vals_hard.append(x_val)
                            y_vals_hard.append(value)


            if not plot_remove_all:
            # Sort the values by x for plotting
                sorted_pairs = sorted(zip(x_vals_easy, y_vals_easy))
                x_vals_easy_sorted, y_vals_easy_sorted = zip(*sorted_pairs)

                sorted_pairs = sorted(zip(x_vals_hard, y_vals_hard))
                x_vals_hard_sorted, y_vals_hard_sorted = zip(*sorted_pairs)



            if plot_3xA100:
                sorted_pairs = sorted(zip(x_vals_only_easy, y_vals_only_easy))
                x_vals_only_easy_sorted, y_vals_only_easy_sorted = zip(*sorted_pairs)
                sorted_pairs = sorted(zip(x_vals_only_hard, y_vals_only_hard))
                x_vals_only_hard_sorted, y_vals_only_hard_sorted = zip(*sorted_pairs)


            for filename in matching_files_normal:
                # Extract the x value from the filename (the number after the last underscore)
                file_path = saved_path + filename
                match = re.search(r'_(\d+)\.txt$', file_path)
                if match:
                    x_val = int(match.group(1))

                    if x_val < x_limit_low:
                        continue
                    if x_val > x_limit_high:
                        continue
                    # values = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            # 每行格式为 'key: value'，我们只提取value
                            if eval_variable in line.strip().split(': ')[0]:
                                value = float(line.strip().split(': ')[1])
                                # values.append(value)
                                break

                    x_vals_normal.append(x_val)
                    y_vals_normal.append(value)

            sorted_pairs = sorted(zip(x_vals_normal, y_vals_normal))
            x_vals_normal_sorted, y_vals_normal_sorted = zip(*sorted_pairs)




            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/')

            matching_files_no_pruning_320k = [f for f in os.listdir(saved_path) if '3xA100_graphcast_seed' in f
                                         and f.endswith("_300000.txt") and len(f) == 33]


            for filename in matching_files_no_pruning_320k:
                # Extract the x value from the filename (the number after the last underscore)
                file_path = saved_path + filename
                match = re.search(r'_(\d+)\.txt$', file_path)
                if match:
                    x_val = int(match.group(1))

                    if x_val < x_limit_low:
                        continue
                    if x_val > x_limit_high:
                        continue
                    # values = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            # 每行格式为 'key: value'，我们只提取value
                            if eval_variable in line.strip().split(': ')[0]:
                                value = float(line.strip().split(': ')[1])
                                # values.append(value)
                                break

                    x_vals_300k_no_pruning_6seeds_3xA100.append(x_val)
                    y_vals_300k_no_pruning_6seeds_3xA100.append(value)


            loss_mean = np.mean(y_vals_300k_no_pruning_6seeds_3xA100, axis=0)
            loss_mean = np.repeat(loss_mean, len(x_vals_normal_sorted))
            loss_std = np.std(y_vals_300k_no_pruning_6seeds_3xA100, axis=0)
            loss_std = np.repeat(loss_std, len(x_vals_normal_sorted))


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '4xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr4/')

            x_vals_only_hard_BI_lr4_4xA100, y_vals_only_hard_BI_lr4_4xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high
            )




            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_lr35/')

            x_vals_only_hard_BI_lr35_4xV100, y_vals_only_hard_BI_lr35_4xV100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s24/')

            x_vals_only_hard_BI_s24_3xA100, y_vals_only_hard_BI_s24_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)



            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_right_raALL0_s26/')

            x_vals_only_hard_BI_s26_3xA100, y_vals_only_hard_BI_s26_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r10_static/')

            x_vals_only_hard_r10_static_3xA100, y_vals_only_hard_r10_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r10_static/')

            x_vals_only_easy_r10_static_3xA100, y_vals_only_easy_r10_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            # only for test
            # x_total_loss, y_total_loss = collect_training_loss(saved_path, 'loss', x_limit_low, x_limit_high)
            #
            # plt.plot(x_total_loss, y_total_loss, marker='o')  # 使用 'o' 在每个点标记位置
            # # 添加标题和标签
            # plt.title("training loss")
            # plt.xlabel("iterations")
            # plt.ylabel("Total loss")
            # plt.grid(True)
            # plt.show()

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r70_static/')

            x_vals_only_hard_r70_static_3xA100, y_vals_only_hard_r70_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            # saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
            #               '3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r70_static/')
            #
            # x_vals_only_easy_r70_static_3xA100, y_vals_only_easy_r70_static_3xA100 = collect_data(
            #     saved_path, eval_variable, x_limit_low, x_limit_high)


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r50_static/')

            x_vals_only_hard_r50_static_3xA100, y_vals_only_hard_r50_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_S2L_k350_r30_static/')

            x_vals_seed3_S2L_k350_r30_static_3xA100, y_vals_seed3_S2L_k350_r30_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_S2L_k350_r50_static/')

            x_vals_seed3_S2L_k350_r50_static_3xA100, y_vals_seed3_S2L_k350_r50_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_S2L_k350_r70_static/')

            x_vals_seed3_S2L_k350_r70_static_3xA100, y_vals_seed3_S2L_k350_r70_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)


            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r50_static/')

            x_vals_only_easy_r50_static_3xA100, y_vals_only_easy_r50_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '4xV100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static_new/')

            x_vals_only_hard_r30_static_4xV100, y_vals_only_hard_r30_static_4xV100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '4xV100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/')

            x_vals_only_easy_r30_static_4xV100, y_vals_only_easy_r30_static_4xV100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_hard_s100_r30_static/')

            x_vals_only_easy_r30_static_3xA100, y_vals_only_easy_r30_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            saved_path = ('/hpi/fs00/home/zi.yang/research_project/ArchesWeather/modelstore/ArchesModel/results/'
                          '3xA100_ag_1_graphcast_seed3_infobatch_prune_easy_s100_r30_static/')

            x_vals_only_hard_r30_static_3xA100, y_vals_only_hard_r30_static_3xA100 = collect_data(
                saved_path, eval_variable, x_limit_low, x_limit_high)

            plot_loss_curve = True

            if plot_loss_curve:
                # Plotting the Z500 values
                plt.figure(figsize=(10, 6))
                # plt.figure(figsize=(15, 9))
                # plt.figure(figsize=(5, 3))

                if eval_variable != 'loss':
                    plt.plot(x_vals_normal_sorted, y_vals_normal_sorted, linewidth=1, marker='o', color='black',
                             linestyle='--', label='No Pruning')


                # plt.plot(x_vals_normal_sorted, loss_mean, label='No Pruning (Mean)', linewidth=1, marker='o', color='black',
                #          linestyle=':',)
                # plt.fill_between(x_vals_normal_sorted, loss_mean - loss_std, loss_mean + loss_std, color='black', alpha=0.2,)
                #                  # label='Mean ± Std')

        # 画一个最后epoch的结果的辅助直线
                # y_vals_aux = [y_vals_normal_sorted[-1]] * len(y_vals_normal_sorted)
                # plt.plot(x_vals_normal_sorted, y_vals_aux, linewidth=1, marker='o', color='black',
                #          linestyle='--')

                if eval_variable =='loss':
                    results_index = 1
                else:
                    results_index = 0

                if plot_4xV100:
                    if plot_remove_all:
                        plt.plot(x_vals_only_hard_BI_lr4_4xA100[results_index], y_vals_only_hard_BI_lr4_4xA100[results_index], linewidth=1, marker='o',
                                 label='Remove All hard Samples using InfoBatch - lr4')
                        plt.plot(x_vals_only_hard_r30_static_4xV100[results_index], y_vals_only_hard_r30_static_4xV100[results_index], linewidth=1, marker='o',
                                 label='Remove All Easy Samples static - remove 70%')
                        plt.plot(x_vals_only_easy_r30_static_4xV100[results_index], y_vals_only_easy_r30_static_4xV100[results_index], linewidth=1, marker='o',
                                 label='Remove All Hard Samples static - remove 70%')
                    else:
                        plt.plot(x_vals_easy_sorted, y_vals_easy_sorted, linewidth=1, marker='o',
                                 # label='Remove 50% Hard Samples using InfoBatch - BS=4')
                                 label='Remove 50% Hard Samples using InfoBatch - lr 3e-4')
                                 # label='Remove All Hard Samples using InfoBatch - lr 3e-4 fake for plot! not all is 50%')

                        plt.plot(x_vals_hard_sorted, y_vals_hard_sorted, linewidth=1, marker='o',
                                 ## label='Remove 50% Easy Samples using InfoBatch - BS=4')
                                 label='Remove 50% Easy Samples using InfoBatch')

                        plt.plot(x_vals_only_hard_BI_lr35_4xV100[results_index], y_vals_only_hard_BI_lr35_4xV100[results_index], linewidth=1, marker='o',
                                 label='Remove All Hard Samples using InfoBatch- lr 3.5e-4')

                        plt.plot(x_vals_only_hard_BI_lr4_4xA100[results_index], y_vals_only_hard_BI_lr4_4xA100[results_index], linewidth=1, marker='o',
                                 label='Remove All Hard Samples using InfoBatch - lr 4e-4')





                else:
                    if plot_remove_all:
                        # plt.plot(x_vals_only_easy_sorted, y_vals_only_easy_sorted, linewidth=1, marker='o',
                        #          label='Remove All Hard Samples using InfoBatch')
                        # plt.plot(x_vals_only_hard_sorted, y_vals_only_hard_sorted, linewidth=1, marker='o',
                        #          label='Remove All Easy Samples using InfoBatch')

                        # plt.plot(x_vals_only_hard_BI_s24_3xA100[results_index], y_vals_only_hard_BI_s24_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove All Hard Samples using InfoBatch - step 240k')

                        # plt.plot(x_vals_only_hard_BI_s26_3xA100[results_index], y_vals_only_hard_BI_s26_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove All Hard Samples using InfoBatch - step 260k')

                        # plt.plot(x_vals_only_easy_r10_static_3xA100[results_index], y_vals_only_easy_r10_static_3xA100[results_index], linewidth=1, marker='o',
                                 # label='Remove 90% samples')
                                 # label='Remove hard Samples static - remove 90%')

                        # plt.plot(x_vals_only_hard_r10_static_3xA100[results_index], y_vals_only_hard_r10_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove easy Samples static - remove 90%')

                        # current there is no results
                        # plt.plot(x_vals_only_hard_r30_static_3xA100[results_index], y_vals_only_hard_r30_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove easy Samples static - remove 70%')

                        plt.plot(x_vals_only_easy_r30_static_3xA100[results_index], y_vals_only_easy_r30_static_3xA100[results_index], linewidth=1, marker='o',
                                 # label='Remove 70% samples')
                                 label='Remove hard Samples static - remove 70%')

                        # plt.plot(x_vals_only_hard_r50_static_3xA100[results_index], y_vals_only_hard_r50_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove easy Samples static - remove 50%')
                        #
                        plt.plot(x_vals_only_easy_r50_static_3xA100[results_index], y_vals_only_easy_r50_static_3xA100[results_index], linewidth=1, marker='o',
                                 # label='Remove 50% samples')
                                 label='Remove hard Samples static - remove 50%')

                        # plt.plot(x_vals_only_easy_r70_static_3xA100[results_index], y_vals_only_easy_r70_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove hard Samples static - remove 30%')

                        plt.plot(x_vals_only_hard_r70_static_3xA100[results_index], y_vals_only_hard_r70_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove 30% samples')
                                 label='Remove easy Samples static - remove 30%')

                        # plt.plot(x_vals_seed3_S2L_k350_r70_static_3xA100[results_index], y_vals_seed3_S2L_k350_r70_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove based on S2L-k350 - remove 30%')
                        #
                        # plt.plot(x_vals_seed3_S2L_k350_r50_static_3xA100[results_index], y_vals_seed3_S2L_k350_r50_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove based on S2L-k350 - remove 50%')
                        #
                        # plt.plot(x_vals_seed3_S2L_k350_r30_static_3xA100[results_index], y_vals_seed3_S2L_k350_r30_static_3xA100[results_index], linewidth=1, marker='o',
                        #          label='Remove based on S2L-k350 - remove 70%')




                    else:
                        plt.plot(x_vals_only_easy_sorted, y_vals_only_easy_sorted, linewidth=1, marker='o',
                                 label='Remove All Hard Samples using InfoBatch')

                        plt.plot(x_vals_only_hard_sorted, y_vals_only_hard_sorted, linewidth=1, marker='o',
                                 label='Remove All Easy Samples using InfoBatch')

                        plt.plot(x_vals_easy_sorted, y_vals_easy_sorted, linewidth=1, marker='o',
                                 label='Remove 50% Hard Samples using InfoBatch')
                        plt.plot(x_vals_hard_sorted, y_vals_hard_sorted, linewidth=1, marker='o',
                                 label='Remove 50% Easy Samples using InfoBatch')

                plt.title(f'RMSE scores of {eval_variable} across training iterations on test set', fontsize=22)
                plt.xlabel('Iterations', fontsize=16)
                plt.ylabel(f'RMSE scores of {eval_variable}', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                # plt.xlim(100000, 320000)
                plt.legend(fontsize=16)
                # plt.legend(fontsize=12)
                plt.grid(True)
                plt.show()
                if eval_variable != 'loss':
                    print(eval_variable)
                    print(f'no pruning {y_vals_normal_sorted[-1]:.2f}')
                    print(f'keep 70%   {y_vals_only_hard_r70_static_3xA100[0][-1]:.2f}    {(y_vals_only_hard_r70_static_3xA100[0][-1]/y_vals_normal_sorted[-1]-1)*100:.1f}')
                    print(f'keep 50%   {y_vals_only_easy_r50_static_3xA100[0][-1]:.2f}    {(y_vals_only_easy_r50_static_3xA100[0][-1]/y_vals_normal_sorted[-1]-1)*100:.1f}')
                    print(f'keep 30%   {y_vals_only_easy_r30_static_3xA100[0][-1]:.2f}    {(y_vals_only_easy_r30_static_3xA100[0][-1]/y_vals_normal_sorted[-1]-1)*100:.1f}')
                    print(f'keep 10%   {y_vals_only_easy_r10_static_3xA100[0][-1]:.2f}    {(y_vals_only_easy_r10_static_3xA100[0][-1]/y_vals_normal_sorted[-1]-1)*100:.1f}')
                    print('\n')
                    # print('remove 30%: ', y_vals_only_hard_r70_static_3xA100[-1])
                    # print('without removing: ', y_vals_normal_sorted[-1])
                    # print('decreased %:', 100.0*(y_vals_only_hard_r70_static_3xA100[-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])
                    # avg_list.append(100.0*(y_vals_only_hard_r70_static_3xA100[-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])
                    avg_list.append(100.0*(y_vals_only_easy_r10_static_3xA100[0][-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])
                    # avg_list.append(100.0*(y_vals_only_easy_r30_static_3xA100[-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])
                    # avg_list.append(100.0*(y_vals_only_hard_r50_static_3xA100[-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])
                    # avg_list.append(100.0*(y_vals_only_easy_r50_static_3xA100[-1]-y_vals_normal_sorted[-1])/y_vals_normal_sorted[-1])

            plot_final_results = False
            if plot_final_results:
                plt.figure(figsize=(8, 6))
                x_label = np.array([0.1, 0.3, 0.5, 0.7])
                loss_mean = loss_mean[:len(x_label)]
                loss_std = loss_std[:len(x_label)]

                plt.plot(x_label, loss_mean[:len(x_label)], label='No Pruning (Mean)', linewidth=1, marker='o', color='black',
                         linestyle=':',)
                plt.fill_between(x_label, loss_mean - loss_std, loss_mean + loss_std, color='black', alpha=0.2,)
                y_values = []
                y_values.append(y_vals_only_easy_r10_static_3xA100[-1])
                y_values.append(y_vals_only_easy_r30_static_3xA100[-1])
                y_values.append(y_vals_only_easy_r50_static_3xA100[-1])
                y_values.append(y_vals_only_hard_r70_static_3xA100[-1])
                plt.plot(x_label, y_values, label='Static Pruning', linewidth=2,
                         marker='o',)
                plt.title(f'RMSE scores of {eval_variable} on test set', fontsize=22)
                plt.xlabel('Data remaining after pruning', fontsize=16)
                plt.ylabel(f'RMSE scores of {eval_variable}', fontsize=16)
                plt.xticks(x_label, [f"{int(val * 100)}%" for val in x_label], fontsize=16)
                plt.yticks(fontsize=16)
                # plt.xlim(100000, 320000)
                plt.legend(fontsize=18)
                # plt.legend(fontsize=12)
                plt.grid(True)
                plt.show()



if plot_loss_curve and eval_variable != 'loss':
    print(f'{eval_variable}')
    mean_value = sum(avg_list) / len(avg_list)
    print("Mean value:", mean_value)
    # removing 30% easy， rmse decrease 1.8%
    # removing 50% hard， rmse decrease 2.913%
    # removing 50% easy， rmse decrease 4.04%
    # removing 70% hard， rmse decrease 6.85%