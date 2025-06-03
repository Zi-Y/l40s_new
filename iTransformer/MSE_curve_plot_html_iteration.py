import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots  # 【新增】用于生成子图
import torch


# ========================================
# 1. 遍历所有组合（pm, pr）并加载各 seed 的 test_loss 曲线和对应的 step
# ========================================

# pm_values = [0, 1, 2, 3, 4, 9,]
# pm_values = [0, 4, 5, 6, 7]
# pm_values = [0, 6, 7]
# pm_values = [0, 4, 9]


pm_values = [0, 1, 2, 3, 9, 10]
pm_values = [0, 4, 9, 10, 11]
pm_values = [0,  9, 11]
# pm_values = [0,  1, 2, 3, 4, 9, ]
pm_values = [-1, 0,  4, 9, 12, 13, 14, 15,]
pm_values = [-1, 0,  9, 12,]
# pm_values = [0,  1, 2, 3, 4, 9, 10, 11]
pm_values = [0, 1, 2, 3, 4, 9, 13, 14, 15, 16]
pm_values = [0, 4, 9, 12, 13, 14, 15, 16, 17, 18]
pm_values = [0, 9, 12, 16, 18, 21, 22]

pm_values = [-1,0, 1, 3, 2, 22, 4, 9, 12, 18, 19, 16]


pr_values = [0, 10, 30, 50, 70, 90]
pr_values = [0, 50]
pr_values = [0, 10, 30, 50, 70,]

# pr_values = [0, 10, 90]
# pr_values = [0, 10]
# seeds = range(51)  # seed 从 0 到 51
# seeds = range(27)  # seed 从 0 到 27
# seeds = range(5)  # seed 从 0 到 27
seeds = range(20)  # seed 从 0 到 27
pm_names = {
    -1: "No Pruning",
    0: "Random Pruning",
    1: "Avg Loss (sample, static)",
    2: "S2L (sample, static)",
    3: "TTDS (sample, static)",
    4: "InfoBatch (sample, dynamic)",
    5: "I-IB",
    6: "IB KM", # keep middle
    7: "IB RM", # remove middle
    8: "IB R-part", # remove a part
    9: "Modified SLM - per batch (token, dynamic)", #"Token level pruning, per batch", #"SLM-token (dynamic)",
    10: "IB token",
    11: "IB token pr",
    12: "Modified SLM - global (token, dynamic)",#"Token level pruning, global", #"Global SLM-token , same as 9" Global token level pruning (dynamic),
    13: "SLM-token, using (loss at 1 epoch - current loss)",
    14: "SLM-token, using (previous epoch loss - current loss)", # using (previous epoch loss - current loss)
    15: "Token level pruning, per batch, using loss change rate", # SLM-token, using loss difference divided by previous epoch loss, (previous epoch loss - current loss)/previous epoch loss
    16: "random token pruning",
    17: "SLM-token, using loss at 1 epoch - current loss divided by previous epoch loss", # (loss at 1 epoch - current loss)/previous epoch loss
    18: "Trend error - per batch (token, static)",# batch level pruning
    19: "Trend error - global (token, static)",
    21: "Trend error - sample level pruning, remove easy",
    22: "Trend error (sample, static)", # trend error - sample level pruning, remove hard",
    100: "calculate sample level gradient l2 norm, dot, cosine similarity",
    101: "calculate time point level (each time point mean 21 variables at a singe time point) gradient l2 norm, dot, cosine similarity",
}



# 15 - 19 使用infobatch
[0, 1, 2, 3, 4, 9]
# 用 data_dict 保存每个组合的平均曲线（x轴为 step）和标准差，字典的键形如 "pm0, PR 90"
data_dict = {}
# dir_list = ["./checkpoints", "./checkpoints_torchGaussianNLL_weather",
#             "./checkpoints_no_early_stopping_iteration_based"]
dir_list = ["./checkpoints", ]

# dir_name = ["CRPS", "NLL loss", "MSE loss"]
dir_name = [""]
# True False
epoch_eval = True
use_val_loss_choose = False

predicted_len_list = [96, 192, 336, 720]
predicted_len_list = [96]
for dir_index, base_dir in enumerate(dir_list):
    for predicted_len in predicted_len_list:

        for pm in pm_values:
            for pr in pr_values:

                if pr == 0 and (pm != 0 or pm != -1):
                    continue

                # only need original results without pruning
                # if pr != 0 and pm == 0:
                #     continue


                if pm == 12:
                    base_dir = "./PM12_iter_6830_tokens_checkpoints"
                    # base_dir = "./checkpoints"
                # elif pm in 13:
                #     base_dir = "./PM13_token_pr_rate<0_checkpoints"
                elif pm in (13, 14, 15, 16):
                    base_dir = "./pm13-16_checkpoints"

                elif pm == 17:
                    base_dir = "./pm17_checkpoints"

                elif pm == 18:
                    base_dir = "./pm18_checkpoints"

                elif pm == 19:
                    base_dir = "./pm19_checkpoints"

                elif pm in (21, 22):
                    base_dir = "./pm21-22_checkpoints"

                elif pm == 10:
                    base_dir = "./pm10_checkpoints"
                elif pm == 9:
                    base_dir = "./PM9_iter_6830_tokens_checkpoints"
                    # base_dir = "./pm9_epoch_5_checkpoints"

                    # base_dir = "./pm4_to_9_checkpoints"

                elif pm >= 4 and pm <= 8:
                    base_dir = "./pm4_to_9_checkpoints"
                elif pm >= 1 and pm <= 3:
                    base_dir = "./pm0_to_3_checkpoints"
                elif pm == 0 and pr == 0:
                    base_dir = "./PM9_iter_6830_tokens_checkpoints"
                elif pm == -1:
                    base_dir = "./PM9_iter_6830_tokens_checkpoints"
                elif pm == 0 and pr != 0:
                    base_dir = "./pm0_to_3_checkpoints"
                else:
                    base_dir = "./checkpoints"


                # for low_t in [10, 20, 30, 40]:
                # for low_t in [0, 20, 40, 60, 80]:
                # for low_t in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, -10, -30, -50, -70]:
                # for low_t in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, ]
                for low_t in [10, 30, 50, 70]:
                    for low_t_r in [10, 30, 50, 70]:
                    # for low_t in [10, ]:
                        # 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 -0.1 -0.3 -0.5 -0.7
                        # 构造模型文件夹的固定部分（去掉 seed 前缀）
                        if pm == 6 or pm == 7:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr} l{low_t}"
                            # seed0_pm6_pr30_low10_high10_weather_96_192_iTransformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
                            model_part = f"pm{pm}_pr{pr}_low{low_t}_high{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        # model_part = f"pm{pm}_pr{pr}_ECL_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        elif pm == 8:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr} l{low_t}"
                            model_part = f"pm{pm}_pr{pr}_low10_high10_start{low_t}_int20_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        #     seed13_pm8_pr30_low10_high10_start20_int20_weather_96_336_iTransformer_custom_ftM_sl96_ll48_pl336_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0

                        elif pm == 9:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{low_t}"
                            model_part = f"pm{pm}_pr10_low10_high10_start10_int25_tr{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        # seed1_pm9_pr10_low10_high10_start10_int25_tr50_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0
                        elif pm == 10:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr} TR{low_t}"
                            model_part = f"pm{pm}_pr{pr}_low10_high10_start10_int25_tr{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        # seed30_pm10_pr70_low10_high10_start10_int25_tr70_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0

                        elif pm == 11:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr} TR{low_t} TPT{low_t_r}"
                            model_part = f"pm{pm}_pr{pr}_low{low_t_r}_high10_start10_int25_tr{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        # pm11_pr90_low10_high10_start10_int25_tr50_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0

                        elif pm == 12:
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{low_t}"
                            model_part = f"pm{pm}_pr10_low10_high10_start10_int25_tr{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"

                        elif pm in (13, 14, 15, 16, 17, 18, 19):
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{low_t}"
                            model_part = f"pm{pm}_pr10_low10_high10_start10_int25_tr{low_t}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"

                        elif pm in (21, 22):
                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{low_t}"
                            model_part = f"pm{pm}_pr{low_t}_low10_high10_start10_int25_tr10_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"


                        elif pm <= 3 and pm >= 0:
                            if pm==0 and pr==0:
                                label = f"No pruning PR{pr}"
                            else:
                                label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr}"
                            model_part = f"pm{pm}_pr{pr}_low10_high10_start10_int25_tr10_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
                        # seed29_pm2_pr90_low10_high10_start10_int25_tr10_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0

                        elif pm == -1:
                                label = f"No pruning PR{pr}"
                                model_part = f"pm0_pr0_low10_high10_start10_int25_tr10_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"

                        else:
                            model_part = f"pm{pm}_pr{pr}_weather_96_{predicted_len}_iTransformer_custom_ftM_sl96_ll48_pl{predicted_len}_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"

                            label = f"{dir_name[dir_index]}{pm_names[pm]} P{predicted_len} PR{pr}"

                        # 第一遍：遍历各个 seed，找出所有 seed 中最大的 epoch 编号
                        global_epoch_max = -1
                        seed_data_list = []
                        # 【新增功能】存储每个 seed 在最佳 epoch（即 vali_MSE_loss 最小时）的指标：
                        # 现在记录 best_step（对应的 global_step）、best_vali_MSE_loss、best_test_MSE_loss
                        best_metrics_list = []
                        for seed in seeds:
                            folder_name = f"seed{seed}_{model_part}"
                            folder_path = os.path.join(base_dir, folder_name)
                            if not os.path.exists(folder_path):
                                print(f"Warning: {folder_path} 不存在")
                                continue
                            # 加载该 seed 下所有文件（要求文件名形如 epoch_{epoch}_results.npy）
                            epoch_results = {}
                            for file in os.listdir(folder_path):
                                # if (pm == 0 and pr == 0) or (pm == 9):
                                if epoch_eval:
                                    m = re.match(r"epoch_(\d+)_results\.npy", file)
                                    if m:
                                        epoch = int(m.group(1))
                                        data = np.load(os.path.join(folder_path, file))
                                        epoch_results[epoch] = (data[0], data[6], data[3])
                                else:
                                    m = re.match(r"iter_(\d+)_results\.npy", file)
                                    if m:
                                        epoch = int(int(m.group(1))/100-1)
                                        data = np.load(os.path.join(folder_path, file))
                                        epoch_results[epoch] = (data[0], data[6], data[3])
                                    # else:
                                    #     m = re.match(r"epoch_(\d+)_results\.npy", file)
                                    #     if m:
                                    #         epoch = int(m.group(1))
                                    #         data = np.load(os.path.join(folder_path, file))
                                    #         epoch_results[epoch] = (data[0], data[6], data[3])


                                    # if 'MSE' in dir_name[dir_index]:
                                    #     # 假设数据格式为 np.array([step, train_loss, vali_loss, test_loss])
                                    #     # 保存 (step, test_MSE_loss, vali_MSE_loss)
                                    #     epoch_results[epoch] = (data[0], data[6], data[3])
                                    # # elif 'NLL' in dir_name[dir_index]:
                                    # else:
                                    #     # np.array([global_step, epoch + 1, train_loss, vali_MSE_loss, vali_MAE_loss,
                                    #     #                               vali_total_loss,
                                    #     #                               test_MSE_loss, test_MAE_loss, test_total_loss]))
                                    #     epoch_results[epoch] = (data[0], data[6], data[3])
                            if not epoch_results:
                                continue
                            local_max = max(epoch_results.keys())
                            if local_max > global_epoch_max:
                                global_epoch_max = local_max

                            # 【新增功能】找出该 seed 下 vali_MSE_loss 最小时对应的 step，以及对应的 test_MSE_loss
                            best_step = None
                            best_vali = np.inf
                            best_test = np.inf
                            # best_test = np.nan
                            for ep, (step_val, test_val, vali_val) in epoch_results.items():
                                if use_val_loss_choose:
                                    if vali_val < best_vali:
                                        best_vali = vali_val
                                        best_test = test_val
                                        best_step = step_val
                                else:
                                    if test_val < best_test:
                                        best_vali = vali_val
                                        best_test = test_val
                                        best_step = step_val
                            if best_step is not None:
                                best_metrics_list.append((best_step, best_vali, best_test))

                            seed_data_list.append(epoch_results)

                        if global_epoch_max < 0 or len(seed_data_list) == 0:
                            print(f"No data found for combination {label}")
                            continue

                        # 全局总 epoch 数（注意 epoch 从 0 开始）
                        global_epoch_count = global_epoch_max + 1

                        # 第二遍：对每个 seed 构造完整曲线，
                        # 若缺失某些 epoch则直接用 np.nan 表示
                        seed_steps_list = []
                        seed_loss_list = []
                        for epoch_results in seed_data_list:
                            full_steps = []
                            full_losses = []
                            for epoch in range(global_epoch_count):
                                if epoch in epoch_results:
                                    # 修改此处以适应新的 tuple 结构，多出的 vali_MSE_loss 用 _ 忽略
                                    step_val, loss_val, _ = epoch_results[epoch]
                                else:
                                    step_val, loss_val = np.nan, np.nan
                                full_steps.append(step_val)
                                full_losses.append(loss_val)
                            seed_steps_list.append(full_steps)
                            seed_loss_list.append(full_losses)

                        # 将所有 seed 的曲线转换为 numpy 数组后，计算均值和标准差（在 seed 维度上）
                        seed_steps_array = np.array(seed_steps_list)  # shape: (num_seeds, global_epoch_count)
                        seed_loss_array = np.array(seed_loss_list)  # shape: (num_seeds, global_epoch_count)
                        mean_steps = np.nanmean(seed_steps_array, axis=0)
                        mean_loss = np.nanmean(seed_loss_array, axis=0)
                        std_loss = np.nanstd(seed_loss_array, axis=0)

                        # 使用高斯滤波对 loss 曲线进行平滑处理
                        # mean_loss_smooth = gaussian_filter1d(mean_loss, sigma=0.1)
                        # std_loss_smooth = gaussian_filter1d(std_loss, sigma=0.1)

                        # 【新增功能】计算所有 seed 的最佳指标均值和标准差
                        best_step_avg = np.nan
                        best_step_std = np.nan
                        best_vali_avg = np.nan
                        best_test_avg = np.nan
                        best_test_std = np.nan
                        if best_metrics_list:
                            best_step_avg = np.mean([item[0] for item in best_metrics_list])
                            best_step_std = np.std([item[0] for item in best_metrics_list])
                            best_vali_avg = np.mean([item[1] for item in best_metrics_list])
                            best_test_avg = np.mean([item[2] for item in best_metrics_list])
                            best_test_std = np.std([item[2] for item in best_metrics_list])

                        # 保存数据，值为 (平均的 step, 平滑后的均值, 平滑后的标准差, (best_step_avg, best_step_std, best_vali_avg, best_test_avg, best_test_std))
                        data_dict[label] = (mean_steps, mean_loss, std_loss,
                                            (best_step_avg, best_step_std, best_vali_avg, best_test_avg, best_test_std))

# ========================================
# 2. 使用 Plotly 绘图，确保阴影区域与主曲线强绑定，x轴为 step
# ========================================

fig = go.Figure()
color_list = [
    'rgba(50, 100, 150, 1)',
    'rgba(200, 50, 50, 1)',
    'rgba(50, 200, 50, 1)',
    'rgba(200, 150, 50, 1)',
    'rgba(200, 100, 50, 1)'
    'rgba(50, 50, 200, 1)',
    'rgba(150, 50, 200, 1)',
    'rgba(50, 200, 200, 1)',
    'rgba(200, 50, 150, 1)',
    'rgba(100, 50, 50, 1)',
    'rgba(150, 200, 50, 1)',
    'rgba(100, 100, 200, 1)',
    'rgba(50, 150, 100, 1)',
    'rgba(100, 200, 150, 1)',
    'rgba(200, 200, 50, 1)',
    'rgba(50, 50, 100, 1)',
    'rgba(200, 50, 200, 1)',
    'rgba(50, 200, 100, 1)',
    'rgba(100, 50, 200, 1)',
    # 'rgba(200, 100, 50, 1)'
]

color_list = [
    'rgba(214, 39, 40, 1)',    # 红
    'rgba(31, 119, 180, 1)',   # 蓝
    'rgba(255, 127, 14, 1)',   # 橙
    'rgba(44, 160, 44, 1)',    # 绿
    'rgba(148, 103, 189, 1)',  # 紫
    'rgba(140, 86, 75, 1)',    # 棕
    'rgba(227, 119, 194, 1)',  # 粉紫
    'rgba(127, 127, 127, 1)',  # 灰
    'rgba(188, 189, 34, 1)',   # 橄榄绿
    'rgba(23, 190, 207, 1)',   # 青蓝
    'rgba(174, 199, 232, 1)',  # 淡蓝
    'rgba(255, 187, 120, 1)',  # 淡橙
    'rgba(152, 223, 138, 1)',  # 淡绿
    'rgba(255, 152, 150, 1)',  # 淡红
    'rgba(197, 176, 213, 1)',  # 淡紫
    'rgba(196, 156, 148, 1)',  # 淡棕
    'rgba(247, 182, 210, 1)',  # 淡粉
    'rgba(199, 199, 199, 1)',  # 浅灰
    'rgba(219, 219, 141, 1)',  # 淡黄绿
    'rgba(158, 218, 229, 1)'   # 淡青蓝
]

color_list = [
    # Okabe & Ito 9 色调色板（色盲友好）
    'rgba(0, 0, 0, 1)',        # 黑色
    'rgba(230, 159, 0, 1)',    # 橙色
    'rgba(86, 180, 233, 1)',   # 浅蓝
    'rgba(0, 158, 115, 1)',    # 墨绿色
    'rgba(0, 204, 204, 1)',    # 青色
    'rgba(0, 114, 178, 1)',    # 深蓝
    'rgba(213, 94, 0, 1)',     # 朱红（深橙）
    'rgba(204, 121, 167, 1)',  # 紫红
    'rgba(128, 128, 128, 1)',  # 中灰
    # 如需更多类别，可考虑以下附加色（选用性更强）
    'rgba(86, 86, 214, 1)',    # 蓝紫
    'rgba(255, 128, 0, 1)',    # 亮橙
]

color_index = 0

for label, (mean_steps, mean_loss_smooth, std_loss_smooth, *rest) in data_dict.items():
    color = color_list[color_index % len(color_list)]
    color_index += 1
    # 阴影颜色设为半透明
    shadow_color = color.replace("1)", "0.3)")

    # 添加主曲线 trace，x 轴使用平均后的 step 值
    fig.add_trace(go.Scatter(
        x=mean_steps,
        y=mean_loss_smooth,
        mode='lines',
        name=label,
        legendgroup=label,
        line=dict(width=3, shape='spline', smoothing=0.0, color=color),
        hoverinfo='x+y+name'
    ))

    # 添加阴影区域 trace (与主曲线绑定)
    fig.add_trace(go.Scatter(
        x=np.concatenate([mean_steps, mean_steps[::-1]]),
        y=np.concatenate([mean_loss_smooth - std_loss_smooth, (mean_loss_smooth + std_loss_smooth)[::-1]]),
        fill='toself',
        fillcolor=shadow_color,
        line=dict(color='rgba(255,255,255,0)'),
        legendgroup=label,
        showlegend=False,
        name=label,
        hoverinfo='skip'
    ))

title_name = ''
if 'ECL' in model_part:
    title_name = 'Test MSE loss on ECL Dataset'
elif model_part == 'weather':
    title_name = 'Test MSE loss on Weather Dataset'
fig.update_layout(
    title=dict(text=title_name, font=dict(size=20), x=0.5),
    xaxis_title="Step",
    yaxis_title="Test MSE Loss",
    template="plotly_white",
    hoverlabel=dict(namelength=-1)
)
if 'ECL' in model_part:
    output_path = "ECL_interactive_plot_iteration.html"
elif 'weather' in model_part:
    output_path = "Weather_interactive_plot_iteration.html"
pio.write_html(fig, output_path)
print(f"Interactive plot saved as {output_path}")

# 按方法分组，并提取每个 label 中 "PR" 后的值以及 best_test_avg, best_test_std
grouped_data = {}
for label, data in data_dict.items():
    # 尝试提取 "PR" 后的数值
    match = re.search(r'PR(\d+)', label)
    if not match:
        continue
    pr_value = int(match.group(1))
    best_test_avg = data[3][3]
    best_test_std = data[3][4]
    # 提取方法名称（假设方法名称在 label 中出现，参考 pm_names 中的值）
    method_found = None
    for mname in pm_names.values():
        prefix = re.split(r'P(?=\d)', label, maxsplit=1)[0][:-1]
        if mname == prefix:
            method_found = mname
            break

    # if label.startswith(mname):
        # if mname in label:
        #     method_found = mname
        #     break
    if method_found is None:
        continue
    grouped_data.setdefault(method_found, []).append((pr_value, best_test_avg, best_test_std))

# 对每个方法内的数据按照 PR 值排序
for method in grouped_data:
    grouped_data[method].sort(key=lambda x: x[0])

# 新建一个 Figure 用于绘制多条折线图
fig_pr = go.Figure()
color_index = 0

for method, values in grouped_data.items():
    pr_values = [v[0] for v in values]
    best_test_avgs = [v[1] for v in values]
    best_test_stds = [v[2] for v in values]

    color = color_list[color_index % len(color_list)]
    color_index += 1
    # 阴影颜色设为半透明
    shadow_color = color.replace("1)", "0.3)")

    # 计算上下界
    upper_bound = [avg + std for avg, std in zip(best_test_avgs, best_test_stds)]
    lower_bound = [avg - std for avg, std in zip(best_test_avgs, best_test_stds)]

    # 添加阴影区域 trace (与主曲线绑定)
    fig_pr.add_trace(go.Scatter(
        x=pr_values + pr_values[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor=shadow_color,
        line=dict(color='rgba(255,255,255,0)'),
        legendgroup=method,
        showlegend=False,
        name=method,
        hoverinfo='skip'
    ))

    # 添加主折线 trace
    fig_pr.add_trace(go.Scatter(
        x=pr_values,
        y=best_test_avgs,
        mode='lines+markers',
        name=method,
        line=dict(width=3, color=color),
        legendgroup=method,
        showlegend=True,

    ))

# 更新图表布局
fig_pr.update_layout(
    # title=f"Best Test MSE Loss vs Pruning rate per Method, epoch_eval={epoch_eval}, use_val_loss_choose={use_val_loss_choose}",
    title=f"Best Test MSE Loss vs Pruning rate per Method",
    xaxis_title="Pruning rate (%)",
    yaxis_title="Best Test MSE Loss",
    template="plotly_white",
    hoverlabel=dict(
        # font_size=14,  # 调整字体大小，避免被裁剪
        namelength=-1  # -1 表示不裁剪名称
    ),
)

# 保存为新的 html 文件
output_pr_path = "Best_Test_MSE_Loss_per_PR.html"
pio.write_html(fig_pr, output_pr_path)
print(f"Interactive PR plot saved as {output_pr_path}")



# ========================================
# 【新增功能】在同一个 html 文件上画2个折线图：
# 1. x轴为 label，y轴为 vali_MSE_loss 最小时对应的 step 的均值（即 best_step_avg）及标准差
# 2. x轴为 label，y轴为 test_MSE_loss 的均值（在 vali_MSE_loss 最小时，即 best_test_avg）及标准差
# ========================================

labels = []
best_step_mean = []
best_step_std_list = []
best_test_mean = []
best_test_std_list = []
for label, value in data_dict.items():
    labels.append(label)
    best_step_mean.append(value[3][0])
    best_step_std_list.append(value[3][1])
    best_test_mean.append(value[3][3])
    best_test_std_list.append(value[3][4])

# 使用子图将2个图画在同一 html 文件中，且改为折线图，并添加阴影区域表示标准差
fig_combined = make_subplots(rows=2, cols=1,
                             subplot_titles=("Average Best Step (min vali_MSE_loss) per Label",
                                             "Average Test MSE Loss at Best Step per Label"))

# 第一子图：Average Best Step
# 计算上界和下界
upper_best_step = (np.array(best_step_mean) + np.array(best_step_std_list)).tolist()
lower_best_step = (np.array(best_step_mean) - np.array(best_step_std_list)).tolist()
# 添加阴影区域
fig_combined.add_trace(go.Scatter(
    x=labels + labels[::-1],
    y=upper_best_step + lower_best_step[::-1],
    fill='toself',
    fillcolor='rgba(0,0,255,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    legendgroup="Average Best Step",
    showlegend=True,
    name="Average Best Step",
    hoverinfo='skip'
), row=1, col=1)
# 添加均值折线
fig_combined.add_trace(go.Scatter(
    x=labels,
    y=best_step_mean,
    mode='lines+markers',
    name="Average Best Step"
), row=1, col=1)

# 第二子图：Average Test MSE Loss
upper_best_test = (np.array(best_test_mean) + np.array(best_test_std_list)).tolist()
lower_best_test = (np.array(best_test_mean) - np.array(best_test_std_list)).tolist()
# 添加阴影区域
fig_combined.add_trace(go.Scatter(
    x=labels + labels[::-1],
    y=upper_best_test + lower_best_test[::-1],
    fill='toself',
    fillcolor='rgba(0,255,0,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    legendgroup="Average Test MSE Loss",
    showlegend=True,
    name="Average Test MSE Loss",
    hoverinfo='skip'
), row=2, col=1)
# 添加均值折线
fig_combined.add_trace(go.Scatter(
    x=labels,
    y=best_test_mean,
    mode='lines+markers',
    name="Average Test MSE Loss"
), row=2, col=1)

fig_combined.update_xaxes(title_text="Label", row=1, col=1)
fig_combined.update_yaxes(title_text="Average Best Step", row=1, col=1)
fig_combined.update_xaxes(title_text="Label", row=2, col=1)
fig_combined.update_yaxes(title_text="Average Test MSE Loss", row=2, col=1)
fig_combined.update_layout(title="Combined Metrics Plots", template="plotly_white")

pio.write_html(fig_combined, "Combined_interactive_plot.html")
print("Interactive combined plot saved as Combined_interactive_plot.html")

# ========================================
# 3. 按照方法（pm_names）分组，绘制横坐标为方法，纵坐标为 test_MSE_loss 的折线图
# ========================================

# 创建两个字典，用于存放每个方法对应的 best_test_avg 和 best_test_std
grouped_test_loss = {method: [] for method in pm_names.values()}
grouped_test_std = {method: [] for method in pm_names.values()}

# 遍历 data_dict 中所有组合，提取最佳指标（best_test_avg, best_test_std）
# 注意：label 的开头为对应的方法名（因为 dir_name 为 ""），因此可以使用 startswith 判断
for label, (mean_steps, mean_loss, std_loss, best_metrics) in data_dict.items():
    best_test_avg = best_metrics[3]
    best_test_std = best_metrics[4]
    for method in pm_names.values():
        if label.startswith(method):
            grouped_test_loss[method].append(best_test_avg)
            grouped_test_std[method].append(best_test_std)
            break

# 按照 pm_names 中的顺序进行排序
ordered_methods = [pm_names[pm] for pm in sorted(pm_names.keys())]

# 计算每个方法下的均值和标准差（若该方法有多个组合，则取均值与 std）
method_list = []
avg_test_loss_list = []
std_test_loss_list = []
for method in ordered_methods:
    values = grouped_test_loss[method]
    stds = grouped_test_std[method]
    # 如果没有数据，则跳过
    if len(values) == 0:
        continue
    method_list.append(method)
    avg_test_loss_list.append(np.mean(values))
    std_test_loss_list.append(np.std(values))

# 使用 Plotly 绘制结果，利用误差棒展示标准差
fig_method = go.Figure()

fig_method.add_trace(go.Scatter(
    x=method_list,
    y=avg_test_loss_list,
    mode='lines+markers',
    name='Test MSE Loss',
    error_y=dict(
        type='data',
        array=std_test_loss_list,
        visible=True
    )
))

fig_method.update_layout(
    title="Test MSE Loss per Method",
    xaxis_title="Method",
    yaxis_title="Test MSE Loss",
    template="plotly_white"
)

pio.write_html(fig_method, "Test_MSE_Loss_per_Method.html")
print("Interactive plot saved as Test_MSE_Loss_per_Method.html")

# # ========================================
# # 4. 绘制每个 method 的曲线
# # 要求每个 method 画一条曲线，横坐标为 label，纵坐标为 test_MSE_loss
# # ========================================
#
# import collections
#
# # 分组：key 为方法名称，value 为列表，每个元素为 (label, best_test_avg, best_test_std)
# method_data = collections.defaultdict(list)
# for label, (mean_steps, mean_loss, std_loss, best_metrics) in data_dict.items():
#     best_test_avg = best_metrics[3]
#     best_test_std = best_metrics[4]
#     # 根据 label 的开头确定方法名称
#     for method in pm_names.values():
#         if label.startswith(method):
#             method_data[method].append((label, best_test_avg, best_test_std))
#             break
#
# # 可选：对每个方法的数据按 label 排序（这里按字母顺序排序）
# for method in method_data:
#     method_data[method].sort(key=lambda x: x[0])
#
# # 绘制每个 method 的曲线，每条曲线横坐标为对应的 label，纵坐标为 best_test_avg，误差棒为 best_test_std
# fig_methods_curve = go.Figure()
# for method, items in method_data.items():
#     labels_method = [item[0] for item in items]
#     test_loss_method = [item[1] for item in items]
#     test_loss_std_method = [item[2] for item in items]
#     fig_methods_curve.add_trace(go.Scatter(
#         x=labels_method,
#         y=test_loss_method,
#         mode='lines+markers',
#         name=method,
#         error_y=dict(
#             type='data',
#             array=test_loss_std_method,
#             visible=True
#         )
#     ))
#
# fig_methods_curve.update_layout(
#     title='Test MSE Loss Curves per Method',
#     xaxis_title='Label',
#     yaxis_title='Test MSE Loss',
#     template='plotly_white'
# )
#
# pio.write_html(fig_methods_curve, 'Test_MSE_Loss_Curves_per_Method.html')
# print('Interactive plot saved as Test_MSE_Loss_Curves_per_Method.html')