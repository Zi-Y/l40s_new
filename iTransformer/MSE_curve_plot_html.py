import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
import plotly.io as pio

# ========================================
# 1. 遍历所有组合（pm, pr）并加载各 seed 的 test_loss 曲线
# ========================================

base_dir = "./checkpoints"
pm_values = [0, 1, 2, 3, 4]
pm_values = [0, 4]
pr_values = [0, 10, 30, 50, 70, 90]
seeds = range(51)  # seed 从 0 到 51
# seeds = range(40)
# seeds = range(16, 51)
# seed 在5一下结果不太准确，因为infobatch有问题
# seeds = range(5, 51)
# seed 从 0 到 51
pm_names = {
    0: "Random",
    1: "Avg Loss",
    2: "S2L",
    3: "TTDS",
    4: "Infobatch",
}
# 用 data_dict 保存每个组合的平均曲线和标准差，字典的键形如 "pm0, PR 90"
data_dict = {}

for pm in pm_values:
    for pr in pr_values:
        if pr == 0:
            if pm!=0:
                continue
        # 构造模型文件夹的固定部分（去掉 seed 前缀）
        model_part = f"pm{pm}_pr{pr}_ECL_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"
        # model_part = f"pm{pm}_pr{pr}_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0"

        label = f"{pm_names[pm]}, PR {pr}"

        # 第一遍：遍历各个 seed，找出所有 seed 中最大的 epoch 编号
        global_epoch_max = -1
        # global_epoch_max = 23
        # 临时存储每个 seed 的数据，存为 tuple：(epoch_results字典, best_value)
        seed_data_list = []
        for seed in seeds:
            folder_name = f"seed{seed}_{model_part}"
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} 不存在")
                continue
            # 加载该 seed 下所有文件（要求文件名形如 epoch_{epoch}_results.npy）
            epoch_results = {}
            for file in os.listdir(folder_path):
                m = re.match(r"epoch_(\d+)_results\.npy", file)
                if m:
                    epoch = int(m.group(1))
                    data = np.load(os.path.join(folder_path, file))
                    # 假设数据格式为 np.array([step, train_loss, vali_loss, test_loss])
                    epoch_results[epoch] = data[-1]
            if not epoch_results:
                continue
            # 更新该组合的全局最大 epoch
            local_max = max(epoch_results.keys())
            if local_max > global_epoch_max:
                global_epoch_max = local_max
            # 找出该 seed 中 test_loss 最小的那个 epoch，并取其数值作为最佳值
            best_epoch = min(epoch_results, key=epoch_results.get)
            best_value = epoch_results[best_epoch]
            seed_data_list.append((epoch_results, best_value))

        if global_epoch_max < 0 or len(seed_data_list) == 0:
            print(f"No data found for combination {label}")
            continue

        # 全局总 epoch 数（注意 epoch 从 0 开始）
        global_epoch_count = global_epoch_max + 1

        # 第二遍：对每个 seed 构造完整曲线，若缺失某些 epoch则用该 seed 的最佳值补全
        seed_curves = []
        for (epoch_results, best_value) in seed_data_list:
            full_curve = []
            for epoch in range(global_epoch_count):
                if epoch in epoch_results:
                    full_curve.append(epoch_results[epoch])
                else:
                    full_curve.append(best_value)
            seed_curves.append(full_curve)

        # 将所有 seed 的曲线转换为 numpy 数组后，计算均值和标准差（在 seed 维度上）
        seed_data_array = np.array(seed_curves)  # shape: (num_seeds, global_epoch_count)
        mean_curve = np.nanmean(seed_data_array, axis=0)
        std_curve = np.nanstd(seed_data_array, axis=0)

        # 可选：使用高斯滤波进行平滑处理
        mean_curve_smooth = gaussian_filter1d(mean_curve, sigma=0.1)
        std_curve_smooth = gaussian_filter1d(std_curve, sigma=0.1)

        # 保存数据，值为 (平滑后的均值, 平滑后的标准差, 总 epoch 数)
        data_dict[label] = (mean_curve_smooth, std_curve_smooth, global_epoch_count)

# ========================================
# 2. 使用 Plotly 绘图，确保阴影区域与主曲线强绑定
# ========================================

fig = go.Figure()
color_list = [
    'rgba(50, 100, 150, 1)',
    'rgba(200, 50, 50, 1)',
    'rgba(50, 200, 50, 1)',
    'rgba(200, 150, 50, 1)',
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
    'rgba(200, 100, 50, 1)'
]
color_index = 0

for label, (mean_curve_smooth, std_curve_smooth, global_epoch_count) in data_dict.items():
    x_axis = np.arange(global_epoch_count)
    color = color_list[color_index % len(color_list)]
    color_index += 1
    # 阴影颜色设为半透明
    shadow_color = color.replace("1)", "0.3)")

    # 添加主曲线 trace，设置 legendgroup 绑定阴影区域
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=mean_curve_smooth,
        mode='lines',
        name=label,
        legendgroup=label,
        line=dict(width=3, shape='spline', smoothing=0.0, color=color),
        hoverinfo='x+y+name'
    ))

    # 添加阴影区域 trace，设置 legendgroup 与主曲线一致，同时 showlegend=False 防止重复显示图例
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_axis, x_axis[::-1]]),
        y=np.concatenate([mean_curve_smooth - std_curve_smooth, (mean_curve_smooth + std_curve_smooth)[::-1]]),
        fill='toself',
        fillcolor=shadow_color,
        line=dict(color='rgba(255,255,255,0)'),
        legendgroup=label,
        showlegend=False,
        hoverinfo='skip'
    ))

fig.update_layout(
    title=dict(text="各组合的 Test MSE Loss 平均曲线", font=dict(size=20), x=0.5),
    xaxis_title="Epoch",
    yaxis_title="Test MSE Loss",
    template="plotly_white",
    hoverlabel=dict(
        # font_size=14,  # 调整字体大小，避免被裁剪
        namelength=-1  # -1 表示不裁剪名称
    ),
)

output_path = "interactive_plot.html"
pio.write_html(fig, output_path)
print(f"Interactive plot saved as {output_path}")