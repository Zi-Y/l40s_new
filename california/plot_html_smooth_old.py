import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from scipy.ndimage import gaussian_filter1d
import plotly.express as px

# -----------------------------
# 1. 定义参数和数据加载部分
# -----------------------------
pm_values = [-1, 3, 4]  # 使用的 pm 值（方法顺序按此列表排序）
pr_values = [0.3, 0.5, 0.7, 0.9]  # pr 值（在每个方法内部按此顺序排序）
seeds = range(2001)  # 种子编号 0 到 2000
iterations = np.arange(10, 1010, 10)  # 假设每 10 次迭代保存结果

data_dict = {}

# pm 值到方法名称的映射
pm_names = {
    -1: "random",
    0: "avg_loss",
    2: "S2L",
    3: "avg_loss_NoWeight",
    4: "S2L_NoWeight"
}

# 动态加载 .npy 文件数据
for pm in pm_values:
    for pr in pr_values:
        all_seed_data = []
        for seed in seeds:
            file_path = f"./results_california_data_pruning_singlePR_new/pm_{pm}/pr_{int(pr * 100)}/results_seed{seed}_pr{pr}_pm{pm}_metric_epoch30_metric_seed0.npy"
            if os.path.exists(file_path):
                data = np.load(file_path)
                all_seed_data.append(data)
            else:
                print(f"Warning: File not found - {file_path}")
        if all_seed_data:
            label = pm_names.get(pm, f"pm{pm}")
            # key 格式为 "方法名_pr{pr}"，例如 "random_pr0.3"
            data_dict[f"{label}_pr{pr}"] = np.stack(all_seed_data, axis=-1)

# -----------------------------
# 2. 构建 legend 排序依据（按照方法顺序和 pr 值顺序）
# -----------------------------
# 先构建方法排序字典，按照 pm_values 的顺序
method_order = {}
for idx, pm in enumerate(pm_values):
    method_name = pm_names.get(pm, f"pm{pm}")
    method_order[method_name] = idx

# -----------------------------
# 3. 根据最后一个迭代点的均值对方法进行降序排序，用于 hover 显示顺序
# -----------------------------
sorted_keys = sorted(
    data_dict.keys(),
    key=lambda key: np.nanmean(data_dict[key][-1, :]),
    reverse=True  # 降序：值最大的排在最前面
)

# -----------------------------
# 4. 定义辅助函数与颜色设置
# -----------------------------
def hex_to_rgba(hex_color, alpha):
    """
    将 HEX 颜色转换为 RGBA 字符串格式
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# 使用 Plotly Express 内置的颜色序列
color_sequence = px.colors.qualitative.Plotly
color_index = 0

# -----------------------------
# 5. 创建交互式图表并添加数据
# -----------------------------
fig = go.Figure()

# 遍历排序后的 keys，用于 hover 顺序（trace 添加顺序决定 hover 的排列顺序）
for key in sorted_keys:
    data = data_dict[key]
    # 计算均值和标准差（沿着种子维度）
    mean_curve = np.nanmean(data, axis=-1)
    std_curve = np.nanstd(data, axis=-1)

    # 使用高斯滤波平滑曲线（可根据需要调整 sigma 参数）
    mean_curve_smooth = gaussian_filter1d(mean_curve, sigma=0.1)
    std_curve_smooth = gaussian_filter1d(std_curve, sigma=0.1)

    # 去除可能出现的 NaN 或无穷值
    mean_curve_smooth = np.nan_to_num(mean_curve_smooth, nan=0.0)
    std_curve_smooth = np.nan_to_num(std_curve_smooth, nan=0.0)

    # 为当前 trace 分配颜色
    line_color = color_sequence[color_index % len(color_sequence)]
    fill_color = hex_to_rgba(line_color, 0.2)
    color_index += 1

    # 解析 key 得到方法名和 pr 值，例如 "random_pr0.3"
    parts = key.split("_pr")
    method_label = parts[0]
    pr_val = float(parts[1])
    # 计算 legend 排序值：先按方法顺序，再按 pr_values 顺序（乘以 10 以保证方法之间有较大间隔）
    legend_rank = method_order.get(method_label, 100) * 10 + pr_values.index(pr_val)

    # 添加均值曲线 trace，并自定义 hover 信息，同时设置 legendrank 控制 legend 的排序
    fig.add_trace(go.Scatter(
        x=iterations,
        y=mean_curve_smooth,
        mode='lines',
        name=key,
        legendgroup=key,
        legendrank=legend_rank,  # 用于控制 legend 顺序
        line=dict(width=2, color=line_color),
        hovertemplate=f"<b>{key}</b><br>Iteration: %{{x}}<br>MSE: %{{y:.4f}}<extra></extra>"
    ))

    # 添加标准差阴影区域 trace（不显示在 legend 中）
    fig.add_trace(go.Scatter(
        x=np.concatenate([iterations, iterations[::-1]]),
        y=np.concatenate([mean_curve_smooth - std_curve_smooth, (mean_curve_smooth + std_curve_smooth)[::-1]]),
        fill='toself',
        fillcolor=fill_color,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=key + " Variance",
        legendgroup=key,
        hoverinfo='skip'
    ))

# -----------------------------
# 6. 更新图表布局
# -----------------------------
fig.update_layout(
    title={
        'text': "Mean of MSE scores on test set with Variance (Over 2000 Seeds)",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24, 'family': 'Arial', 'color': '#333'}
    },
    hovermode='x unified',  # 统一悬停模式：鼠标悬停时显示所有 trace 在该 x 值的 y 值，排列顺序为添加 trace 的顺序（即根据 y 值降序）
    xaxis=dict(
         title=dict(
             text="Iteration",
             font=dict(size=18, family='Arial', color='#333')
         ),
         tickfont=dict(size=14, family='Arial', color='#333'),
         showgrid=True,
         gridcolor='rgba(200,200,200,0.5)',
         zeroline=False,
         rangeslider=dict(visible=False),  # 关闭下方缩略图
         showspikes=True,                # 开启 x 轴 spikelines
         spikemode='across',             # 竖线横跨整个绘图区域
         spikesnap='cursor',             # 竖线跟随鼠标位置
         spikethickness=1,
         spikecolor='#333'
    ),
    yaxis=dict(
         title=dict(
             text="MSE scores",
             font=dict(size=18, family='Arial', color='#333')
         ),
         tickfont=dict(size=14, family='Arial', color='#333'),
         showgrid=True,
         gridcolor='rgba(200,200,200,0.5)',
         zeroline=False
    ),
    legend=dict(
         title="Methods",
         font=dict(size=14, family='Arial', color='#333'),
         bordercolor='rgba(0,0,0,0.1)',
         borderwidth=1
    ),
    template='plotly_white',
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=80, r=40, t=100, b=80)
)

# -----------------------------
# 7. 保存为 HTML 文件
# -----------------------------
output_path = "/home/zi/research_project/california/interactive_plot.html"
pio.write_html(fig, output_path, auto_open=True)
print(f"Interactive plot saved as {output_path}")