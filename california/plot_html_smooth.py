import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from scipy.ndimage import gaussian_filter1d

# Define parameters
pm_values = [-1, 1, 3, 4]  # 使用的 pm 值（方法顺序按此列表排序）
# pm_values = [-1,]  # 使用的 pm 值（方法顺序按此列表排序）
# pr_values = [0.3, 0.5, 0.7, 0.9]  # pr 值（在每个方法内部按此顺序排序）
pr_values = [0.0, 0.3, 0.5, 0.7,]  # pr 值（在每个方法内部按此顺序排序）
pr_values = [0.0, 0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7, -0.9]  # pr 值（在每个方法内部按此顺序排序）
# seeds = range(2001)  # Seeds from 0 to 2000
seeds = range(201)  # Seeds from 0 to 2000
iterations = np.arange(10, 1010, 10)  # Assuming results saved every 10 iterations

data_dict = {}

# Naming mapping for pm values
pm_names = {
    -1: "Random",
    0: "Avg Loss",
    1: "S2L",
    2: "TTDS",
    # 3: "Avg Loss (No Weight)",
    3: "Avg Loss",
    # 4: "TTDS (No Weight)"
    4: "TTDS"
}

# Load .npy files dynamically
for pm in pm_values:
    for pr in pr_values:

        if (pr == 0.0):
            if not (pm == -1 or pm == -2):
                continue
        if (pr < 0.0):
            if (pm == -1 or pm == 1):
                continue

        all_seed_data = []
        for seed in seeds:
            file_path = f"./results_california_data_pruning_singlePR_new/pm_{pm}/pr_{int(pr * 100)}/results_seed{seed}_pr{pr}_pm{pm}_metric_epoch30_metric_seed0.npy"
            if os.path.exists(file_path):
                data = np.load(file_path)
                all_seed_data.append(data)
            else:
                print(f"Warning: File not found - {file_path}")

        if all_seed_data:
            label = pm_names.get(pm, f"PM {pm}")  # Use custom names if available, else default pmX format
            data_dict[f"{label}, PR {pr}"] = np.stack(all_seed_data, axis=-1)  # Stack across seeds
color_list = [
    'rgba(50, 100, 150, 1)',   # 深蓝色
    'rgba(200, 50, 50, 1)',    # 红色
    'rgba(50, 200, 50, 1)',    # 亮绿色
    'rgba(200, 150, 50, 1)',   # 橙色
    'rgba(50, 50, 200, 1)',    # 纯蓝色
    'rgba(150, 50, 200, 1)',   # 深紫色
    'rgba(50, 200, 200, 1)',   # 青色
    'rgba(200, 50, 150, 1)',   # 玫红色
    'rgba(100, 50, 50, 1)',    # 深棕色
    'rgba(150, 200, 50, 1)',   # 黄绿色
    'rgba(100, 100, 200, 1)',  # 淡蓝色
    'rgba(200, 100, 200, 1)',  # 浅紫色
    'rgba(50, 150, 100, 1)',   # 绿色
    'rgba(100, 200, 150, 1)',  # 浅绿色
    'rgba(200, 200, 50, 1)',   # 金黄色
    'rgba(50, 50, 100, 1)',    # 深蓝灰色
    'rgba(200, 50, 200, 1)',   # 粉紫色
    'rgba(50, 200, 100, 1)',   # 青绿色
    'rgba(100, 50, 200, 1)',   # 蓝紫色
    'rgba(200, 100, 50, 1)'    # 砖红色
]
color_ID = 0
# Create interactive plot
fig = go.Figure()

for key, data in data_dict.items():
    mean_curve = np.nanmean(data, axis=-1)  # Mean over seeds
    std_curve = np.nanstd(data, axis=-1)  # Std over seeds

    # Smooth the curves using Gaussian filter
    mean_curve_smooth = gaussian_filter1d(mean_curve, sigma=0.1)
    std_curve_smooth = gaussian_filter1d(std_curve, sigma=0.1)

    # Ensure no NaN or Infinite values in the computed statistics
    mean_curve_smooth = np.nan_to_num(mean_curve_smooth, nan=0.0)
    std_curve_smooth = np.nan_to_num(std_curve_smooth, nan=0.0)

    # Generate a unique color for each curve
    # color = f'rgba({np.random.randint(50, 200)}, {np.random.randint(50, 200)}, {np.random.randint(50, 200)}, 1)'
    color = color_list[color_ID]
    color_ID +=1
    if color_ID >= len(color_list):
        color_ID = 0
    shadow_color = color.replace("1)", "0.3)")  # Make the shadow very transparent

    # Add mean curve with smooth line and hover effects
    fig.add_trace(go.Scatter(
        x=iterations,
        y=mean_curve_smooth,
        mode='lines',
        name=key,
        line=dict(width=3, shape='spline', smoothing=0.0, color=color),
        hoverinfo='x+y+name',
        opacity=0.9,
        legendgroup=key,  # Bind curve and shaded area together
        showlegend=True
    ))

    # Add variance region as smooth shaded area (linked to mean curve visibility)
    fig.add_trace(go.Scatter(
        x=np.concatenate([iterations, iterations[::-1]]),
        y=np.concatenate([mean_curve_smooth - std_curve_smooth, (mean_curve_smooth + std_curve_smooth)[::-1]]),
        fill='toself',
        fillcolor=shadow_color,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip',  # Prevent hover display on variance area
        legendgroup=key,  # Ensures it toggles together with the main curve
        visible=True
    ))

fig.update_layout(
    title=dict(
        text="Avg MSE Scores On California Housing Prices Dataset Over Iterations",
        font=dict(size=20, family="Arial, sans-serif"),
        x=0.5,  # Center title
        y=0.95
    ),
    xaxis_title="Iterations",
    yaxis_title="MSE Scores",
    legend_title="Pruning Methods & Rates",
    template="plotly_white",
    font=dict(size=14, family="Arial, sans-serif"),
    xaxis=dict(showgrid=True, zeroline=False, tickmode='linear', dtick=20, title_font=dict(size=16)),
    yaxis=dict(showgrid=True, zeroline=False, title_font=dict(size=16)),
    hovermode="closest",  # Remove vertical hover line
    margin=dict(l=60, r=60, t=60, b=60),
    plot_bgcolor="white",
    hoverlabel=dict(
        # font_size=14,  # 调整字体大小，避免被裁剪
        namelength=-1  # -1 表示不裁剪名称
    )
)

# Save as HTML
output_path = "/home/zi/research_project/california/random_pruning_interactive_plot.html"
pio.write_html(fig, output_path)
print(f"Interactive plot saved as {output_path}")
