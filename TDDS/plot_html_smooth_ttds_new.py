import pandas as pd
import numpy as np
from collections import defaultdict
import re
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from scipy.ndimage import gaussian_filter1d

# Load the CSV file
file_path = "./wandb_export_2025-02-03T12_04_32.783+01_00.csv"
df = pd.read_csv(file_path)

# Extract relevant columns (iteration + val_acc columns)
columns_to_keep = ['iteration'] + [col for col in df.columns if 'val_acc' in col and '__' not in col]
df_cleaned = df[columns_to_keep]

# Drop rows where all val_acc values are NaN
df_cleaned = df_cleaned.dropna(subset=[col for col in df_cleaned.columns if 'val_acc' in col], how='all')

# Forward fill NaN values to maintain continuity in plots
df_cleaned = df_cleaned.ffill()

# Ensure all values are properly converted to float
df_cleaned[columns_to_keep[1:]] = df_cleaned[columns_to_keep[1:]].astype(float)

# Replace infinite values with NaN and drop rows with NaN in validation accuracy columns
df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned.dropna(subset=columns_to_keep[1:], how='all', inplace=True)


# Function to extract method, pruning rate (pr), and seed from column names
def extract_info(column_name):
    match = re.search(r'(.+)_pr([\d.]+)_resnet18_cifar100_seed(\d+)', column_name)
    if match:
        method, pr, seed = match.groups()
        return method, float(pr), int(seed)
    return None, None, None


# Organizing data into the expected dictionary format
data_dict = defaultdict(list)

for col in columns_to_keep[1:]:
    method, pr, seed = extract_info(col)
    if pr==0.9:
        continue
    if method is not None:
        if method in 'TTDS_loss':
            method = method[:4]
            # continue
        elif method in 'ranodom':
            method = 'Random'

        label = f"{method}, PR {pr}"
        data_dict[label].append(df_cleaned[col].values)  # Collecting seed data

# Convert list to numpy arrays and stack along last axis
for key in data_dict.keys():
    data_dict[key] = np.stack(data_dict[key], axis=-1)

iterations = df_cleaned[columns_to_keep[0]].values

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
    # 'rgba(200, 100, 200, 1)',  # 浅紫色
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
    mean_curve_smooth = gaussian_filter1d(mean_curve, sigma=4)
    std_curve_smooth = gaussian_filter1d(std_curve, sigma=4)

    # Ensure no NaN or Infinite values in the computed statistics
    mean_curve_smooth = np.nan_to_num(mean_curve_smooth, nan=0.0)
    std_curve_smooth = np.nan_to_num(std_curve_smooth, nan=0.0)

    # Generate a unique color for each curve
    # color = f'rgba({np.random.randint(50, 200)}, {np.random.randint(50, 200)}, {np.random.randint(50, 200)}, 1)'
    color = color_list[color_ID]
    color_ID +=1

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
        text="Classification accuracy on CIFA100 Over Iterations",
        font=dict(size=20, family="Arial, sans-serif"),
        x=0.5,  # Center title
        y=0.95
    ),
    xaxis_title="Iterations",
    yaxis_title="Accuracy (%)",
    legend_title="Pruning Methods & Rates",
    template="plotly_white",
    font=dict(size=14, family="Arial, sans-serif"),
    xaxis=dict(showgrid=True, zeroline=False, tickmode='linear', dtick=2000, title_font=dict(size=16)),
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
output_path = "/home/zi/research_project/TDDS/random_TTDS_interactive_plot.html"
pio.write_html(fig, output_path)
print(f"Interactive plot saved as {output_path}")
