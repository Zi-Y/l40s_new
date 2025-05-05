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
file_path = "./wandb_export_2025-02-11T15_54_21.912+01_00.csv"
df = pd.read_csv(file_path)

# Define relevant columns for plotting
y_columns = {
    # "TDDS keep 30% samples": "TDDS keep 30% samples - val_loss",
    "Random, PR 0.0": "No pruning - val_loss",
    "Random, PR 0.3": "seed3_70%_normal_test_loss - val_loss",
    "Random, PR 0.5": "seed3_50%_normal_test_loss - val_loss",
    "Random, PR 0.7": "Random pruning keep 30% samples - val_loss",
}

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

# Create an interactive figure
fig = go.Figure()

# Function to extract method, pruning rate (pr), and seed from column names



# Add traces for each method
for label, col in y_columns.items():
    valid_data = df[["trainer/global_step", col]].dropna()
    # Generate a unique color for each curve
    # color = f'rgba({np.random.randint(50, 200)}, {np.random.randint(50, 200)}, {np.random.randint(50, 200)}, 1)'
    color = color_list[color_ID]
    color_ID +=1
    fig.add_trace(go.Scatter(
        x=valid_data["trainer/global_step"],
        y=valid_data[col],
        mode='lines',
        name=label,
        line=dict(width=3, shape='spline', smoothing=0.0, color=color),
        hoverinfo='x+y+name',
        opacity=0.9,
        legendgroup=label,  # Bind curve and shaded area together
        showlegend=True
    ))

fig.update_layout(
    title=dict(
        text="Weighted mean of MSE scores on Era5 Over Iterations",
        font=dict(size=20, family="Arial, sans-serif"),
        x=0.5,  # Center title
        y=0.95
    ),
    xaxis_title="Iterations",
    yaxis_title="MSE scores",
    legend_title="Pruning Methods & Rates",
    template="plotly_white",
    font=dict(size=14, family="Arial, sans-serif"),
    xaxis=dict(showgrid=True, zeroline=False, range=[0, 310000],
               tickmode='linear', dtick=10000, title_font=dict(size=16)),
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
output_path = "./Era5_interactive_plot.html"
pio.write_html(fig, output_path)
print(f"Interactive plot saved as {output_path}")
