import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict
import re

# Load the CSV file
file_path = "wandb_export_2025-01-30T13_46_37.442+01_00.csv"  # 修改为你的文件路径
file_path = "wandb_export_2025-01-30T14_45_14.361+01_00.csv"  # 修改为你的文件路径
df = pd.read_csv(file_path)

# Drop unnecessary columns (keeping only iteration and validation accuracy columns)
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

# Function to extract base name without seed
def extract_base_name(full_name):
    return re.sub(r"_seed\d+", "", full_name)

# Creating grouped dictionary
grouped_curves = defaultdict(list)
for col in columns_to_keep[1:]:
    base_name = extract_base_name(col)
    grouped_curves[base_name].append(col)

# Function to clean label names by removing '_resnet18_cifar100'
def clean_label_name(label):
    return label.replace('_resnet18_cifar100', '')

# Automatically select all curves
selected_labels = list(grouped_curves.keys())

# Create interactive plot
fig = go.Figure()

for base_name in selected_labels:
    cleaned_label = clean_label_name(base_name)
    data = df_cleaned[grouped_curves[base_name]].to_numpy(dtype=np.float64)
    mean_curve = np.nanmean(data, axis=1)
    std_curve = np.nanstd(data, axis=1)

    # Ensure no NaN or Infinite values in the computed statistics
    mean_curve = np.nan_to_num(mean_curve, nan=0.0)
    std_curve = np.nan_to_num(std_curve, nan=0.0)

    # Add mean curve
    fig.add_trace(go.Scatter(
        x=df_cleaned['iteration'],
        y=mean_curve,
        mode='lines',
        name=cleaned_label,
        line=dict(width=2),
        visible=True,
        legendgroup=cleaned_label  # Link mean curve and variance to the same group
    ))

    # Add variance region as shaded area (linked to mean curve visibility)
    fig.add_trace(go.Scatter(
        x=np.concatenate([df_cleaned['iteration'], df_cleaned['iteration'][::-1]]),
        y=np.concatenate([mean_curve - std_curve, (mean_curve + std_curve)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=cleaned_label + " Variance",
        legendgroup=cleaned_label,  # Ensures it toggles together with the main curve
        visible=True  # This ensures it hides when the corresponding curve is hidden
    ))

fig.update_layout(
    title="Mean Validation Accuracy with Variance (Grouped by Category)",
    xaxis_title="Iteration",
    yaxis_title="Validation Accuracy",
    legend_title="Curves",
    template="plotly_white"
)

# Save as HTML
pio.write_html(fig, "../california/interactive_plot.html")
print("Interactive plot saved as interactive_plot.html")
