import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Load the CSV file
file_path = "wandb_export_2025-01-30T13_46_37.442+01_00.csv"  # 修改为你的文件路径
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


# Plot grouped means and variances
plt.figure(figsize=(12, 6))

for base_name, cols in grouped_curves.items():
    cleaned_label = clean_label_name(base_name)
    data = df_cleaned[cols].to_numpy(dtype=np.float64)
    mean_curve = np.nanmean(data, axis=1)
    std_curve = np.nanstd(data, axis=1)

    # Ensure no NaN or Infinite values in the computed statistics
    mean_curve = np.nan_to_num(mean_curve, nan=0.0)
    std_curve = np.nan_to_num(std_curve, nan=0.0)

    # Plot mean curve
    plt.plot(df_cleaned['iteration'], mean_curve, label=cleaned_label, linewidth=2)

    # Fill variance region
    plt.fill_between(df_cleaned['iteration'].values, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

plt.xlabel("Iteration")
plt.ylabel("Validation Accuracy")
plt.title("Mean Validation Accuracy with Variance (Grouped by Category)")
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.savefig("output.png")
# plt.show()