import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
import plotly.io as pio

# Load the data
file_path = "AirQualityUCI.csv"
data_all = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)

# Define the features to be analyzed
features_new = ['T', 'RH', 'AH',
                'PT08.S2(NMHC)',
                'PT08.S3(NOx)', 'PT08.S4(NO2)',
                'PT08.S5(O3)', 'PT08.S1(CO)']

# Select the features and drop NaN values
data_pd = data_all[features_new].dropna()

# Define colors based on academic journal conventions
input_color = "#1f77b4"  # Professional blue
sensor_color = "#ff7f0e"  # Professional orange



IDs = np.arange(len(data_pd))

# 手动shuffle & split (8:2)
num_samples = len(data_pd)
indices = np.arange(num_samples)

np.random.seed(42)
random.seed(42)

np.random.shuffle(indices)
train_size = int(0.8 * num_samples)
train_idx = indices[:train_size]
# test_idx = indices[train_size:]

# 构造训练集 (3列: X, y, ID)
data = data_pd.iloc[train_idx]
ID_train = IDs[train_idx]




# Define pruning rates
pruning_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

# Create the figure
fig, axes = plt.subplots(len(pruning_rates), len(features_new), figsize=(15, 12), sharex=False,
                         sharey=False)
                         # sharey=True)
fig.suptitle("Feature Distributions with Different Pruning Rates", fontsize=16)

# Add column labels for input and target
for i, feature in enumerate(features_new[:-1]):
    axes[0, i].set_title(f"{feature}\n(Input)", fontsize=12)
axes[0, -1].set_title(f"{features_new[-1]}\n(Target)", fontsize=12)

# Plot distributions for different pruning rates
for j, rate in enumerate(pruning_rates):
    reduced_data = data.sample(frac=(1 - rate), random_state=42) if rate > 0 else data
    num_samples_reduced = reduced_data.shape[0]

    # Add pruning rate labels to the left-most column without box
    axes[j, 0].annotate(f"Pruning Rate {int(rate * 100)}%\n", xy=(-0.5, 0.5), xycoords='axes fraction',
                        fontsize=12, ha='right', va='center', rotation=90)

    for i, feature in enumerate(features_new):
        axes[j, i].hist(reduced_data[feature].dropna(), bins=50, alpha=0.75,
                        color=input_color if feature != features_new[-1] else sensor_color)
        axes[j, i].tick_params(axis='x', rotation=45)
        if j == (len(pruning_rates)-1):
            axes[j, i].set_xlabel(feature)
        if i == 0:
            axes[j, i].set_ylabel("Frequency")

plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
plt.savefig('1data_distribution_pruning.png')

plot_all_features_distribution = False
if plot_all_features_distribution:
    # Create individual histograms for each feature after dropping NaN values separately
    plt.figure(figsize=(15, 12))
    features_new = ['T', 'RH', 'AH',
                    'CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)',
                    'PT08.S2(NMHC)', 'PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', ]
    # 'NMHC(GT)'
    for i, feature in enumerate(features_new, 1):
        feature_data = data_all[feature].dropna()  # Drop NaN values only for this feature
        num_samples_feature = feature_data.shape[0]  # Count valid samples for this feature

        plt.subplot(3, 4, i)
        plt.hist(feature_data, bins=50, alpha=0.75)
        plt.title(f"{feature} Distribution\nTotal Samples: {num_samples_feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig('all_features_distribution.png')
