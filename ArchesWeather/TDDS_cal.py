import numpy as np
import os
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle

# Define paths and constants
data_dir = '/mnt/ssd/zi/4xl40s_ag_1_seed3_right_loss_npy/infobatch_loss_values'  # Replace with the directory containing all npy files

# 匹配文件名中的 epoch 和 device 值的正则表达式
pattern = re.compile(r'Epoch_(\d+)_device_(\d+)\.npy')

# 提取所有文件中的 epoch 和 device 值
max_epoch = -1
max_device = -1

for filename in os.listdir(data_dir):
    match = pattern.search(filename)
    if match:
        epoch = int(match.group(1))  # 提取 epoch 值
        device = int(match.group(2))  # 提取 device 值
        max_epoch = max(max_epoch, epoch)
        max_device = max(max_device, device)

# 输出结果
if max_epoch >= 0 and max_device >= 0:
    num_epochs = max_epoch  # 总 epoch 数，不要最后一个epoch。因为这一个epoch sample的数量不全
    num_devices = max_device + 1  # 总 device 数
    print(f"最大 epoch 值为: {max_epoch}，总 epoch 数为: {num_epochs}")
    print(f"最大 device 值为: {max_device}，总 device 数为: {num_devices}")
else:
    print("未找到符合条件的文件。请检查文件夹路径或文件命名规则。")

sample_id_index = -4  # Column index for sample ID
loss_index = -1       # Column index for loss

# Function to load and merge data for all devices in one epoch
def load_and_merge_epoch(epoch_idx, data_dir, num_devices):
    epoch_data = []

    for device_idx in range(num_devices):
        file_path = os.path.join(data_dir, f'Epoch_{epoch_idx}_device_{device_idx}.npy')
        data = np.load(file_path)
        epoch_data.append(data)
    # Merge and sort by sample ID
    merged_data = np.vstack(epoch_data)
    merged_data = merged_data[np.argsort(merged_data[:, sample_id_index])]  # Sort by sample ID
    return merged_data

# Step 1: Load and process all epochs
all_epochs = []
for epoch_idx in range(num_epochs):
    epoch_data = load_and_merge_epoch(epoch_idx, data_dir, num_devices)
    all_epochs.append(epoch_data)

# Step 2: Initialize the result matrix
num_samples = all_epochs[0].shape[0]
loss_differences = np.zeros((num_samples, num_epochs))  # Matrix to store loss differences
loss_differences[:, 0] = all_epochs[0][:, sample_id_index]  # Set the first column as sample IDs

loss_curves = np.zeros((num_samples, num_epochs+1))
loss_curves[:, 0] = all_epochs[0][:, sample_id_index]

# Step 3: Calculate loss differences
for epoch_idx in range(1, num_epochs):
    loss_diff = all_epochs[epoch_idx][:, loss_index] - all_epochs[epoch_idx - 1][:, loss_index]
    loss_differences[:, epoch_idx] = loss_diff
    loss_curves[:, epoch_idx] = all_epochs[epoch_idx-1][:, loss_index]

loss_curves[:, -1] = all_epochs[num_epochs-1][:, loss_index]

# Preview the result
print("All epoch data saved. Shape of one epoch data:", all_epochs[0].shape)
print("Loss differences matrix saved. Shape:", loss_differences.shape)
# print("Preview of the loss differences matrix (first 5 rows):")
# print(loss_differences[:5, :])


# 提取样本 ID 和 loss_differences 的值
sample_ids = loss_differences[:, 0]  # 样本 ID
loss_values = loss_differences[:, 1:]  # g_t(x_n)，去掉样本 ID

# 时间步总数 T
T = loss_values.shape[1]

# 计算 R(x_n)
R_xn = np.zeros(len(sample_ids))  # 初始化 R(x_n) 的结果

for i in range(len(sample_ids)):
    g_t = np.abs(loss_values[i, :])  # 取 |g_t(x_n)|
    g_mean = np.mean(g_t)            # 计算 \overline{g(x_n)}
    R_xn[i] = np.sum((g_t - g_mean) ** 2)  # 按公式求和

# 合并样本 ID 和 R(x_n)
result = np.column_stack((sample_ids, R_xn))

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(R_xn, bins=200, edgecolor='black', alpha=0.7)
plt.title('Histogram of $\\mathcal{R}(x_n)$', fontsize=16)
plt.xlabel('$\\mathcal{R}(x_n)$ Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 根据 R_xn 对样本 ID 排序
sorted_indices = np.argsort(R_xn)  # 获取排序索引
sorted_sample_ids = sample_ids[sorted_indices]  # 按索引对样本 ID 排序
sorted_sample_ids = sorted_sample_ids.astype(int)
sorted_R_xn_values = R_xn[sorted_indices]  # 按索引对 R_xn 排序
# 合并排序后的结果
sorted_result = np.column_stack((sorted_sample_ids, sorted_R_xn_values))

save_mean_of_loss_ranking_npy = False
if save_mean_of_loss_ranking_npy:
    # Step 1: Exclude the 0th column (sample IDs) and compute row means for remaining columns
    row_means = np.mean(loss_curves[:, 1:], axis=1)  # Compute means row-wise excluding the 0th column

    # Step 2: Get indices to sort rows by their means
    sorted_indices = np.argsort(row_means)  # Indices that sort the array

    # Step 3: Sort the entire `loss_curves` array by these indices
    sorted_loss_curves = row_means[sorted_indices]
    sorted_real_ID = loss_curves[sorted_indices][:,0]
    output_path_sorted = "/home/zi/research_project/ArchesWeather/4xl40s_ag_1_seed3_mean_loss_default_320k.npy"  # 替换为你的保存路径
    np.save(output_path_sorted, sorted_indices)



save_ranking_npy = False
if save_ranking_npy:
    # 保存排序结果
    sorted_sample_ids = sorted_sample_ids - 4
    output_path_sorted = "/home/zi/research_project/ArchesWeather/4xl40s_ag_1_seed3_TDDS_sam2lar.npy"  # 替换为你的保存路径
    np.save(output_path_sorted, sorted_sample_ids)

#
plot_loss_curve = False
if plot_loss_curve:

    indices = np.linspace(0, num_samples - 1, 5, dtype=int)
    selected_index = sorted_sample_ids[indices] - 4

    selected_loss_curves = loss_curves[selected_index]
    # Extract sample IDs and loss values for plotting
    sample_ids = selected_loss_curves[:, 0].astype(int)
    loss_values = selected_loss_curves[:, 1:]

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(loss_values):
        plt.plot(range(1, loss_values.shape[1] + 1), losses, label=f'Sample ID: {sample_ids[i]}')

    # Customize the plot
    plt.title('Loss Curves for Selected Samples, from TDDS small to large')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

plot_month_day_histo = False
if plot_month_day_histo:
    # Load the file to inspect its contents
    file_path = '/home/zi/research_project/ArchesWeather/sample_list.pkl'
    # Attempt to load the file using pickle
    try:
        with open(file_path, 'rb') as file:
            pickle_data = pickle.load(file)
            # Display the first 5 entries for clarity
            pickle_data_preview = {key: pickle_data[key] for key in list(pickle_data.keys())[:5]}
    except Exception as e:
        str(e)

    # Extract month and day for each entry in the data
    month_day_dict = {key: (value[2].astype('datetime64[M]').item().month, value[2].astype('datetime64[D]').item().day)
                      for key, value in pickle_data.items()}

    month_day_array = np.array(list(month_day_dict.values()))

    # for keep_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for keep_rate in [0.3, -0.3, 0.5, -0.5, 0.7, -0.7,]:
        if keep_rate > 0.0:
            after_pruning_sample_ids = sorted_sample_ids[:int(len(sorted_sample_ids) * keep_rate)]
        else:
            after_pruning_sample_ids = sorted_sample_ids[int(len(sorted_sample_ids) * keep_rate):]

        remaining_samples_month_day = month_day_array[after_pruning_sample_ids]
        # Plot histogram for days
        days = month_day_array[:, 1]
        months = month_day_array[:, 0]

        # Calculate frequencies for days and months
        unique_days, day_counts = np.unique(days, return_counts=True)
        unique_months, month_counts = np.unique(months, return_counts=True)

        # Calculate frequencies for 100% data
        unique_days_all, day_counts_all = np.unique(days, return_counts=True)
        unique_months_all, month_counts_all = np.unique(months, return_counts=True)

        # Calculate frequencies for 50% sampled data
        sampled_days = remaining_samples_month_day[:, 1]
        sampled_months = remaining_samples_month_day[:, 0]

        unique_days_sampled, day_counts_sampled = np.unique(sampled_days, return_counts=True)
        unique_months_sampled, month_counts_sampled = np.unique(sampled_months, return_counts=True)

        # Calculate proportions
        proportions_days = day_counts_sampled / day_counts_all
        proportions_months = month_counts_sampled / month_counts_all

        # Plot bar chart for days with proportions as percentages
        plt.figure(figsize=(12, 6))
        plt.bar(unique_days_all, day_counts_all, label='Full Samples', align='center', edgecolor='black', color='steelblue',
                alpha=0.7)
        plt.bar(unique_days_sampled, day_counts_sampled, label='Remaining Samples', align='center', edgecolor='black',
                color='darkorange', alpha=0.9)

        # Add percentage labels on top of the bars for sampled data
        for i, (x, proportion) in enumerate(zip(unique_days_sampled, proportions_days)):
            plt.text(x, day_counts_sampled[i] + 5, f"{proportion * 100:.0f}%", ha='center', fontsize=10, color='black')

        plt.title(f'Frequency of Days: Full vs Sampled Data with Proportions {100.0*keep_rate}', fontsize=14)
        plt.xlabel('Day of the Month', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(unique_days_all, fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # Plot bar chart for months with proportions as percentages
        plt.figure(figsize=(12, 6))
        plt.bar(unique_months_all, month_counts_all, label='Full Samples', align='center', edgecolor='black',
                color='steelblue', alpha=0.7)
        plt.bar(unique_months_sampled, month_counts_sampled, label='Remaining Samples', align='center', edgecolor='black',
                color='darkorange', alpha=0.9)

        # Add percentage labels on top of the bars for sampled data
        for i, (x, proportion) in enumerate(zip(unique_months_sampled, proportions_months)):
            plt.text(x, month_counts_sampled[i] + 5, f"{proportion * 100:.0f}%", ha='center', fontsize=10, color='black')

        plt.title(f'Frequency of Months: Full vs Sampled Data with Proportions {100.0*keep_rate}', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(unique_months_all, fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

compare_other_rank = False
if compare_other_rank:
    for keep_rate in [0.1, 0.3, 0.5, 0.7, 0.9,]:
        s2l_50 = np.load(f'/home/zi/research_project/ArchesWeather/4xV100_ag_1_graphcast_seed3_loss_kmean_550_k{int(100.0*keep_rate)}.npy')

        # 4xV100_ag_1_graphcast_seed3_loss_kmean_550_k70
        after_pruning_sample_ids = sorted_sample_ids[-int(len(sorted_sample_ids) * keep_rate):]
        after_pruning_sample_ids = after_pruning_sample_ids - 4

        # Find the intersection of the two arrays
        intersection = np.intersect1d(s2l_50, after_pruning_sample_ids)
        print(f'S2L - keep rate {keep_rate}, % of intersection: {len(intersection)/len(after_pruning_sample_ids)}')

        mean_loss_seed3 = np.load(
            f'/home/zi/research_project/ArchesWeather/mean_loss_seed3_default_320k.npy')

        mean_loss_seed3 = mean_loss_seed3[-int(len(sorted_sample_ids) * keep_rate):]

        intersection = np.intersect1d(mean_loss_seed3, after_pruning_sample_ids)
        print(f'mean_loss_seed3 - keep rate {keep_rate}, % of intersection: {len(intersection)/len(after_pruning_sample_ids)}')

        print('\n')









# 打印结果预览
print("计算完成，结果如下：")
print(result[:5])  # 打印前 5 行预览