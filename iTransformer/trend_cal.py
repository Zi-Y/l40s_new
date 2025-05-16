import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_positive_distribution(tensor, bins=30):
    """
    绘制 tensor 中大于 0 的元素的分布图（直方图 + KDE）

    参数:
        tensor: 任意形状的 PyTorch 张量
        bins:   直方图的分箱数，默认为 30
    """
    # 取出大于 0 的元素并转换为 NumPy 数组
    pos_vals = tensor[tensor > 0].cpu().numpy()

    if pos_vals.size == 0:
        print("没有大于 0 的元素可供绘制。")
        return

    pos_vals_sorted_desc = pos_vals[np.argsort(pos_vals)[::-1]]
    # 新建画布
    plt.figure()

    # 绘制归一化直方图
    plt.hist(pos_vals, bins=bins, density=True, edgecolor='black', alpha=0.7)

    # # 叠加核密度估计（KDE）曲线
    # kde = gaussian_kde(pos_vals)
    # x_vals = np.linspace(pos_vals.min(), pos_vals.max(), 200)
    # plt.plot(x_vals, kde(x_vals), linewidth=2)

    # 标注
    plt.title('Distribution of Trend Error Elements > 0')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

trend_error = np.load('/mnt/ssd/zi/itransformer_results/trend_scores/seed0_pm0_pr0_low10_high10_start0_int20_tr30_test101_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/trend_error_train_set_all_sample_all_tokens.npy')
trend_ano_score = np.load('/mnt/ssd/zi/itransformer_results/trend_scores/seed0_pm0_pr0_low10_high10_start0_int20_tr30_test101_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/trend_anomaly_score_train_set_all_sample.npy')

trend_ano_score_sorted_desc = trend_ano_score[np.argsort(trend_ano_score[:,1])[::-1]]

trend_error_tensor = torch.tensor(trend_error, dtype=torch.float32)
save_ranking_list = True
lay_back_list = [1, 2, 4, 8, 16, 32, 48, 64, 96]

if save_ranking_list:
    trend_error_avg_dim = np.sum(trend_error, axis=2)
    for lay_back in lay_back_list:
        trend_error_avg_dim_lay_back = trend_error_avg_dim[:, -lay_back:]
        trend_error_avg_dim_lay_back = np.sum(trend_error_avg_dim_lay_back, axis=1)
        ratio_positive = np.mean(trend_error_avg_dim_lay_back > 0.0)
        print(f"lay_back: {lay_back}, 大于 0 的元素的比例: {ratio_positive:.4f}")
        # 1. 获取按值从小到大排序后的索引
        sorted_idx = np.argsort(trend_error_avg_dim_lay_back)

        # 2. 根据这些索引取出对应的值
        sorted_vals = trend_error_avg_dim_lay_back[sorted_idx]

        # 3. 拼成 (n, 2) 的数组：第一列原始索引，第二列对应的值
        result = np.column_stack((sorted_idx, sorted_vals))
        np.save(f'/mnt/ssd/zi/itransformer_results/trend_scores/seed0_pm0_pr0_low10_high10_start0_int20_tr30_test101_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/trend_anomaly_score_train_set_all_sample_lay_back_{lay_back}.npy', result)



flat_diff = trend_error_tensor.contiguous().reshape(-1)

# 计算大于 0 的元素个数
count_pos = (flat_diff > 0).sum().item()

# 计算总元素个数
total = flat_diff.numel()

# 计算比例
ratio_pos = count_pos / total

print(f"大于 0 的元素个数: {count_pos}, 比例: {ratio_pos:.4f}")  # 保留四位小数

plot_positive_distribution(flat_diff, bins=100)

for token_pr_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # 计算近似分位数对应的 kth 索引
    k = max(1, int((1.0 - abs(token_pr_rate)) * flat_diff.numel()))
    threshold, _ = flat_diff.kthvalue(k)

    # 根据阈值生成权重
    weights = (trend_error_tensor <= threshold).float()

    print(f'token_pr_rate {token_pr_rate}, threshold {threshold}, percent of used tokens {100.0 * weights.mean().item()}')