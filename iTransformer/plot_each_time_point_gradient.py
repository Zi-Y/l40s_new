import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os


metric_id = 3
# for metric_id in [0, 1, 2, 3, 10, 11, 12, 13]:
# for metric_id in [0, 2, 3, 10, 12, 13]:
for metric_id in [20]:

    file_path = ("/home/local/zi/research_project/"
                 "iTransformer/checkpoints_pm101_all_time/"
                 "seed0_pm101_pr0_low10_high10_start0_int20_"
                 "tr10_test101_iTransformer_custom_ftM_sl96_"
                 "ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_"
                 "ebtimeF_dtTrue_exp_projection_0/"
                 "data_shapley_l2_dot_cos_scores_epoch_10.npy")

    sample_level_cos_similarity_file = ("/home/local/zi/research_project/"
                                        "iTransformer/checkpoints_test/"
                                        "seed0_pm100_pr0_low10_high10_start0_i"
                                        "nt20_tr10_test101_iTransformer_custom_ftM_"
                                        "sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_"
                                        "ebtimeF_dtTrue_exp_projection_0/"
                                        "data_shapley_l2_dot_cos_scores_epoch_10.npy")
    all_time_point_all_variable_all_epoch_loss = ("/mnt/ssd/zi/iTransformer/all_saved_seed0_"
                                                  "trained_per_token_loss/"
                                                  "all_saved_seed0_pm0_pr0_low10_high10_"
                                                  "start0_int20_test1_iTransformer_custom_"
                                                  "ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_"
                                                  "ebtimeF_dtTrue_exp_projection_0/"
                                                  "loss_all_epoch_all_variable_all_stamp0.npy")
    if metric_id <= 9:
        trained_per_time_point_metric = np.load(file_path)
    elif metric_id <= 19:
        trained_per_time_point_metric = np.load(sample_level_cos_similarity_file)
    elif metric_id <= 29:
        trained_per_time_point_metric = np.load(all_time_point_all_variable_all_epoch_loss)

    # data_shapley_scores[idx_in_full_dataset, epoch, time_point_index, 0] = 100.0 * l2_norm_flat_train_grad_value
    # data_shapley_scores[idx_in_full_dataset, epoch, time_point_index, 1] = 100.0 * l2_norm_flat_val_grad.item()  # l2_norm_flat_val_grad已经是标量tensor
    # data_shapley_scores[idx_in_full_dataset, epoch, time_point_index, 2] = 100.0 * actual_dot_product_value
    # data_shapley_scores[idx_in_full_dataset, epoch, time_point_index, 3] = 100.0 * cosine_similarity_value

    # trained_per_time_point_metric = np.load(sample_level_cos_similarity_file)
    for epoch in range(11):
        if metric_id <= 9:
            # 因为epoch3的test loss最低，所以使用1-3个epoch的值
            cosine_similarity_per_time_point_epoch = trained_per_time_point_metric[:, :epoch+1, :, metric_id]
            # 求每一个epoch的平均
            cosine_similarity_per_time_point_epoch = np.mean(cosine_similarity_per_time_point_epoch, axis=1)


            # 假设已有 cosine_similarity_per_time_point_epoch，形状 (36696, 96)
            # 这里为了演示，先随便造一个：
            # cosine_similarity_per_time_point_epoch = np.random.random((36696, 96))

            # 1. 读取原始形状
            num_samples, window_len = cosine_similarity_per_time_point_epoch.shape
            # num_samples = 36696, window_len = 96

            # 2. 计算总时间点数
            total_time_points = num_samples + window_len - 1  # 36696 + 96 - 1 = 36791

            # 3. 初始化新矩阵 M，全部填 None
            M = np.full((total_time_points, window_len), None, dtype=object)

            # 4. 将原来每列 j 的数据，复制到 M 第 j 列对应的行区间 [j : j + num_samples]
            for j in range(window_len):
                # 原来 cosine_similarity_per_time_point_epoch[:, j] 对应的起始绝对时间 t = i + j，其中 i 从 0 到 num_samples-1
                # 所以那个区间 t ∈ [j, j + num_samples - 1]，总共 num_samples 个点
                M[j : j + num_samples, j] = cosine_similarity_per_time_point_epoch[:, j]

            # 移除未出现96次的样本
            M = M[95:36696]
            print('test')
            # 这样，M.shape == (36791, 96)，并且
            # 对于任何 0 ≤ i < 36696, 0 ≤ j < 96，都有：
            #    M[i + j, j] == cosine_similarity_per_time_point_epoch[i, j]
            # 边界处那些没有“窗口”能覆盖到 (t, j) 的位置，会保持为 None。
            mean_per_time = M

        elif metric_id <= 19:
            if epoch == 0:
                trained_per_time_point_metric = trained_per_time_point_metric[:,:,metric_id-10]
                mean_per_time = trained_per_time_point_metric
            else:
                continue
        elif metric_id <= 29:
            trained_per_time_point_metric_one_epoch = trained_per_time_point_metric[:, epoch, :, :]
            mean_per_time = np.mean(trained_per_time_point_metric_one_epoch, axis=2)



        N, D = mean_per_time.shape
        vmin, vmax = np.percentile(mean_per_time, [0.1, 99.9])

        if metric_id <= 9:
            y=list(range(95, N+95))
            saved_folder = './plot_results_per_time_point/per_time_point_gradient/'
            os.makedirs(saved_folder, exist_ok=True)
        elif metric_id <= 19:
            y=list(range(N))
            saved_folder = './plot_results_per_time_point/per_sample_gradient/'
            os.makedirs(saved_folder, exist_ok=True)
        elif metric_id <= 29:
            y=list(range(N))
            saved_folder = './plot_results_per_time_point/per_time_point_loss/'
            os.makedirs(saved_folder, exist_ok=True)

        # 2) 构建 Heatmap，并传入 zmin/zmax
        fig = go.Figure(
            data=go.Heatmap(
                z=mean_per_time,
                x=[f'Time point {i + 1}' for i in range(D)],
                y=y,
                zmin=vmin,  # colorbar 下限
                zmax=vmax,  # colorbar 上限
                colorbar=dict(
                    title='平均 Loss',
                    ticks='outside',
                    tickvals=[vmin, (vmin + vmax) / 2, vmax],
                    ticktext=[f'{vmin:.2f}', f'{(vmin + vmax) / 2:.2f}', f'{vmax:.2f}']
                )
            )
        )

        if metric_id == 3:
            file_name = f'cosine_similarity_token_and_val_per_time_point_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 2:
            file_name = f'dot_product_token_and_val_per_time_point_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 1:
            file_name = f'l2_norm_val_gradient_per_time_point_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 0:
            file_name = f'l2_norm_train_gradient_per_time_point_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 10:
            file_name = f'l2_norm_train_gradient_per_sample_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 11:
            file_name = f'l2_norm_val_gradient_per_sample_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 12:
            file_name = f'dot_product_sample_and_val_per_sample_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 13:
            file_name = f'cosine_similarity_sample_and_val_per_time_point_epoch{epoch}_heatmap_clipped.html'
        elif metric_id == 20:
            file_name = f'loss_per_time_point_epoch{epoch}_heatmap_clipped.html'
        # data_shapley_scores[idx_in_full_dataset, epoch, 0] = 1.0 * l2_norm_flat_train_grad_value
        # data_shapley_scores[idx_in_full_dataset, epoch, 1] = 1.0 * l2_norm_flat_val_grad.item()  # l2_norm_flat_val_grad已经是标量tensor
        # data_shapley_scores[idx_in_full_dataset, epoch, 2] = 100.0 * actual_dot_product_value
        # data_shapley_scores[idx_in_full_dataset, epoch, 3] = 100.0 * cosine_similarity_value

        fig.update_layout(
            title=f'mean_per_time 热力图 {file_name[:-21]}（色条范围 [{vmin:.2f}, {vmax:.2f}]）',
            xaxis_title='变量 维度',
            yaxis_title='时间 步',
            yaxis=dict(autorange='reversed')
        )

        # 导出 HTML 片段
        html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
        with open(saved_folder +file_name, 'w') as f:
            f.write(html_str)

        # fig.show()