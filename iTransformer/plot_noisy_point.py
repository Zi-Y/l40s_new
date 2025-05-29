import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

file_path = ("/mnt/ssd/zi/itransformer_results/"
             "trend_scores/seed0_pm0_pr0_low10_high10_start0_int20_tr30_test101_"
             "iTransformer_custom_ftM_sl96_ll48_"
             "pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/trend_error_train_set_all_sample_all_tokens.npy")
trained_per_token_loss = np.load(file_path)
presure_per_token = trained_per_token_loss[:, :, 0]
temp_per_token = trained_per_token_loss[:, :, 1]

cal_gradinet_sim_train_val = True
if cal_gradinet_sim_train_val:
    sample_level_dot_similarity_file = ("/home/local/zi/research_project/iTransformer/"
                      "pm100_checkpoints_dot_val/seed0_pm100_pr0_low10_high10_"
                      "start0_int20_tr10_test101_iTransformer_custom_ftM_"
                      "sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_"
                      "dtTrue_exp_projection_0/"
                      "data_shapley_scores_epoch_10.npy")

    sample_level_cos_similarity_file = ("/home/local/zi/"
                                        "research_project/iTransformer/"
                                        "pm100_checkpoints_cos_sim/"
                                        "seed0_pm100_pr0_low10_high10_start0_"
                                        "int20_tr10_test101_iTransformer_custom_"
                                        "ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_"
                                        "fc1_ebtimeF_dtTrue_exp_projection_0/"
                                        "data_shapley_scores_epoch_10.npy")


    # sample_level_similarity = np.load(sample_level_dot_similarity_file)
    sample_level_similarity = np.load(sample_level_cos_similarity_file)


    def plot_distribution(data, bins=50, smooth_window=5):
        """
        绘制一维数据的分布曲线：先按 bins 划分直方图，计算密度，
        然后对密度值进行简单移动平均平滑，再绘制曲线。

        参数：
        - data: 一维 numpy 数组
        - bins: 直方图的柱子数量，默认 50
        - smooth_window: 用于平滑的窗口大小，默认 5
        """
        # 计算直方图
        counts, bin_edges = np.histogram(data, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 移动平均平滑
        window = np.ones(smooth_window) / smooth_window
        smooth_counts = np.convolve(counts, window, mode='same')

        # 绘制分布曲线
        plt.figure()
        plt.plot(centers, smooth_counts)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    plot_distribution(sample_level_similarity[:, 0], bins=100, smooth_window=7)
    for epoch in range(1, 11):
        arr = sample_level_similarity[:, epoch]
        # 使用 Plotly 的 graph_objects 创建 Figure
        fig = go.Figure(
            data=go.Scatter(
                y=arr,
                # mode='lines+markers',  # 折线并带有数据点标记
                mode='markers',  # 折线并带有数据点标记
                line=dict(width=2),
                marker=dict(size=6)
            ),
            layout=go.Layout(
                title=f"Epoch{epoch} Num组的折线图示例",
                xaxis=dict(title="索引"),
                yaxis=dict(title="值"),
                template="plotly_white"
            )
        )

        # 将图表保存为 HTML 文件
        fig.write_html(f"./gradient_similarity_plot/gradient_cos_epoch{epoch}_line_chart.html", include_plotlyjs="cdn")
        fig.show()
        print('test')




cal_mean_value_of_each_token = False
# 计算每一个token的再每次出现的均值，也就是出现了96次，求和在除以96

if cal_mean_value_of_each_token:
    saved_all_loss = ("/mnt/ssd/zi/iTransformer/"
                      "all_saved_seed0_trained_per_token_loss/"
                      "all_saved_seed0_pm0_pr0_low10_high10_start0_"
                      "int20_test1_iTransformer_custom_ftM_sl96_ll48"
                      "_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_"
                      "exp_projection_0/loss_all_epoch_all_variable_all_stamp0.npy")
    trained_loss = np.load(saved_all_loss)
    trained_loss = np.mean(trained_loss, axis=1)

    # 假设你的 trained_per_token_loss 已经是一个 NumPy 数组，形状为 (W, L, D)
    # W: 窗口数量（这里是 36696）
    # L: 每个窗口的长度（这里是 96）
    # D: 每个时间点的变量维度（这里是 21）
    trained_trend_error = trained_per_token_loss

    mean_per_time_list = []

    for name_id, trained in enumerate([trained_loss, trained_trend_error]):
        W, L, D = trained.shape

        # 计算最终结果的时间步长度：N = 窗口数 + 窗口长度 - 1
        # 每个窗口覆盖 L 个时刻，首窗口从 0 到 L-1，末窗口从 W-1 到 (W-1)+(L-1)
        # 因此总时刻数 N = (W-1)+(L-1) + 1 = W + L - 1
        N = W + L - 1

        # 初始化两个数组：
        # sum_per_time: 用来累加每个绝对时刻下所有窗口对应的损失值之和，形状 (N, D)
        # count_per_time: 用来统计每个绝对时刻下被多少个窗口包含，形状 (N,)
        sum_per_time = np.zeros((N, D), dtype=trained.dtype)
        count_per_time = np.zeros((N,), dtype=np.int32)

        # 遍历每个偏移 offset（相当于窗口中的位置），将所有窗口在该偏移处的值一次性累加
        for offset in range(L):
            # trained[:, offset, :]    —— 形状 (W, D)，代表所有窗口在相对时刻 offset 的 D 维损失
            # sum_per_time[offset:offset+W]
            #   —— 形状也是 (W, D)，对应绝对时刻从 offset 到 offset+(W-1)
            #
            # 举例:
            #  offset=0 时，将 trained[:,0,:] 累加到 sum_per_time[0:W]
            #  offset=1 时，将 trained[:,1,:] 累加到 sum_per_time[1:W+1]
            sum_per_time[offset: offset + W] += trained[:, offset, :]

            # 对应的计数：从 offset 到 offset+W-1 这 W 个绝对时刻，
            # 每个时刻又被当前 offset 这一批 W 个窗口包含了一次
            count_per_time[offset: offset + W] += 1

        # 到此为止：
        # sum_per_time[t, d] 存储的是“绝对时刻 t，变量维度 d”在所有覆盖这个时刻的窗口上的损失之和
        # count_per_time[t] 存储的是“绝对时刻 t”被多少个窗口覆盖

        # 计算均值：在每个时刻 t，把累加和除以对应的窗口覆盖数
        # count_per_time 是 (N,) 形状，扩展到 (N,1) 以便跟 (N,D) 相除
        mean_per_time = sum_per_time / count_per_time[:, None]




        # 把结果放大一遍更好的在pycharm上显示
        mean_per_time = mean_per_time * 1e5
        # 结果：
        # mean_per_time.shape == (N, D)
        # 第 t 行（0 <= t < N）即为“绝对时刻 t”下，21 维变量的平均 loss

        #  Min–Max 归一化（缩放到 [0, 1] 区间）
        mean_per_time_list.append(
            1e5 * (mean_per_time - mean_per_time.min()) / (mean_per_time.max() - mean_per_time.min()))

        vmin, vmax = np.percentile(mean_per_time, [5, 95])

        # 2) 构建 Heatmap，并传入 zmin/zmax
        fig = go.Figure(
            data=go.Heatmap(
                z=mean_per_time,
                x=[f'Var {i + 1}' for i in range(D)],
                y=list(range(N)),
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

        if name_id==1:
            file_name = 'trend_error_mean_per_time_heatmap_clipped.html'
        else:
            file_name = 'loss_mean_per_time_heatmap_clipped.html'

        fig.update_layout(
            title=f'mean_per_time 热力图 {file_name[:-35]}（色条范围 [{vmin:.2f}, {vmax:.2f}]）',
            xaxis_title='变量 维度',
            yaxis_title='时间 步',
            yaxis=dict(autorange='reversed')
        )

        # 导出 HTML 片段
        html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
        with open(file_name, 'w') as f:
            f.write(html_str)

        fig.show()

    mean_per_time = mean_per_time_list[0] - mean_per_time_list[1]
    vmin, vmax = np.percentile(mean_per_time, [5, 95])

    # 2) 构建 Heatmap，并传入 zmin/zmax
    fig = go.Figure(
        data=go.Heatmap(
            z=mean_per_time,
            x=[f'Var {i + 1}' for i in range(D)],
            y=list(range(N)),
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


    file_name = 'difference_loss-trend_error_mean_per_time_heatmap_clipped.html'

    fig.update_layout(
        title=f'mean_per_time 热力图 difference (loss-trend error, 色条范围 [{vmin:.2f}, {vmax:.2f}]）',
        xaxis_title='变量 维度',
        yaxis_title='时间 步',
        yaxis=dict(autorange='reversed')
    )

    # 导出 HTML 片段
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=False)
    with open(file_name, 'w') as f:
        f.write(html_str)

    fig.show()


# 1. 获取最大值
max_val = presure_per_token.max()

# 2. 获取扁平化后最大值的索引
flat_idx = presure_per_token.argmax()

# 3. 将扁平索引转换为多维坐标
coord = np.unravel_index(flat_idx, presure_per_token.shape)

print(f"最大值为：{max_val}")
print(f"坐标为：{coord}")

# 1. 获取最大值
max_val = temp_per_token.max()

# 2. 获取扁平化后最大值的索引
flat_idx = temp_per_token.argmax()

# 3. 将扁平索引转换为多维坐标
coord = np.unravel_index(flat_idx, temp_per_token.shape)

print(f"temp 最大值为：{max_val}")
print(f"temp 坐标为：{coord}")

for v_id, variable in enumerate([presure_per_token, temp_per_token]):
    # 扁平化矩阵
    flat = variable.flatten()

    # 得到降序排列的索引
    sorted_idx_desc = np.argsort(flat)[::-1]

    # 最大的 10 个元素在扁平数组中的索引
    top10_idx = sorted_idx_desc[:10]
    # 最小的 10 个元素在扁平数组中的索引
    bottom10_idx = sorted_idx_desc[-10:]

    # 将扁平索引转换为 (行, 列)
    top10_coords = [np.unravel_index(i, variable.shape) for i in top10_idx]
    bottom10_coords = [np.unravel_index(i, variable.shape) for i in bottom10_idx]

    top10_coords_np = np.array(top10_coords)
    bottom10_coords_np = np.array(bottom10_coords)

    # 打印结果
    print("最大的 10 个元素及其所在列：")
    for idx, (r, c) in zip(top10_idx, top10_coords):
        print(f"值 {flat[idx]:.6f}，列 {c}")

    # print("\n最小的 10 个元素及其所在列：")
    # for idx, (r, c) in zip(bottom10_idx, bottom10_coords):
    #     print(f"值 {flat[idx]:.6f}，列 {c}")

    # 1. 计算每一行的和
    row_sums = variable.sum(axis=1)

    # 2. 对行和进行排序，得到索引数组（从小到大）
    sorted_row_indices = np.argsort(row_sums)

    # 3. 取最小的 5 个行索引
    smallest5_indices = sorted_row_indices[:5]

    # 4. 取出这 5 行对应的行和
    smallest5_sums = row_sums[smallest5_indices]

    print("行和最小的 5 行索引（从 0 开始）：", smallest5_indices)
    print("对应的行和：", smallest5_sums)

    import plotly.graph_objects as go

    # 构造 X 轴数据
    x = list(range(96))

    # 创建 Figure 对象
    fig = go.Figure()

    previous_ID = -100

    # 添加 10 条折线，每条线的 y 值为 x 与系数 (i+1) 的乘积
    for i in range(top10_coords_np.shape[0]):
    # for i in range(1):

        sample_id = top10_coords_np[i,0]

        if previous_ID == -100:
            previous_ID = sample_id
        else:

            if (previous_ID - 96 < sample_id) and (sample_id < previous_ID +96):
                continue
            else:
                previous_ID = sample_id

        noisy_index = top10_coords_np[i,1]
        y = np.load(f"/home/local/zi/research_project/iTransformer/sample_label_plot/sample_{sample_id}.npy")[:,v_id]
        z = np.round(100.0*variable[sample_id], 2)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=f'Sample ID: {sample_id}, max error index: {noisy_index}, value {z[noisy_index]}',
            customdata=z,
            hovertemplate=(
                'x: %{x}<br>'
                'y: %{y}<br>'
                'Trend error: %{customdata}<extra></extra>'
            )
        ))

    previous_ID = -100
    for i in smallest5_indices:
    # for i in smallest5_indices[0:1]:
        sample_id = i
        if previous_ID == -100:
            previous_ID = sample_id
        else:

            if (previous_ID - 96 < sample_id) and (sample_id < previous_ID +96):
                continue
            else:
                previous_ID = sample_id

        z = np.round(100.0*variable[sample_id], 2)
        noisy_index = np.argmax(variable[sample_id])
        y = np.load(f"/home/local/zi/research_project/iTransformer/sample_label_plot/sample_{sample_id}.npy")[:,v_id]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=f'Sample ID: {sample_id}, max error index: {noisy_index}, value {z[noisy_index]}',
            customdata=z,
            hovertemplate=(
                'x: %{x}<br>'
                'y: %{y}<br>'
                'Trend error: %{customdata}<extra></extra>'
            )
        ))

    # 配置布局
    if v_id ==0:
        y_label ='Air presure'
    else:
        y_label = 'Temperature'
    fig.update_layout(
        title='Samples with highest and lowest trend error',
        xaxis_title='time',
        yaxis_title=y_label,
        legend_title='Legend',
        hoverlabel=dict(
            # font_size=14,  # 调整字体大小，避免被裁剪
            namelength=-1  # -1 表示不裁剪名称
        ),
    )

    # 将图表写入 HTML
    if v_id ==0:
        file_name = 'lines_presure.html'
    else:
        file_name = 'lines_temp.html'
    fig.write_html(
        file=file_name,  # 输出文件名
        full_html=True,  # 包括 <html><head>... 等完整页面
        auto_open=False  # 设为 True 则生成后会自动在浏览器打开
    )

    print("已生成 HTML 文件：10_lines.html，双击即可在浏览器中查看。")





print('test')
# top10_coords_np[:,0]
top_index = [33625, 33534, 33626, 33617, 33536, 20368, 33525, 20451, 20452,
       20458, 14857, 14948, 14946, 14934, 14947, 14933, 14945, 14937, 14935,
       14940]
bottom_index = [27941, 27940, 27939, 27942, 27943, 8000, 8001, 7999, 7998, 4890]




