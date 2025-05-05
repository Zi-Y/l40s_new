import os

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go




for seed in [0, 1, 2, 3]:
    # data = np.load(f'/home/local/zi/research_project/iTransformer/checkpoints/attention_seed{seed}'
    #                f'_pm0_pr0_low10_high10_test1_iTransformer_custom_ftM_sl96_ll48_pl192_dm512'
    #                f'_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/'
    #                f'attention_all_epoch_seed_{seed}.npy')
    # data = np.load("/home/local/zi/research_project/"
    #                "iTransformer/checkpoints/"
    #                "attention_150k_seed0_pm0_pr0_low10_high10_start0_int20_test1_iTransformer_"
    #                "custom_ftM_sl96_ll48_pl192_dm512_nh8_el3_dl1_df512_fc1_"
    #                "ebtimeF_dtTrue_exp_projection_0/"
    #                "attention_all_epoch_seed_0.npy")

    data = np.load(f"/home/local/zi/research_project/iTransformer/checkpoints/"
                   f"attention_150k_score_seed{seed}_pm0_"
                   f"pr0_low10_high10_start0_"
                   f"int20_test1_iTransformer_custom_ftM_sl96_"
                   f"ll48_pl192_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_"
                   f"dtTrue_exp_projection_0/attention_all_epoch_seed_{seed}.npy")

    # data = np.load(f'/home/local/zi/research_project/iTransformer/checkpoints/attention_seed{seed}'
    #                f'_pm0_pr0_low10_high10_test1_iTransformer_custom_ftM_sl96_ll48_pl192_dm512'
    #                f'_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/'
    #                f'attention_all_epoch_seed_{seed}.npy')

    for epoch in range(11):
        X = data[:,epoch,:]
        # 2. 对数据进行 L2 归一化，确保每个样本为单位向量
        # X_normalized = normalize(X, norm='l2', axis=1)
        X_normalized = X

        # 3. 使用 DBSCAN 进行聚类
        # 余弦距离 = 1 - 余弦相似度，这里设置 eps = 0.2 对应余弦相似度 0.8
        eps_value = 0.2
        min_samples = 500
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X_normalized)

        # 输出聚类结果统计
        unique_labels, counts = np.unique(labels, return_counts=True)
        result = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        print(f"Epoch{epoch}, seed{seed},聚类结果（标签: 样本数):{result}")

        # 4. 使用 PCA 将数据降维到 2 维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_normalized)

        # 构造 DataFrame 用于后续绘图
        df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels.astype(str)
        })

        # 5. 利用 go.Figure() 创建交互式散点图
        fig = go.Figure()

        # 对于每个聚类（包括噪声 -1），添加一个散点图 trace
        for cluster in sorted(df['Cluster'].unique(), key=lambda x: int(x) if x != '-1' else -1):
            cluster_df = df[df['Cluster'] == cluster]
            fig.add_trace(go.Scatter(
                x = cluster_df['PC1'],
                y = cluster_df['PC2'],
                mode = 'markers',
                name = f'Cluster {cluster}',
                marker = dict(
                    size = 5,
                    line =dict(width=0.5)
                ),
                text = cluster_df['Cluster']
            ))

        # 设置图形布局
        fig.update_layout(
            title = "DBSCAN 聚类结果可视化 -epoch {}, {}".format(epoch, result),
            # xaxis_title = "主成分 1",
            # yaxis_title = "主成分 2",
            legend_title = "Cluster ID"
        )

        # 6. 将图形保存为 HTML 文件
        path = f'./DBSCAN_cluster_new/seed{seed}/'
        if not os.path.exists(path):
            os.makedirs(path)
        fig.write_html(path + f"clusters_{epoch}_seed_{seed}.html")
        print("可视化结果已保存为 clusters.html")