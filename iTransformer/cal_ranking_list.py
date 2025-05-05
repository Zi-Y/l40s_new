import os, random

import numpy as np

# loss_all_epoch_raw = np.load("/home/local/zi/research_project/"
#                          "iTransformer/checkpoints/"
#                          "0_0_ECL_96_96_iTransformer_custom_M_"
#                          "ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_"
#                          "df1_fctimeF_ebTrue_dt'Exp'_projection_0/"
#                          "loss_all_epoch_seed_0.npy")

loss_all_epoch_raw = np.load("/home/zi/research_project/iTransformer/checkpoints/seed0_pm0_pr0_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0/loss_all_epoch_seed_0.npy")
# 删除所有全 0 列
loss_all_epoch = loss_all_epoch_raw[:, ~(loss_all_epoch_raw == 0).all(axis=0)]

# 过滤掉可能是全0行(一般是无用行)
mask = np.any(loss_all_epoch[:, 1:] != 0, axis=1)
loss_all_epoch = loss_all_epoch[mask]

# 找到 loss_all_epoch[:, 1:] 中为 0 的位置
rows, cols = np.where(loss_all_epoch[:, 1:] == 0)

# 获取对应的第 0 列值
no_appeared_sample_ids = loss_all_epoch[rows, 0]
print(f'removed sample rate: {len(no_appeared_sample_ids) * 100.0 / len(loss_all_epoch[:, 0]):.2f}%')
saved_rank_path = "./rank_list/"
if not os.path.exists(saved_rank_path):
    os.makedirs(saved_rank_path)

for pruning_method in [1, 2, 3]:
    if (pruning_method == 1):
        # avg loss
        avg_loss_matrix = loss_all_epoch[:, :2].copy()  # shape=(?,2)
        avg_loss_matrix[:, 1] = loss_all_epoch[:, 1:].mean(axis=1)

        values = avg_loss_matrix[:, 1]
        mn, mx = values.min(), values.max()
        if mx - mn < 1e-12:
            avg_loss_matrix[:, 1] = np.ones_like(values, dtype=np.float32)
        else:
            avg_loss_matrix[:, 1] = (values - mn) / (mx - mn)

        sorted_metrics = avg_loss_matrix[np.argsort(avg_loss_matrix[:, 1])]
        # np.save(f'{saved_rank_path}/seed0_pm{pruning_method}_pr0_ECL_96_96_iTransformer_'
        #         f'custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_'
        #         f'df512_fc1_ebtimeF_dtTrue_exp_projection_0', sorted_metrics)

        np.save(f'{saved_rank_path}/seed0_pm{pruning_method}_pr0_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0', sorted_metrics)

    #


    elif pruning_method == 2:
        # S2L => KMeans
        n_clusters = 100
        from sklearn.cluster import KMeans

        data = loss_all_epoch[:, 1:]  # 只要 loss
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init=10, max_iter=30, random_state=0)
        kmeans.fit(data)

        # # Get one sample from each cluster
        # unique_labels = np.unique(kmeans.labels_)
        # samples_from_clusters = []

        # for label in unique_labels:
        #     sample = data[kmeans.labels_ == label][0]  # Take the first sample from each cluster
        #     samples_from_clusters.append(sample)

        cluster_counts = np.bincount(kmeans.labels_)
        # Create a dictionary with cluster ranks as keys and indices as values
        cluster_indices_dict = {}
        sorted_clusters = sorted(range(len(cluster_counts)), key=lambda x: cluster_counts[x], reverse=False)

        for rank, cluster_index in enumerate(sorted_clusters, start=1):
            indices_in_cluster = np.where(kmeans.labels_ == cluster_index)[0]
            sample_id_in_cluster = loss_all_epoch[:, 0][indices_in_cluster]
            cluster_indices_dict[rank] = sample_id_in_cluster.tolist()

        sample_id_list = loss_all_epoch[:, 0]
        # pruning_rate = 0.1
        for pr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # pr = pruning_rate
            keep_rate = 1.0 - pr
            B = int(keep_rate * data.shape[0])
            S = []
            K = n_clusters
            for k, cluster_examples in cluster_indices_dict.items():
                cluster_examples = list(map(int, cluster_examples))
                R_k = (B - len(S)) // (K - k + 1)
                if len(cluster_examples) <= R_k:
                    S.extend(cluster_examples)
                else:
                    S.extend(random.sample(cluster_examples, R_k))

            n_all = len(sample_id_list)
            full_set = set(sample_id_list)
            subset = set(S)
            removed_IDs = sorted(list(full_set - subset))
            print(f'S2L --- pr:{pr}, removed_IDs rate: {len(removed_IDs) / n_all}')
            sorted_sample_ids = []
            sorted_sample_ids.extend(removed_IDs)
            sorted_sample_ids.extend(S)
            print(f'S2L --- sorted_sample_ids: {len(sorted_sample_ids)}, all: {len(sample_id_list)}')

            sorted_metrics = np.column_stack((sorted_sample_ids, np.ones(len(sorted_sample_ids))))
            # np.save(f'{saved_rank_path}/seed0_pm{pruning_method}_pr{int(100*pr)}_ECL_96_96_iTransformer_'
            #         f'custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_'
            #         f'df512_fc1_ebtimeF_dtTrue_exp_projection_0', sorted_metrics)

            np.save(f'{saved_rank_path}/seed0_pm{pruning_method}_pr{int(100*pr)}_'
                    f'weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_'
                    f'nh8_el3_dl1_df512_fc1_ebtimeF_'
                    f'dtTrue_exp_projection_0', sorted_metrics)
    #


    elif (pruning_method == 3):
        # TTDS
        num_samples = loss_all_epoch.shape[0]
        num_epochs = loss_all_epoch.shape[1] - 1
        loss_differences = np.zeros((num_samples, num_epochs))
        loss_differences[:, 0] = loss_all_epoch[:, 0]

        for ep_i in range(1, num_epochs + 1):
            if ep_i == 1:
                continue
            diff = loss_all_epoch[:, ep_i] - loss_all_epoch[:, ep_i - 1]
            loss_differences[:, ep_i-1] = diff

        sample_ids = loss_differences[:, 0].astype(int)
        # 跳过 0列(ID), 1列(第一epoch?), 取2列开始
        loss_values = np.abs(loss_differences[:, 2:])
        g_mean = np.mean(loss_values, axis=1, keepdims=True)
        R_xn = np.sum((loss_values - g_mean) ** 2, axis=1)
        sorted_indices = np.argsort(R_xn)
        sorted_sample_ids = sample_ids[sorted_indices]
        sorted_Rxn = R_xn[sorted_indices]
        # if pruning_method == 4:
        #     sorted_Rxn[:] = 1.0

        values = sorted_Rxn
        mn, mx = values.min(), values.max()
        if mx - mn < 1e-12:
            normalized = np.ones_like(values, dtype=np.float32)
        else:
            normalized = (values - mn) / (mx - mn)

        sorted_metrics = np.column_stack((sorted_sample_ids, normalized))
        # np.save(f'{saved_rank_path}/seed0_pm{pruning_method}_pr0_ECL_96_96_iTransformer_'
        #         f'custom_ftM_sl96_ll48_pl96_dm512_nh8_el3_dl1_'
        #         f'df512_fc1_ebtimeF_dtTrue_exp_projection_0', sorted_metrics)
        np.save(
            f'{saved_rank_path}/seed0_pm{pruning_method}_pr0_'
            f'weather_96_96_iTransformer_custom_ftM_sl96_'
            f'll48_pl96_dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_exp_projection_0',
            sorted_metrics)

print(loss_all_epoch.shape)
