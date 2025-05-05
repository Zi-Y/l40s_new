import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random, os
from types import SimpleNamespace
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.colors
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# =================================
# 1. 设置随机种子
# =================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =================================
# 2. 定义一些可能会用到的函数
# =================================

def check_tensor_values_in_set(value_set, tensor):
    """
    调试辅助函数：检查 batch_ID 是否包含了已移除的 ID。
    如果不需要，可自行删除或注释。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入的 tensor 必须是 torch.Tensor 类型")
    if tensor.dim() != 1:
        raise ValueError("输入的 tensor 必须是一维张量")

    tensor_values = set(tensor.tolist())
    common_values = tensor_values.intersection(value_set)

    if common_values:
        print('removed ID in training set')
        raise ValueError(f"张量中的以下值存在于集合中: {common_values}")


def prune_dataset(dataset, pruning_rate, sorted_metrics):
    """
    使用 sorted_metrics 对 dataset 进行裁剪。
    dataset: TensorDataset(...) or 兼容的结构, 其每条包含 (X, y, ID, w?)
    sorted_metrics: (N,2) => (sample_id, metric_val)，按 metric_val 升序排序。
    pruning_rate>0 => 移除最容易的部分； <0 => 移除最难的部分
    """
    # dataset[i] => (X_i, y_i, ID_i, w_i) (若只有3列，则 w_i 不存在)
    # ID 在第3个位置(索引2)

    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Dataset is empty. Cannot prune from an empty dataset.")

    # 收集所有 ID
    all_ids = []
    for i in range(n_samples):
        sample_id = dataset[i][2].item()
        all_ids.append(sample_id)

    sorted_sample_ids = sorted_metrics[:, 0]  # (N,)
    # 计算要移除多少个
    remove_count = int(n_samples * abs(pruning_rate))

    if pruning_rate >= 0:
        # 移除最“容易”的前 remove_count 个
        removed_ids = sorted_sample_ids[:remove_count]
    else:
        # 移除最“困难”的后 remove_count 个
        removed_ids = sorted_sample_ids[-remove_count:]

    removed_ids_set = set(map(int, removed_ids))

    remaining_indices = []
    for i in range(n_samples):
        if all_ids[i] not in removed_ids_set:
            remaining_indices.append(i)

    if len(remaining_indices) == 0:
        raise ValueError("After pruning, there are no samples left. Check pruning_rate or sorted_metrics.")

    # 构建新的 TensorDataset
    # dataset.tensors => 可能是 (X_all, Y_all, ID_all, W_all) 或 (X_all, Y_all, ID_all)
    # 先做一次安全解包
    old_tensors = dataset.tensors
    ncol = len(old_tensors)

    # 根据 remaining_indices 进行索引切片
    new_tensors_list = []
    for col_i in range(ncol):
        new_tensors_list.append(old_tensors[col_i][remaining_indices])

    # 生成新的 TensorDataset
    pruned_dataset = TensorDataset(*new_tensors_list)
    return pruned_dataset, removed_ids_set


def prune_dataset_old(dataset, pruning_rate, sorted_metrics):
    """
    旧版示例函数，保留以供参考。如果不需要也可以删除。
    """
    random_pruning = False
    if random_pruning:
        n_samples = len(dataset)
        num_to_keep = int(n_samples * (1 - pruning_rate))
        if num_to_keep <= 0:
            raise ValueError("After pruning, there are 0 samples left.")
        indices = torch.randperm(n_samples)[:num_to_keep]
        old_tensors = dataset.tensors
        new_tensors_list = []
        for col_i in range(len(old_tensors)):
            new_tensors_list.append(old_tensors[col_i][indices])
        return TensorDataset(*new_tensors_list)

    else:
        # 原来的 old 方法, 仅作演示
        n_samples = len(dataset)
        all_ids = []
        for i in range(n_samples):
            all_ids.append(int(dataset[i][2].item()))

        sample_ids = sorted_metrics[:, 0]
        remove_count = int(n_samples * abs(pruning_rate))
        if pruning_rate >= 0.0:
            removed_ids = sample_ids[:remove_count]
        else:
            removed_ids = sample_ids[-remove_count:]

        removed_ids_set = set(map(int, removed_ids))

        remaining_indices = []
        for i, sid in enumerate(all_ids):
            if sid not in removed_ids_set:
                remaining_indices.append(i)

        if len(remaining_indices) == 0:
            raise ValueError("After pruning, there are no samples left (old).")

        old_tensors = dataset.tensors
        new_tensors_list = []
        for col_i in range(len(old_tensors)):
            new_tensors_list.append(old_tensors[col_i][remaining_indices])
        return TensorDataset(*new_tensors_list)


# =================================
# 3. 定义模型
# =================================
class BostonHousingModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(BostonHousingModel, self).__init__()
        layers = []
        in_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =================================
# 4. 测试集评估函数
# =================================
def evaluate_model(model, loader, criterion, device):
    """
    对 loader 中的数据做普通 MSE，不加样本权重。
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y, batch_ID, batch_w in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)  # 标准 MSE
            total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset)


# =================================
# 5. 单次训练过程 (加权)
# =================================
def train_model(model,
                train_loader,
                test_loader,
                optimizer,
                criterion,
                device,
                iterations,
                seed=None,
                removed_ids_set=None):
    """
    对 model 进行 iterations 次训练。用逐样本加权: (MSE * batch_w).mean().
    每隔10 iter在 test_loader 上测一次 loss。
    """
    if seed is not None:
        set_seed(seed)

    model.to(device)
    iteration = 0
    test_losses = []

    mse_no_reduce = nn.MSELoss(reduction='none')

    while iteration < iterations:
        model.train()
        for batch_X, batch_y, batch_ID, batch_w in train_loader:
            # 如需严格检查 removed_ids_set，可启用:
            # check_tensor_values_in_set(removed_ids_set, batch_ID)

            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss_per_sample = mse_no_reduce(outputs, batch_y)
            loss = (loss_per_sample * batch_w).mean()
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 10 == 0:
                test_loss = evaluate_model(model, test_loader, criterion, device)
                test_losses.append(test_loss)

            if iteration >= iterations:
                break

    return test_losses


# =================================
# 6. 计算 metrics 并(可选)写入 Dataset 的第四列 weight
# =================================
def call_pruning_metrics(base_train_dataset, test_loader, config, device):
    """
    计算某种度量(根据 pruning_method)，得到 sorted_metrics:
        - 0: avg loss
        - 1: S2L
        - 2: TTDS

    当 pr=0或2：sorted_metrics = (N,2) => (sample_id, metric_val)，再做Min-Max归一后写入第4列 weight。
    当 pr=1 (S2L)：原始逻辑里仅返回一个 dict { p: removed_indices }，并不会给所有样本写分数。

    返回 (best_params, sorted_metrics)
      - best_params: 人工写死，也可改成你自己的搜索结果
      - sorted_metrics: (N,2) 或 dict (在S2L场景)
    """
    print("Starting hyperparameter optimization...")

    best_params = {
        'learning_rate': 0.01,
        'batch_size': 1024,
        'dropout_rate': 0.1
    }
    lr = best_params['learning_rate']
    bs = best_params['batch_size']
    dr = best_params['dropout_rate']

    set_seed(config.metric_seed)
    print(f'set seed for metric calculation: {config.metric_seed}')

    # =====【【 新增分支开始 】】=====
    # 新增分支：如果 pruning_method 为 -1，则直接跳过模型训练，
    # 随机生成 rank（排序顺序），但最终将每个样本的权重均设置为 1.


    # base_train_dataset => (X, y, ID) 或 (X, y, ID, w)
    old_tensors = base_train_dataset.tensors
    ncol = len(old_tensors)

    # 无论几列，前3列必是 X, Y, ID
    X_all, Y_all, ID_all = old_tensors[0], old_tensors[1], old_tensors[2]

    # 构造 DataLoader，用来跑 metric (采用 MSELoss(reduction='none'))
    # 如果已经有第4列 weight，也不会影响此过程(只是不使用)
    if ncol == 4:
        # dataset => (X, y, ID, w)
        dataset_for_metric = TensorDataset(X_all, Y_all, ID_all)  # 暂时不把w传给该 DataLoader
    else:
        # dataset => (X, y, ID)
        dataset_for_metric = base_train_dataset

    train_loader = DataLoader(dataset_for_metric, batch_size=bs, shuffle=True, drop_last=False)

    # 构造模型
    model = BostonHousingModel(config.input_dim, config.hidden_layers, dr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_metric_cal = nn.MSELoss(reduction='none')
    iterations = config.iterations

    model.to(device)
    iteration = 0
    if config.metric_epoch <= 0:
        num_epochs = int(math.ceil(iterations / len(train_loader)))
    else:
        num_epochs = config.metric_epoch

    dataset_size = len(dataset_for_metric)
    loss_all_epoch = np.zeros((dataset_size, num_epochs + 1))

    sample_id_list = []
    # 第0列存 ID
    for i in range(dataset_size):
        sid_i = ID_all[i].item()
        sample_id_list.append(sid_i)
        loss_all_epoch[i, 0] = sid_i

    epoch = 0
    while iteration < iterations:
        model.train()
        for batch_data in train_loader:
            if len(batch_data) == 3:
                batch_X, batch_y, batch_ID = batch_data
            else:
                # 理论上不会进入该分支(因为 dataset_for_metric 只有3列)
                raise ValueError("batch_data shape mismatch in call_pruning_metrics")

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion_metric_cal(outputs, batch_y)  # shape=(batch_size,)
            loss.mean().backward()
            optimizer.step()

            iteration += 1

            # 记录损失
            loss_cpu = loss.detach().cpu().numpy().squeeze()
            batch_ID_np = batch_ID.numpy()

            for i_id, sid_val in enumerate(batch_ID_np):
                row_idx = np.where(loss_all_epoch[:, 0] == sid_val)[0][0]
                loss_all_epoch[row_idx, epoch + 1] = float(loss_cpu[i_id])

            if iteration >= iterations:
                break
        epoch += 1
        if epoch >= config.metric_epoch:
            break

    # 过滤掉可能是全0行(一般是无用行)
    mask = np.any(loss_all_epoch[:, 1:] != 0, axis=1)
    loss_all_epoch = loss_all_epoch[mask]

    sorted_metrics = None
    if config.pruning_method >= 100 and config.pruning_method < 200:
        loss_all_epoch = np.load('/home/zi/research_project/TDDS/Cifar100_loss_np_epoch30.npy')
    elif config.pruning_method >= 200 and config.pruning_method < 300:
        loss_all_epoch = np.load('/home/local/zi/research_project/quality_air/Era5_4xV100_ag_1_graphcast_no_GC_seed3_loss_all_epochs.npy')
        loss_all_epoch = np.column_stack((np.arange(loss_all_epoch.shape[0]), loss_all_epoch))

    elif config.pruning_method >= 300 and config.pruning_method < 400:
        # CaliforniaHousing
        loss_all_epoch = loss_all_epoch

    elif config.pruning_method >= 400 and config.pruning_method < 500:
        # loss_all_epoch_raw size: (sample_id, number of samples, total epochs)
        loss_all_epoch_raw = np.load("/home/local/zi/research_project/"
                         "iTransformer/checkpoints_early_stopping_epoch_based/"
                         "0_0_ECL_96_96_iTransformer_custom_M_"
                         "ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_"
                         "df1_fctimeF_ebTrue_dt'Exp'_projection_0/"
                         "loss_all_epoch_seed_0.npy")
        # 删除所有全 0 列
        loss_all_epoch = loss_all_epoch_raw[:, ~(loss_all_epoch_raw == 0).all(axis=0)]
        # 过滤掉可能是全0行(一般是无用行)
        mask = np.any(loss_all_epoch[:, 1:] != 0, axis=1)
        loss_all_epoch = loss_all_epoch[mask]
        loss_all_epoch = np.column_stack((np.arange(loss_all_epoch.shape[0]), loss_all_epoch))

    elif config.pruning_method >= 500 and config.pruning_method < 600:
        # weather
        loss_all_epoch_raw = np.load("/home/local/zi/research_project/iTransformer/"
                                     "all_saved_seed0_trained_per_token_loss/"
                                     "all_saved_seed0_pm0_pr0_low10_high10_start0_int20_"
                                     "test1_iTransformer_custom_ftM_sl96_ll48_pl96_"
                                     "dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_"
                                     "exp_projection_0/loss_all_epoch_all_variable_all_stamp0.npy")
        loss_all_epoch = loss_all_epoch_raw.mean(axis=(-2, -1))
        loss_all_epoch = np.column_stack((np.arange(loss_all_epoch.shape[0]), loss_all_epoch))

    elif config.pruning_method >= 600 and config.pruning_method < 700:
        # weather_all_token
        loss_all_epoch_raw = np.load("/home/local/zi/research_project/iTransformer/"
                                     "all_saved_seed0_trained_per_token_loss/"
                                     "all_saved_seed0_pm0_pr0_low10_high10_start0_int20_"
                                     "test1_iTransformer_custom_ftM_sl96_ll48_pl96_"
                                     "dm512_nh8_el3_dl1_df512_fc1_ebtimeF_dtTrue_"
                                     "exp_projection_0/loss_all_epoch_all_variable_all_stamp0.npy")
        loss_all_epoch = loss_all_epoch_raw.transpose(0, 2, 3, 1).reshape(-1, 11)
        loss_all_epoch = np.column_stack((np.arange(loss_all_epoch.shape[0]), loss_all_epoch))





    #     Era5_4xV100_ag_1_graphcast_no_GC_seed3_loss_all_epochs.npy

    if ((config.pruning_method % 100 == 0)
            or (config.pruning_method % 100 == 3)):
        # avg loss
        avg_loss_matrix = loss_all_epoch[:, :2].copy()  # shape=(?,2)
        avg_loss_matrix[:, 1] = loss_all_epoch[:, 1:].mean(axis=1)
        sorted_metrics = avg_loss_matrix[np.argsort(avg_loss_matrix[:, 1])]
        # if config.pruning_method == 3:
        #     sorted_metrics[:, 1] = 1.0

    elif config.pruning_method % 100 == 1:
        # S2L => KMeans
        n_clusters = 100
        from sklearn.cluster import KMeans
        data = loss_all_epoch[:, 1:]  # 只要 loss
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init=10, max_iter=30, random_state=config.metric_seed)
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
        sorted_metrics = {}
        sorted_metrics['cluster'] = cluster_indices_dict
        for pr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # pr = config.pruning_rate
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
            print(f'S2L --- pr:{pr}, removed_IDs rate: {len(removed_IDs)/n_all}')
            sorted_sample_ids = []
            sorted_sample_ids.extend(removed_IDs)
            sorted_sample_ids.extend(S)
            print(f'S2L --- sorted_sample_ids: {len(sorted_sample_ids)}, all: {len(sample_id_list)}')

            # sorted_metrics = np.column_stack((sorted_sample_ids, np.ones(len(sorted_sample_ids))))
            sorted_metrics[pr] = np.column_stack((sorted_sample_ids, np.ones(len(sorted_sample_ids))))





    elif (config.pruning_method % 100 == 2) or (config.pruning_method % 100 == 4):
        # TTDS
        num_samples = loss_all_epoch.shape[0]
        num_epochs = loss_all_epoch.shape[1] - 1
        loss_differences = np.zeros((num_samples, num_epochs + 1))
        loss_differences[:, 0] = loss_all_epoch[:, 0]

        for ep_i in range(1, num_epochs + 1):
            if ep_i == 1:
                continue
            diff = loss_all_epoch[:, ep_i] - loss_all_epoch[:, ep_i - 1]
            loss_differences[:, ep_i] = diff

        sample_ids = loss_differences[:, 0].astype(int)
        # 跳过 0列(ID), 1列(第一epoch?), 取2列开始
        loss_values = np.abs(loss_differences[:, 2:])
        g_mean = np.mean(loss_values, axis=1, keepdims=True)
        R_xn = np.sum((loss_values - g_mean) ** 2, axis=1)
        sorted_indices = np.argsort(R_xn)
        sorted_sample_ids = sample_ids[sorted_indices]
        sorted_Rxn = R_xn[sorted_indices]
        # if config.pruning_method % 100 == 4:
        #     sorted_Rxn[:] = 1.0
        sorted_metrics = np.column_stack((sorted_sample_ids, sorted_Rxn))

    # 如果得到 (N,2) => 写入 dataset 第4列 weight
    if (sorted_metrics is not None
            and isinstance(sorted_metrics, np.ndarray)
            and sorted_metrics.shape[1] == 2):
        values = sorted_metrics[:, 1]
        mn, mx = values.min(), values.max()
        if mx - mn < 1e-12:
            normalized = np.ones_like(values, dtype=np.float32)
        else:
            normalized = (values - mn) / (mx - mn)

        # 构造 sid -> weight
        weight_map = {}
        for i, sid in enumerate(sorted_metrics[:, 0]):
            weight_map[int(sid)] = float(normalized[i])

        # 给 base_train_dataset 加第4列 weight
        # 先把现有的 3 或 4 列保存
        old_tensors = base_train_dataset.tensors
        ncol = len(old_tensors)

        X_all, Y_all, ID_all = old_tensors[0], old_tensors[1], old_tensors[2]

        # 生成新的 W
        new_weights_list = []
        for i in range(len(X_all)):
            sid_i = int(ID_all[i].item())
            w_i = weight_map.get(sid_i, 0.0)
            new_weights_list.append(w_i)
        W_all_new = torch.tensor(new_weights_list, dtype=torch.float32)



        base_train_dataset.tensors = (X_all, Y_all, ID_all, W_all_new)

    print(f"Parameters: {best_params}")
    return best_params, sorted_metrics, loss_all_epoch


# =================================
# 7. 多次运行并记录平均损失
# =================================
def run_multiple_seeds(base_train_dataset,
                       test_loader,
                       best_params,
                       config,
                       device,
                       seeds=10,
                       sorted_metrics=None):
    """
    对于多个 seed：
      1) 根据 sorted_metrics 裁剪训练集
      2) 训练
      3) 记录测试集loss
    """
    all_test_losses = []

    # for seed_i in range(seeds):
    for seed_i in range(seeds):
        set_seed(config.training_seed)

        # pruning_method=0或2 => sorted_metrics是 (N,2)，可直接 prune
        pruned_dataset, removed_ids_set = prune_dataset(
            base_train_dataset,
            config.pruning_rate,
            sorted_metrics
        )


        train_loader = DataLoader(pruned_dataset,
                                  batch_size=best_params['batch_size'],
                                  shuffle=True,
                                  drop_last=(len(pruned_dataset) > best_params['batch_size']))

        cal_removed_ratio = (len(base_train_dataset) - len(pruned_dataset)) / len(base_train_dataset)
        print(f"Seed={seed_i}, pruning_rate={config.pruning_rate}, "
              f"TrainSize={len(base_train_dataset)} -> {len(pruned_dataset)}, removed={cal_removed_ratio:.2f}")

        model = BostonHousingModel(config.input_dim,
                                   config.hidden_layers,
                                   best_params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        test_losses = train_model(model,
                                  train_loader,
                                  test_loader,
                                  optimizer,
                                  criterion,
                                  device=device,
                                  iterations=config.iterations,
                                  seed=None,
                                  removed_ids_set=removed_ids_set)
        all_test_losses.append(test_losses)
    if seeds <= 1:
        for i, val in enumerate(all_test_losses[0], start=1):
            wandb.log({"Iteration": i * 10, "Avg Test Loss": val})
        results_saved_path = (f'./results_air_data_pruning_singlePR_new/drop_new/'
                              f'pm_{config.pruning_method}/'
                              f'pr_{int(100 * config.pruning_rate)}/')
        if not os.path.exists(results_saved_path):
            os.makedirs(results_saved_path)
        np.save(results_saved_path + f"results_seed{config.training_seed}_"
                                     f"pr{config.pruning_rate}_pm{config.pruning_method}"
                                     f"_metric_epoch{config.metric_epoch}_metric_seed{config.metric_seed}.npy",
                all_test_losses[0])

    else:
        # 取平均
        avg_test_losses = np.mean(all_test_losses, axis=0)
        for i, val in enumerate(avg_test_losses, start=1):
            wandb.log({"Iteration": i * 10, "Avg Test Loss": val})


# =================================
# 8. 主流程 (main 函数)
# =================================
def main():
    parser = argparse.ArgumentParser(description="California Housing with optional pruning")
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--pruning_rate', type=float, default=0.1, help='Dataset pruning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 32], help='Sizes of hidden layers')
    parser.add_argument('--wandb_project', type=str, default='california_housing', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='only_for_test', help='WandB run name')
    parser.add_argument('--seeds', type=int, default=1, help='Number of random seeds for multiple runs')
    parser.add_argument('--metric_seed', type=int, default=0, help='random seed for metric calculation')
    parser.add_argument('--training_seed', type=int, default=0, help='random seed for training')
    parser.add_argument('--pruning_method', type=int, default=0, help='0-avg loss, 1-S2L, 2-TTDS')
    parser.add_argument('--metric_epoch', type=int, default=30, help='calculate metrics in the first X epoch')
    args = parser.parse_args()

    config = SimpleNamespace(
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        pruning_rate=args.pruning_rate,
        hidden_layers=args.hidden_layers,
        seeds=args.seeds,
        training_seed=args.training_seed,
        pruning_method=args.pruning_method,
        metric_seed=args.metric_seed,
        metric_epoch=args.metric_epoch,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )

    pm_names = {
        -1: "Random-pruning",
        1: "S2L",
        0: "Avg-Loss",
        2: "TTDS",
        3: "AvgLoss",
        4: "TTDS",

        101: "S2L",
        100: "Avg-Loss",
        102: "TTDS",
        103: "AvgLoss",
        104: "TTDS",

        201: "S2L",
        200: "Avg-Loss",
        202: "TTDS",
        203: "AvgLoss",
        204: "TTDS",

        301: "S2L",
        300: "Avg-Loss",
        302: "TTDS",
        303: "AvgLoss",
        304: "TTDS",

        401: "S2L",
        400: "Avg-Loss",
        402: "TTDS",
        403: "AvgLoss",
        404: "TTDS",

        501: "S2L",
        500: "Avg-Loss",
        502: "TTDS",
        503: "AvgLoss",
        504: "TTDS",

        601: "S2L",
        600: "Avg-Loss",
        602: "TTDS",
        603: "AvgLoss",
        604: "TTDS",

        701: "S2L",
        700: "Avg-Loss",
        702: "TTDS",
        703: "AvgLoss",
        704: "TTDS",
    }
    # [-1, 1, 3, 4, 101, 103, 104]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    if config.pruning_method < 300:
        # ============== 1) 加载数据并手动切分 ==============
        print("Loading dataset ...")
        data_file = "AirQualityUCI.csv"

        # Step 3: 数据预处理
        data_all = pd.read_csv(data_file, sep=';', decimal=',', na_values=-200)
        # data = data.dropna()
        features = ['CO(GT)', 'T', 'RH', 'AH',
                    'PT08.S1(CO)', 'PT08.S2(NMHC)',
                    'PT08.S3(NOx)', 'PT08.S4(NO2)',
                    'PT08.S5(O3)']
        data = data_all[features].dropna()

        X_raw = data[['T', 'RH', 'AH',
                      'PT08.S1(CO)', 'PT08.S2(NMHC)',
                      'PT08.S3(NOx)', 'PT08.S4(NO2)',
                      'PT08.S5(O3)']].values
        y_raw = data['CO(GT)'].values
        # features_new = ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        #             'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
        #             'PT08.S5(O3)', 'T', 'RH', 'AH',]
        # features_new = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        #             'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
        #             'PT08.S5(O3)', 'T', 'RH', 'AH',]
        # data_all[features_new].dropna()
        # 7344 'NMHC(GT)' 很多NAN
    else:
    # elif (config.pruning_method >= 300) and (config.pruning_method < 400):
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X_raw, y_raw = data.data, data.target

    # 标准化
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_raw)
    y_scaled = y_raw.reshape(-1, 1)
    y_scaled = Y_scaler.fit_transform(y_scaled)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    IDs = torch.arange(len(X_tensor), dtype=torch.int32)

    # 手动shuffle & split (8:2)
    num_samples = len(X_tensor)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_size = int(0.8 * num_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    # 构造训练集 (3列: X, y, ID)
    X_train = X_tensor[train_idx]
    y_train = y_tensor[train_idx]
    ID_train = IDs[train_idx]
    base_train_dataset = TensorDataset(X_train, y_train, ID_train)

    # 构造测试集 (4列: X, y, ID, weight=1.0 便于 evaluate_model 输入)
    X_test = X_tensor[test_idx]
    y_test = y_tensor[test_idx]
    ID_test = IDs[test_idx]
    W_test = torch.ones(len(test_idx), dtype=torch.float32)  # 测试集 weight=1
    test_dataset = TensorDataset(X_test, y_test, ID_test, W_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    config.input_dim = X_train.shape[1]

    # ============== 2) 先计算 metrics (并写入 base_train_dataset 第4列 weight，如 pr=0/2) ==============
    best_params, sorted_metrics, loss_all_epoch = call_pruning_metrics(base_train_dataset, test_loader, config, device=device)
    if config.pruning_method < 100:
        dataset_name = 'AirQualityUCI'
    elif config.pruning_method < 200:
        dataset_name = 'Cifar100'
    elif config.pruning_method < 300:
        dataset_name = 'Era5'
    elif config.pruning_method < 400:
        dataset_name = 'CaliforniaHousing'
    elif config.pruning_method < 500:
        dataset_name = 'ECL'
    elif config.pruning_method < 600:
        dataset_name = 'weather'
    elif config.pruning_method < 700:
        dataset_name = 'weather_all_token'
    elif config.pruning_method < 800:
        dataset_name = 'weather_unique_token'

    plot_feature_distribution = False
    if plot_feature_distribution:
        if config.pruning_method < 100:
            # ==============  plot data distribution ==============
            # Load the data
            file_path = "AirQualityUCI.csv"
            data_all = pd.read_csv(file_path, sep=';', decimal=',', na_values=-200)

            # Define the features to be analyzed
            features_new = ['T', 'RH', 'AH',
                            'PT08.S2(NMHC)',
                            'PT08.S3(NOx)', 'PT08.S4(NO2)',
                            'PT08.S5(O3)', 'PT08.S1(CO)',
                            'CO(GT)']

            # Select the features and drop NaN values
            data_pd = data_all[features_new].dropna()

            # Define colors based on academic journal conventions
            input_color = "#1f77b4"  # Professional blue
            sensor_color = "#ff7f0e"  # Professional orange
            pruning_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


            data = data_pd.iloc[train_idx]
            ID_train_np = IDs[train_idx].numpy()
            n_samples = len(ID_train_np)
            # 收集所有 ID
            all_ids = ID_train_np

            # Create the figure
            fig, axes = plt.subplots(len(pruning_rates), len(features_new), figsize=(15, 12), sharex=False,
                                     sharey=False)
            # sharey=True)
            fig.suptitle(f"Feature Distributions with Different Pruning Rates using pruning method {pm_names[config.pruning_method]}", fontsize=16)

            # Add column labels for input and target
            for i, feature in enumerate(features_new[:-1]):
                axes[0, i].set_title(f"{feature}\n(Input)", fontsize=12)
            axes[0, -1].set_title(f"{features_new[-1]}\n(Target)", fontsize=12)

            # Plot distributions for different pruning rates
            for j, rate in enumerate(pruning_rates):
                if config.pruning_method == -1:
                    reduced_data = data.sample(frac=(1 - rate), random_state=42) if rate > 0 else data
                    num_samples_reduced = reduced_data.shape[0]
                else:
                    if (config.pruning_method % 100 == 3 or config.pruning_method % 100 == 4):
                        sorted_sample_ids = sorted_metrics[:, 0]  # (N,)
                    elif config.pruning_method % 100 == 1:
                        sorted_sample_ids = sorted_metrics[rate][:, 0]
                    # 计算要移除多少个
                    remove_count = int(n_samples * abs(rate))

                    if rate >= 0:
                        # 移除最“容易”的前 remove_count 个
                        removed_ids = sorted_sample_ids[:remove_count]
                    else:
                        # 移除最“困难”的后 remove_count 个
                        removed_ids = sorted_sample_ids[-remove_count:]

                    removed_ids_set = set(map(int, removed_ids))

                    remaining_indices = []
                    for i in range(n_samples):
                        if all_ids[i] not in removed_ids_set:
                            remaining_indices.append(i)
                    reduced_data = data.iloc[remaining_indices]


                # Add pruning rate labels to the left-most column without box
                axes[j, 0].annotate(f"Pruning Rate {int(rate * 100)}%\n", xy=(-0.5, 0.5), xycoords='axes fraction',
                                    fontsize=12, ha='right', va='center', rotation=90)

                for i, feature in enumerate(features_new):
                    axes[j, i].hist(reduced_data[feature].dropna(), bins=50, alpha=0.75,
                                    color=input_color if feature != features_new[-1] else sensor_color)
                    axes[j, i].tick_params(axis='x', rotation=45)
                    if j == (len(pruning_rates) - 1):
                        axes[j, i].set_xlabel(feature)
                    if i == 0:
                        axes[j, i].set_ylabel("Frequency")

            plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
            file_path = f'./plot_results/{pm_names[config.pruning_method]}/1_{dataset_name}_{pm_names[config.pruning_method]}_data_distribution_pruning.png'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close('all')

    plot_metric_distribution = True
    if plot_metric_distribution:
        if (config.pruning_method % 100 == 3
                or config.pruning_method % 100 == 4):
            metric_values = sorted_metrics[:, 1]  # (N,)

            # 计算1%和99%分位数
            lower_bound = np.percentile(metric_values, 1.0)
            upper_bound = np.percentile(metric_values, 99.0)

            # 过滤数据，去除前5%和后5%的值
            metric_values = metric_values[(metric_values >= lower_bound) & (metric_values <= upper_bound)]

            plt.figure(figsize=(8, 6))
            plt.hist(metric_values, bins=100, color='gray', edgecolor='gray', alpha=0.8)
            plt.xlabel('Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(
                f'Histogram of Importance Scores using method {pm_names[config.pruning_method]} on {dataset_name}',
                fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.6, color='black')

            # 适合论文风格的去除边框
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            file_path = f'./plot_results/{pm_names[config.pruning_method]}/2_{dataset_name}_{pm_names[config.pruning_method]}_metric_distribution.png'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)
            plt.close('all')

        elif config.pruning_method % 100 == 1:
            dict_cluster = sorted_metrics['cluster']
            metric_values = [len(values) for values in dict_cluster.values()]


            # 获取键和值的数量
            keys = np.arange(len(dict_cluster))
            values_count = metric_values[::-1]

            # 画条形图（柱状图）
            plt.figure(figsize=(8, 6))
            plt.bar(keys, values_count, color="gray", width=1.0)  # 论文标准一般使用黑色或灰色
            # 添加标签
            plt.xlabel(f"Cluster index (sorted based on cluster size)", fontsize=12)
            plt.ylabel("Size of cluster", fontsize=12)
            plt.title(f"Distribution of cluster sizes using method {pm_names[config.pruning_method]} on {dataset_name}", fontsize=14)

            # 显示网格线（论文风格通常带网格线）
            plt.grid(axis='y', linestyle='--', alpha=0.6, color='black')

            # 显示图表
            file_path = f'./plot_results/{pm_names[config.pruning_method]}/2_{dataset_name}_{pm_names[config.pruning_method]}_metric_distribution.png'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path)


            plt.close('all')


    plot_loss_curve = True
    if plot_loss_curve:
        # 选择Plotly的默认颜色序列
        colors = plotly.colors.qualitative.Set1

        if (config.pruning_method % 100 == 3
                or config.pruning_method % 100 == 4):
            metric_values = sorted_metrics[:, 1]  # (N,)
            ranked_ids = sorted_metrics[:, 0]
        elif config.pruning_method % 100 == 1:
            dict_cluster = sorted_metrics['cluster']

            # metric_values = sorted_metrics[0.0][:, 1]
            ranked_ids = sorted_metrics[0.0][:, 0]

            sample_to_cluster_size = {}

            for cluster_id, samples in dict_cluster.items():
                cluster_size = len(samples)
                for sample in samples:
                    sample_to_cluster_size[sample] = cluster_size

            # 根据sampleID_list返回对应的cluster大小
            metric_values = [sample_to_cluster_size.get(sample_id, 0) for sample_id in ranked_ids]


        loss_curves_list = []
        scores_list = []

        # ranked_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,]
        # ranked_percent = [0.2, 0.4, 0.6, 0.8,]
        ranked_percent = [0.01, 0.5, 0.99,]

        for selected_percent in ranked_percent:
            selected_index = int(selected_percent*len(metric_values))
            seleced_ID = ranked_ids[selected_index]
            # 查找 sample_ID 所在的行
            row = loss_all_epoch[loss_all_epoch[:, 0] == seleced_ID]
            loss_curves_list.append(row[0, 1:])
            scores_list.append(metric_values[selected_index])

        x =np.arange(1, len(loss_curves_list[0])+1)
        fig = go.Figure()

        for i in range(len(scores_list)):
            fig.add_trace(go.Scatter(x=x, y=loss_curves_list[i],
                                     mode='lines',
                                     name=f'The sample ranked in the top {100.0*ranked_percent[i]}%,'
                                          f' score {scores_list[i]:.4f}',
                                     line=dict(width=3,
                                               color=colors[i % len(colors)], dash='solid')))

        # fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)', line=dict(color='gray', dash='dash')))
        # fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='sin(x) * cos(x)', line=dict(color='darkgray', dash='dot')))

        # 设置布局，符合学术风格
        fig.update_layout(
            # title=f'Loss curves of ranked samples based on metric {pm_names[config.pruning_method]}',
            title=dict(
                text=f'Loss curves of ranked samples using metric {pm_names[config.pruning_method]} on {dataset_name}',
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,  # Center title
                y=0.95
            ),
            xaxis_title="Epoch",
            yaxis_title="loss",
            legend_title="Samples of different rankings",
            template="plotly_white",
            font=dict(size=14, family="Arial, sans-serif"),
            xaxis=dict(showgrid=True, zeroline=False, tickmode='linear', dtick=20, title_font=dict(size=16)),
            yaxis=dict(showgrid=True, zeroline=False, title_font=dict(size=16)),
            hovermode="closest",  # Remove vertical hover line
            margin=dict(l=60, r=60, t=60, b=60),
            plot_bgcolor="white",
            hoverlabel=dict(
                # font_size=14,  # 调整字体大小，避免被裁剪
                namelength=-1  # -1 表示不裁剪名称
            )
        )


        # 保存为HTML文件
        html_file = f'./plot_results/{pm_names[config.pruning_method]}/3_{dataset_name}_{pm_names[config.pruning_method]}_loss_curves.html'
        os.makedirs(os.path.dirname(html_file), exist_ok=True)
        pio.write_html(fig, file=html_file, full_html=True)





        # # 绘制折线图，采用学术风格
        # plt.figure(figsize=(8, 6))
        # plt.plot(x, loss_curves_list[0],
        #          label=f'easiest sample, score {scores_list[0]:.4f}',
        #          color='black', linestyle='-', linewidth=1.5)
        #
        # plt.plot(x, loss_curves_list[1],
        #          label=f'infermediate-level sample, score {scores_list[1]:.4f}',
        #          color='darkgray', linestyle='-.', linewidth=1.5)
        #
        # plt.plot(x, loss_curves_list[2],
        #          label=f'hardest sample, score {scores_list[2]:.4f}',
        #          color='gray', linestyle='--', linewidth=1.5)
        #
        # # 轴标签和标题
        # plt.xlabel('X-axis', fontsize=12)
        # plt.ylabel('Y-axis', fontsize=12)
        #
        # plt.title(f'Loss_curves of samples based on metric {pm_names[config.pruning_method]}',
        #           fontsize=14)
        #
        # # 适合论文风格的网格和边框
        # plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        #
        # # 图例
        # plt.legend(fontsize=12, frameon=False)
        # plt.savefig(f'3_{pm_names[config.pruning_method]}_loss_curves.png')
        # plt.close('all')
    #     f'Histogram of Importance Scores using method {pm_names[config.pruning_method]}'









if __name__ == "__main__":
    main()