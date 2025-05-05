import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np
import random, os
from types import SimpleNamespace
import math
from torchvision import datasets, transforms
import torch.nn.functional as F

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
# 2. 定义裁剪函数
# =================================

def check_tensor_values_in_set(value_set, tensor):
    """
    检查一维张量中的任何值是否存在于集合中。

    :param value_set: set, 待比较的集合
    :param tensor: torch.Tensor, 一维张量
    :raises ValueError: 如果张量中的任何值在集合中，则抛出异常
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入的 tensor 必须是 torch.Tensor 类型")
    if tensor.dim() != 1:
        raise ValueError("输入的 tensor 必须是一维张量")

    # 转换为 Python 集合后与 value_set 取交集，检查是否有共同元素
    tensor_values = set(tensor.tolist())
    common_values = tensor_values.intersection(value_set)

    if common_values:
        print('removed ID in training set')
        raise ValueError(f"张量中的以下值存在于集合中: {common_values}")


def prune_dataset(dataset, pruning_rate, sorted_metrics):
    """
    根据 pruning_rate，使用 sorted_metrics 对样本进行“易/难”排序后，删除相应的样本。
    pruning_rate > 0 时，移除最“容易”的那部分样本；pruning_rate < 0 时，移除最“困难”的那部分样本。

    参数：
        dataset: 可能是 TensorDataset，也可能是 Subset(TensorDataset)。
        pruning_rate > 0:  比例，例如 0.1 表示删除其中 10%的最“容易”样本
        pruning_rate < 0:  比例，例如 -0.1 表示删除其中 10%的最“困难”样本
        sorted_metrics: 二维数组，第一列是全局的 sample_id (或者全局索引)，第二列是对应的度量值，
                       排序方式是从“难”到“易”或从“易”到“难”，视你在外部如何计算

    返回：
        一个新的 Subset（嵌套子集）或直接子集（取决于你的需求），仅包含剩余的样本。
    """

    # -------- 1. 获取当前子集里所有样本对应的【全局索引】和【真正的ID】 --------
    # 尤其当传进来的 dataset 本身是 Subset 时，必须一层层往下找到到底层的 TensorDataset。
    all_global_indices = []
    all_ids = []

    def unwrap_dataset(ds):
        # 递归找到最底层的 TensorDataset
        # 并且收集 “ds” 这一层所使用的 indices（如果是 Subset）。
        # 如果 ds 就是 TensorDataset，直接返回 (ds, None)
        if isinstance(ds, Subset):
            # ds.dataset 可能又是下一层 Subset 或者最底层 TensorDataset
            parent_dataset, parent_indices = unwrap_dataset(ds.dataset)
            # ds.indices 是本层子集的索引，要基于 parent_indices 做二次映射
            if parent_indices is not None:
                # 如果上一层也是 Subset，就需要做一下“索引拼接”
                final_indices = [parent_indices[i] for i in ds.indices]
            else:
                # 如果上一层已经是 TensorDataset，那么本层 ds.indices 就是真实的全局索引
                final_indices = ds.indices
            return parent_dataset, final_indices
        else:
            # ds 不是 Subset，一般就是 TensorDataset，返回本身，并指明“我这里没有 indices”
            return ds, None

    # 拿到最底层数据集 & 本子集对应的所有全局索引
    base_dataset, base_indices = unwrap_dataset(dataset)
    if base_indices is None:
        # 说明 dataset 就是 TensorDataset，没有子集概念
        base_indices = range(len(base_dataset))  # 全部索引

    # 提取 ID
    for idx in base_indices:
        # base_dataset[idx] 一般返回 (X, y, ID)
        sample_id = base_dataset[idx][2].item()
        all_global_indices.append(idx)  # 全局索引
        all_ids.append(sample_id)  # 样本的ID值

    num_samples = len(all_global_indices)
    if num_samples == 0:
        raise ValueError("Dataset is empty. Cannot prune from an empty dataset.")

    # -------- 2. 根据 pruning_rate，从 sorted_metrics 里找出要删除的全局 ID --------
    # sorted_metrics[:, 0] 应该是全局的 sample_id
    sorted_sample_ids = sorted_metrics[:, 0]

    # 这里假设 sorted_metrics 已经按度量从“小到大”排好序
    # 如果 pruning_rate > 0: 移除前 X%的样本（“最容易”的部分）
    # 如果 pruning_rate < 0: 移除后 X%的样本（“最困难”的部分）
    if pruning_rate >= 0.0:
        remove_count = int(num_samples * pruning_rate)
        removed_ids = sorted_sample_ids[:remove_count]  # 取最前面的那部分
    else:
        remove_count = int(num_samples * abs(pruning_rate))
        removed_ids = sorted_sample_ids[-remove_count:]  # 取最后面的那部分

    removed_ids_set = set(map(int, removed_ids))

    # -------- 3. 根据要删除的 ID，筛选保留的 local_index 列表 --------
    remaining_indices = []
    for local_i, global_idx in enumerate(all_global_indices):
        # all_ids[local_i] 就是这个样本真正的 ID
        if all_ids[local_i] not in removed_ids_set:
            remaining_indices.append(local_i)

    if len(remaining_indices) == 0:
        raise ValueError("After pruning, there are no samples left. Please check your pruning_rate or sorted_metrics.")

    # -------- 4. 构造一个新的 Subset，指向同一个最底层 base_dataset，但只保留 remaining_indices 指定的那些条目 --------
    final_global_indices = [all_global_indices[i] for i in remaining_indices]
    # 注意，这里我们返回的是 "base_dataset" 的子集，让它持有 final_global_indices
    # 这样就不会一层层嵌套太多 Subset。但如果你想要在原子集上再嵌套一层 Subset，也可以。
    # len(set(map(int, all_ids)) - set(map(int, sorted_sample_ids)))

    pruned_subset = Subset(base_dataset, final_global_indices)
    return pruned_subset, removed_ids_set

def random_prune_dataset(dataset, pruning_rate):
    """
    从给定 dataset 中随机抽取 (1 - pruning_rate) 比例的样本。
    """
    num_samples = len(dataset)
    num_to_keep = int(num_samples * (1 - pruning_rate))
    if num_to_keep <= 0:
        raise ValueError(
            f"After pruning, there are 0 samples left. "
            f"Please reduce pruning_rate (<1.0) or enlarge the dataset."
        )
    indices = torch.randperm(num_samples)[:num_to_keep]
    return Subset(dataset, indices), None
def prune_dataset_old(dataset, pruning_rate, sorted_metrics):
    """
    从给定 dataset 中随机抽取 (1 - pruning_rate) 比例的样本。
    """
    random_pruning = False
    if random_pruning:
        num_samples = len(dataset)
        num_to_keep = int(num_samples * (1 - pruning_rate))
        if num_to_keep <= 0:
            raise ValueError(
                f"After pruning, there are 0 samples left. "
                f"Please reduce pruning_rate (<1.0) or enlarge the dataset."
            )
        indices = torch.randperm(num_samples)[:num_to_keep]
        return Subset(dataset, indices)
    else:
        """
        从给定的 dataset 中删除指定的样本 ID。

        参数：
            dataset: 原始数据集
            pruning_rate > 0, remove easy samples
            pruning_rate < 0, remove hard samples
            sample_ids: 包含需要删除样本 ID 的 NumPy 数组

        返回：
            一个新的 Subset 数据集，删除了指定样本
        """
        num_samples = len(dataset)

        all_sample_ids_new = []
        for idx_in_original in dataset.indices:
            # 这里 idx_in_original 才是真正的“原始数据集”的索引
            sample_id_i = dataset.dataset[idx_in_original][2].item()
            all_sample_ids_new.append(sample_id_i)


        # 提取所有样本的 ID
        all_sample_ids = [dataset.dataset[i][2].item() for i in range(num_samples)]

        sample_ids = sorted_metrics[:,0]

        if pruning_rate >= 0.0:
            # remove easy samples
            removed_ids = sample_ids[:int(num_samples * pruning_rate)]
        else:
            # remove hard samples
            removed_ids = sample_ids[int(num_samples * pruning_rate):]

        # 转换 sample_ids 为集合以提高查找效率
        removed_ids_set = set(map(int, removed_ids))

        # 获取需要保留的索引
        # remaining_indices = [i for i, sample_id in enumerate(all_sample_ids) if sample_id not in removed_ids_set]

        # 初始化一个空列表用于存储剩余的索引
        remaining_indices = []

        # 遍历 all_sample_ids 中的每个 sample_id，同时获取其索引 i
        for i, sample_id in enumerate(all_sample_ids):
            # 检查 sample_id 是否在 removed_ids_set 中
            is_in_removed_ids = sample_id in removed_ids_set

            # 如果 sample_id 不在 removed_ids_set 中
            if not is_in_removed_ids:
                # 将当前索引 i 添加到 remaining_indices 列表中
                remaining_indices.append(i)

        # set(all_sample_ids) == set(sample_ids)
        # set(map(int, all_sample_ids)) == set(map(int, sample_ids))
        # len(set(map(int, all_sample_ids)) - set(map(int, sample_ids)))
        if len(remaining_indices) == 0:
            raise ValueError("After pruning, there are no samples left. Please check your sample_ids.")

        # 返回新的 Subset 数据集
        return Subset(dataset, remaining_indices)

# =================================
# 3. 定义模型
# =================================
class BostonHousingModel(nn.Module):
    # def __init__(self, input_dim, hidden_layers, dropout_rate):
    #     super(BostonHousingModel, self).__init__()
    #     layers = []
    #     in_dim = input_dim
    #     for units in hidden_layers:
    #         layers.append(nn.Linear(in_dim, units))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout_rate))
    #         in_dim = units
    #     layers.append(nn.Linear(in_dim, 10))
    #     self.model = nn.Sequential(*layers)
    #
    # def forward(self, x):
    #     x = x.view(x.size(0), -1)  # 确保输入形状为 [batch_size, 784]
    #     return self.model(x)
    def __init__(self,input_dim, hidden_layers, dropout_rate):
        super(BostonHousingModel, self).__init__()
        # 第一层卷积：输入通道=1 (MNIST 为灰度图)，输出通道=10，卷积核=5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)

        # 第二层卷积：输入通道=10，输出通道=20，卷积核=5
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        # 全连接层：将卷积结果展平后输入，输出 50 维隐藏层
        # 注意：如果没有使用池化层，输入图像为 28x28，经过两次 5x5 的卷积后，
        #      输出的特征图大小为 20x20，因此第二次卷积后的张量尺寸为:
        #      batch_size x 20(通道) x 20(高) x 20(宽)
        #      展平后为 20 * 20 * 20 = 8000
        self.fc1 = nn.Linear(in_features=20 * 20 * 20, out_features=50)

        # 输出层：10 个分类
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        # 第一层卷积 + ReLU
        x = F.relu(self.conv1(x))
        # 第二层卷积 + ReLU
        x = F.relu(self.conv2(x))

        # 将卷积层输出展平为 (batch_size, 特征数)
        x = x.view(x.size(0), -1)

        # 全连接隐藏层 + ReLU
        x = F.relu(self.fc1(x))
        # 输出层 (logits)
        x = self.fc2(x)

        # 如果使用 CrossEntropyLoss，通常不需要在此处再做 softmax
        return x


# =================================
# 4. 评估函数
# =================================
def evaluate_model(model, loader, criterion, device):
    """
    评估模型在给定 loader 上的平均损失（CE）。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y, sample_ID in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# =================================
# 5. 单次训练过程
# =================================
def train_model(model, train_loader, test_loader, optimizer, criterion, device,
                iterations, seed=None, removed_ids_set=None):
    """
    对 model 进行若干次迭代的训练，并在每个 epoch（完整遍历训练集）结束后在 test_loader 上做评估。
    如果传入了 seed，则在本函数内再次固定随机种子，保证可复现性。
    返回每个 epoch 的测试集 loss 列表。
    """
    if seed is not None:
        set_seed(seed)

    model.to(device)
    test_results = []
    iteration = 0

    while iteration < iterations:
        model.train()
        for batch_X, batch_y, sample_ID in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # check_tensor_values_in_set(removed_ids_set, sample_ID)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            iteration += 1
            # if iteration % len(train_loader) == 0:
            if iteration % 10 == 0:
                test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
                test_results.append((test_loss, test_accuracy))

            if iteration >= iterations:
                break

    return test_results


def call_pruning_metrics(base_train_dataset, test_loader, config, device):
    """
    在给定的 base_train_dataset 上进行超参数网格搜索，每次都会重新裁剪数据集并训练。
    返回在搜索中得到的最佳超参数组合。
    """
    print("Starting call_pruning_metrics...")

    best_params = {'learning_rate': 0.01,
                   'batch_size': 1024,
                   'dropout_rate': 0.1}


    lr = best_params['learning_rate']
    bs = best_params['batch_size']
    dr = best_params['dropout_rate']

    if config.pruning_method == -1:
        return best_params, None

    all_seed_scores = []
    for seed in range(config.use_avg_metrics_with_number_seeds):
        config.metric_seed = seed

    # for seed in range(1):

        set_seed(config.metric_seed)

        train_loader = DataLoader(base_train_dataset, batch_size=bs, shuffle=True, drop_last=False)
        model = BostonHousingModel(config.input_dim, config.hidden_layers, dr).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_metric_cal = nn.CrossEntropyLoss(reduction='none')
        iterations = config.iterations

        model.to(device)
        iteration = 0

        num_epochs = int(math.ceil(iterations / len(train_loader)))

        # all samples include train and test samples
        loss_all_epoch = np.zeros((len(base_train_dataset), num_epochs + 1))
        loss_all_epoch[:, 0] = np.arange(len(base_train_dataset))

        saved_path = f'./mnist_CNN_saved_loss_values/seed{config.metric_seed}_lr{lr}_bs{bs}_dr{dr}_iterations{iterations}.npy'

        if not os.path.isfile(saved_path):
            epoch = 0
            while iteration < iterations:
                model.train()
                for batch_X, batch_y, sample_ID in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion_metric_cal(outputs, batch_y)

                    loss.mean().backward()
                    optimizer.step()

                    iteration += 1

                    loss_all_epoch[sample_ID.numpy(), epoch + 1] = loss.detach().cpu().squeeze().numpy()
                    # loss_all_epoch[sample_ID.numpy(), 0] = sample_ID.numpy()
                    # if iteration % 10 == 0:
                    #     test_loss = evaluate_model(model, test_loader, criterion, device)
                    #     test_losses.append(test_loss)

                    if iteration >= iterations:
                        break

                epoch += 1

            # 删除test sample所属的行数：找到需要保留的行：检查第 1 列到最后一列是否存在非零值
            # unprocessed_samples = np.where(loss_all_epoch[:, epoch + 1] == 0)[0]
            # sum(np.any(loss_all_epoch[:, 1:] != 0, axis=1))
            # mask = np.any(loss_all_epoch[:, 1:] != 0, axis=1)
            # # 保留满足条件的行
            # loss_all_epoch = loss_all_epoch[mask]
            if not os.path.exists(os.path.dirname(saved_path)):
                os.makedirs(os.path.dirname(saved_path))
            np.save(saved_path, loss_all_epoch)

        else:
            loss_all_epoch = np.load(saved_path)

        if config.pruning_method == 0:
            # avg loss
            # 计算第 1 列到最后一列的均值
            avg_loss_matrix = loss_all_epoch[:,:2].copy()
            avg_loss_matrix[:, 1] = loss_all_epoch[:, 1:].mean(axis=1)
            sorted_metrics = avg_loss_matrix[avg_loss_matrix[:, 1].argsort()]

            all_seed_scores.append(avg_loss_matrix[:,1])

        elif config.pruning_method == 1:
            # S2L methods to be done
            n_clusters = 350
            from sklearn.cluster import KMeans
            data = loss_all_epoch[:,1:]
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=30, random_state=42)
            kmeans.fit(data)

            # Get one sample from each cluster
            unique_labels = np.unique(kmeans.labels_)
            samples_from_clusters = []

            for label in unique_labels:
                sample = data[kmeans.labels_ == label][0]  # Take the first sample from each cluster
                samples_from_clusters.append(sample)

            # Count the number of samples in each cluster
            cluster_counts = np.bincount(kmeans.labels_)

            # Create a dictionary with cluster ranks as keys and indices as values
            cluster_indices_dict = {}
            sorted_clusters = sorted(range(len(cluster_counts)), key=lambda x: cluster_counts[x], reverse=False)

            for rank, cluster_index in enumerate(sorted_clusters, start=1):
                indices_in_cluster = np.where(kmeans.labels_ == cluster_index)[0]
                cluster_indices_dict[rank] = indices_in_cluster.tolist()


            # Step 3: Iterate through each cluster to select examples
            sorted_metrics = {}
            for pruning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                keep_rate = 1.0 - pruning_rate
                B = int(keep_rate * data.shape[0])
                K = n_clusters
                S = []
                R_k_lists = []
                remained_clusters_size = []
                for k, cluster_examples in cluster_indices_dict.items():
                    # print(k)
                    R_k = (B - len(S)) // (K - k + 1)

                    if len(cluster_examples) <= R_k:
                        S.extend(cluster_examples)
                        remained_clusters_size.append(len(cluster_examples))
                        # R_k_lists.append(len(cluster_examples))
                    else:
                        S.extend(random.sample(cluster_examples, R_k))
                        remained_clusters_size.append(R_k)

                    R_k_lists.append(R_k)
                # 获取范围
                n_all_samples = len(base_train_dataset)
                # 求补集
                full_set = set(range(n_all_samples))
                subset = set(S)
                removed_IDs = full_set - subset

                # 转为列表并排序（可选）
                removed_IDs = sorted(list(removed_IDs))
                sorted_metrics[pruning_rate] = removed_IDs



            # np.save('./3xA100_ag_1_graphcast_seed3_loss_kmean_350_k10.npy', np.array(S))

        elif config.pruning_method == 2:
            num_samples = loss_all_epoch.shape[0]
            loss_differences = np.zeros((num_samples, num_epochs))  # Matrix to store loss differences
            loss_differences[:, 0] = loss_all_epoch[:, 0]  # Set the first column as sample IDs

            # Step 3: Calculate loss differences
            for epoch_idx in range(1, num_epochs):
                loss_diff = loss_all_epoch[:, epoch_idx] - loss_all_epoch[:, epoch_idx-1]
                loss_differences[:, epoch_idx] = loss_diff

            sample_ids = loss_differences[:, 0]  # 样本 ID
            loss_values = loss_differences[:, 1:]  # g_t(x_n)，去掉样本 ID


            # # 计算 R(x_n)
            # R_xn = np.zeros(len(sample_ids))  # 初始化 R(x_n) 的结果
            #
            # for i in range(len(sample_ids)):
            #     g_t = np.abs(loss_values[i, :])  # 取 |g_t(x_n)|
            #     g_mean = np.mean(g_t)  # 计算 \overline{g(x_n)}
            #     R_xn[i] = np.sum((g_t - g_mean) ** 2)  # 按公式求和

            # 计算 |g_t(x_n)|
            g_t = np.abs(loss_values)  # 取绝对值
            # 计算 \overline{g(x_n)}（按行求平均值）
            g_mean = np.mean(g_t, axis=1, keepdims=True)  # 保留维度以便后续广播
            # 按公式求和以计算 R(x_n)
            R_xn = np.sum((g_t - g_mean) ** 2, axis=1)  # 按行求和

            # 根据 R_xn 对样本 ID 排序
            all_seed_scores.append(R_xn)
            sorted_indices = np.argsort(R_xn)  # 获取排序索引
            sorted_sample_ids = sample_ids[sorted_indices]  # 按索引对样本 ID 排序
            sorted_sample_ids = sorted_sample_ids.astype(int)
            sorted_R_xn_values = R_xn[sorted_indices]  # 按索引对 R_xn 排序
            # 合并排序后的结果
            sorted_metrics = np.column_stack((sorted_sample_ids, sorted_R_xn_values))



        # best_test_loss = min(losses)
        # best_score = sum(losses) / len(losses)

    all_seed_scores = np.array(all_seed_scores)
    all_seed_scores = np.mean(all_seed_scores, axis=0)
    sorted_metrics = np.vstack((loss_all_epoch[:, 0], all_seed_scores)).T
    sorted_metrics = sorted_metrics[sorted_metrics[:, 1].argsort()]

    print(f"Parameters: {best_params}")
    return best_params, sorted_metrics

# =================================
# 7. 多次运行并记录平均损失
# =================================
def run_multiple_seeds(base_train_dataset, test_loader, best_params, config, device, seeds=10, sorted_metrics=None):
    """
    使用找到的 best_params 多次训练（不同随机种子），
    并将每个 epoch 的平均测试损失记录到 wandb。
    """
    all_test_results = []

    for seed_i in range(seeds):

        set_seed(seed_i)
        if config.cal_metric_for_each_seed == 1:
            config.metric_seed = seed_i
            _, sorted_metrics = call_pruning_metrics(base_train_dataset, test_loader,
                                                               config, device=device)
        if config.pruning_method == -1:
            pruned_dataset, removed_ids_set = random_prune_dataset(base_train_dataset, config.pruning_rate)
        else:
            pruned_dataset, removed_ids_set = prune_dataset(base_train_dataset, config.pruning_rate, sorted_metrics)

        train_loader = DataLoader(pruned_dataset,
                                  batch_size=best_params['batch_size'],
                                  shuffle=True, drop_last=False)

        cal_removed_ratio = (len(base_train_dataset) - len(pruned_dataset)) / len(base_train_dataset)
        print(f"start training with Seed {seed_i}, pruning rate: {config.pruning_rate}, "
              f"len_all_dataset: {len(base_train_dataset)}, "
              f"len_train_dataset: {len(pruned_dataset)}, "
              f"removed_dataset: {cal_removed_ratio:.2f}")

        model = BostonHousingModel(config.input_dim, config.hidden_layers, best_params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        test_results = train_model(model, train_loader, test_loader, optimizer, criterion,
                                   device=device,
                                   iterations=config.iterations,
                                   seed=None, removed_ids_set=removed_ids_set)
        all_test_results.append(test_results)

    avg_test_results = np.mean(all_test_results, axis=0)
    for epoch, (avg_loss, avg_accuracy) in enumerate(avg_test_results, start=1):
        wandb.log({"Iteration": epoch * 10, "Avg Test Loss": avg_loss, "Avg Test Accuracy": avg_accuracy})

# =================================
# 8. 主流程（main 函数）
# =================================
def main():
    parser = argparse.ArgumentParser(description="Boston Housing Price Prediction (with optional pruning)")

    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--pruning_rate', type=float, default=0.0, help='Dataset pruning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64], help='Sizes of hidden layers')
    parser.add_argument('--wandb_project', type=str,
                        default='m-classification', help='WandB project name')
    # parser.add_argument('--wandb_name', type=str, default='random', help='WandB run name')
    parser.add_argument('--wandb_name', type=str, default='only_for_test', help='WandB run name')
    parser.add_argument('--seeds', type=int, default=2, help='Number of random seeds for multiple runs')
    parser.add_argument('--metric_seed', type=int, default=0, help='random seed to calculate metrics')
    parser.add_argument('--pruning_method', type=int, default=0, help='-1 - random pruning, 0 - avg loss, 1 - S2L, 2 - TTDS')
    parser.add_argument('--cal_metric_for_each_seed', type=int, default=0, help='if cal ranking list for each seeds')
    parser.add_argument('--use_avg_metrics_with_number_seeds', type=int, default=10, help='ranking list is calculated using avg '
                                                                                         'with use_avg_metrics_with_number_seeds seeds')

    args = parser.parse_args()

    config = SimpleNamespace(
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        pruning_rate=args.pruning_rate,
        hidden_layers=args.hidden_layers,
        seeds=args.seeds,
        pruning_method=args.pruning_method,
        metric_seed=args.metric_seed,
        cal_metric_for_each_seed=args.cal_metric_for_each_seed,
        use_avg_metrics_with_number_seeds=args.use_avg_metrics_with_number_seeds,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("Loading dataset ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_indices = torch.arange(len(train_dataset))
    train_dataset = TensorDataset(
        torch.cat([train_dataset[i][0].unsqueeze(0) for i in train_indices]),
        torch.tensor([train_dataset[i][1] for i in train_indices]),
        train_indices
    )

    test_indices = torch.arange(len(test_dataset))
    test_dataset = TensorDataset(
        torch.cat([test_dataset[i][0].unsqueeze(0) for i in test_indices]),
        torch.tensor([test_dataset[i][1] for i in test_indices]),
        test_indices
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    config.input_dim = 28 * 28

    # best_params = hyperparameter_search(train_dataset, test_loader, config, device=device)

    best_params, sorted_metrics = call_pruning_metrics(train_dataset, test_loader,
                                       config, device=device)

    run_name = (
        f"{args.wandb_name}_prune{config.pruning_rate}_"
        f"seeds{config.seeds}"
    )
    wandb.init(project=config.wandb_project, name=run_name, config={
        "iterations": config.iterations,
        "batch_size": best_params['batch_size'],
        "learning_rate": best_params['learning_rate'],
        "dropout_rate": best_params['dropout_rate'],
        "hidden_layers": config.hidden_layers,
        "pruning_rate": config.pruning_rate,
        "number_seeds": config.seeds,
        "metric_seed": config.metric_seed,
        "pruning_method": config.pruning_method,
        "cal_metric_for_each_seed": config.cal_metric_for_each_seed,
        "use_avg_metrics_with_number_seeds": config.use_avg_metrics_with_number_seeds,
    })

    run_multiple_seeds(train_dataset, test_loader, best_params, config, device=device, seeds=config.seeds, sorted_metrics=sorted_metrics)

if __name__ == "__main__":
    main()
