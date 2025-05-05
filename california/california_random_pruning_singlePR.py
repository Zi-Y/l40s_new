import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random, os
from types import SimpleNamespace
from sklearn.datasets import fetch_california_housing

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

def prune_dataset(dataset, pruning_rate):
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
    return Subset(dataset, indices)

# 定义 prune_dataset 函数
# def prune_dataset(X, y, ids, pruning_rate):
#     """
#     从给定数据中随机抽取 (1 - pruning_rate) 比例的样本，保留用户提供的样本 ID。
#     """
#     num_samples = len(X)
#     num_to_keep = int(num_samples * (1 - pruning_rate))
#     if num_to_keep <= 0:
#         raise ValueError(
#             f"After pruning, there are 0 samples left. "
#             f"Please reduce pruning_rate (<1.0) or enlarge the dataset."
#         )
#
#     # 随机打乱样本并抽取部分索引
#     indices = torch.randperm(num_samples)[:num_to_keep]
#
#     # 按索引提取数据、标签和样本 ID
#     pruned_X = X[indices]
#     pruned_y = y[indices]
#     pruned_ids = ids[indices]
#
#     # 返回包含 ID 的数据集
#     dataset = TensorDataset(pruned_X, pruned_y, pruned_ids)
#     return dataset

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
            # layers.append(nn.BatchNorm1d(units))
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =================================
# 4. 评估函数
# =================================
def evaluate_model(model, loader, criterion, device):
    """
    评估模型在给定 loader 上的平均损失（MSE）。
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y, sample_ID in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

# =================================
# 5. 单次训练过程
# =================================
def train_model(model, train_loader, test_loader, optimizer, criterion, device, iterations, seed=None):
    """
    对 model 进行若干次迭代的训练，并在每个 epoch（完整遍历训练集）结束后在 test_loader 上做评估。
    如果传入了 seed，则在本函数内再次固定随机种子，保证可复现性。
    返回每个 epoch 的测试集 loss 列表。
    """
    if seed is not None:
        set_seed(seed)

    model.to(device)
    test_losses = []
    iteration = 0

    while iteration < iterations:
        model.train()
        for batch_X, batch_y, sample_ID in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            iteration += 1
            # if iteration % len(train_loader) == 0:
            if iteration % 10 == 0:
                test_loss = evaluate_model(model, test_loader, criterion, device)
                test_losses.append(test_loss)

            if iteration >= iterations:
                break

    return test_losses

# =================================
# 6. 超参数搜索
# =================================
def hyperparameter_search(base_train_dataset, test_loader, config, device):
    """
    在给定的 base_train_dataset 上进行超参数网格搜索，每次都会重新裁剪数据集并训练。
    返回在搜索中得到的最佳超参数组合。
    """
    print("Starting hyperparameter optimization...")
    param_grid = {
        'learning_rate': [0.01, ],
        # 'learning_rate': [0.01, 0.001, 0.0001],
        # 'batch_size': [1024, 2048, 4096],
        'batch_size': [1024],
        # 'batch_size': [32,],
        # 'dropout_rate': [0.1, 0.2, 0.3]
        'dropout_rate': [0.1]
    }
    best_params = None
    best_score = float('inf')

    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dr in param_grid['dropout_rate']:
                set_seed(42)
                pruned_dataset = prune_dataset(base_train_dataset, config.pruning_rate)
                train_loader = DataLoader(pruned_dataset, batch_size=bs, shuffle=True, drop_last=False)

                model = BostonHousingModel(config.input_dim, config.hidden_layers, dr).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                iterations = config.iterations
                losses = train_model(model, train_loader, test_loader, optimizer, criterion,
                                     device=device,
                                     iterations=iterations,
                                     seed=None)
                # best_test_loss = min(losses)
                best_test_loss = sum(losses[20:]) / len(losses[20:])
                print(f"[lr={lr}, bs={bs}, dr={dr}] -> best_test_loss = {best_test_loss:.6f}")

                if best_test_loss < best_score:
                    best_score = best_test_loss
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'dropout_rate': dr}

    print(f"Best parameters: {best_params} with Test Loss: {best_score}")
    return best_params

# =================================
# 7. 多次运行并记录平均损失
# =================================
def run_multiple_seeds(base_train_dataset, test_loader, best_params, config, device, seeds=10):
    """
    使用找到的 best_params 多次训练（不同随机种子），
    并将每个 epoch 的平均测试损失记录到 wandb。
    """
    all_test_losses = []

    for seed_i in range(seeds):
        set_seed(0)
        pruned_dataset = prune_dataset(base_train_dataset, config.pruning_rate)

        set_seed(seed_i)
        train_loader = DataLoader(pruned_dataset,
                                  batch_size=best_params['batch_size'],
                                  shuffle=True, drop_last=False)

        model = BostonHousingModel(config.input_dim, config.hidden_layers, best_params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        test_losses = train_model(model, train_loader, test_loader, optimizer, criterion,
                                  device=device,
                                  iterations=config.iterations,
                                  seed=None)
        all_test_losses.append(test_losses)

    avg_test_losses = np.mean(all_test_losses, axis=0)
    for epoch, avg_loss in enumerate(avg_test_losses, start=1):
        wandb.log({"Iteration": epoch*10, "Avg Test Loss": avg_loss})

# =================================
# 8. 主流程（main 函数）
# =================================
def main():
    parser = argparse.ArgumentParser(description="Boston Housing Price Prediction (with optional pruning)")

    parser.add_argument('--iterations', type=int, default=1500, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--pruning_rate', type=float, default=0.1, help='Dataset pruning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 32], help='Sizes of hidden layers')
    parser.add_argument('--wandb_project', type=str,
                        # default='boston-prediction', help='WandB project name')
                        default='california_housing', help='WandB project name')
    # parser.add_argument('--wandb_name', type=str, default='random', help='WandB run name')
    parser.add_argument('--wandb_name', type=str, default='random', help='WandB run name')
    parser.add_argument('--seeds', type=int, default=50, help='Number of random seeds for multiple runs')
    args = parser.parse_args()

    config = SimpleNamespace(
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        pruning_rate=args.pruning_rate,
        hidden_layers=args.hidden_layers,
        seeds=args.seeds,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)


    print("Loading dataset ...")
    data = fetch_california_housing()
    X, y = data.data, data.target



    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X = X_scaler.fit_transform(X)
    y = np.expand_dims(y, axis=-1)
    y = Y_scaler.fit_transform(y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    IDs = torch.arange(len(y))

    dataset = TensorDataset(X, y, IDs)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    config.input_dim = X.shape[1]

    best_params = hyperparameter_search(train_dataset, test_loader, config, device=device)

    run_name = (
        f"{args.wandb_name}_prune{config.pruning_rate}_"
        f"lr{best_params['learning_rate']}_bs{best_params['batch_size']}_"
        f"dr{best_params['dropout_rate']}"
    )
    wandb.init(project=config.wandb_project, name=run_name, config={
        "iterations": config.iterations,
        "batch_size": best_params['batch_size'],
        "learning_rate": best_params['learning_rate'],
        "dropout_rate": best_params['dropout_rate'],
        "hidden_layers": config.hidden_layers,
        "pruning_rate": config.pruning_rate,
        "number_seeds": config.seeds,
    })

    run_multiple_seeds(train_dataset, test_loader, best_params, config, device=device, seeds=config.seeds)

if __name__ == "__main__":
    main()
