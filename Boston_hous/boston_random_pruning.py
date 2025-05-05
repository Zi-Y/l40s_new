import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
from types import SimpleNamespace


# ---------------------------------------
# 1. 设置随机种子
# ---------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------
# 2. 数据集裁剪函数
# ---------------------------------------
def prune_dataset(dataset, pruning_rate):
    """
    从给定 dataset 中随机抽取 (1 - pruning_rate) 比例的样本。
    """
    num_samples = len(dataset)
    num_to_keep = int(num_samples * (1 - pruning_rate))
    if num_to_keep <= 0:
        raise ValueError(
            f"After pruning, 0 samples left. "
            f"Please reduce pruning_rate <1.0 or enlarge the dataset."
        )
    indices = torch.randperm(num_samples)[:num_to_keep]
    return Subset(dataset, indices)


# ---------------------------------------
# 3. 模型定义
# ---------------------------------------
class BostonHousingModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(BostonHousingModel, self).__init__()
        layers = []
        in_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(units))
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------
# 4. 评估函数
# ---------------------------------------
def evaluate_model(model, loader, criterion, device, step=None):
    """
    评估模型在给定 loader 上的平均损失（MSE）。
    若传入 step，则会使用 wandb.log({"Test Loss": ...}, step=step).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(loader.dataset)
    # if step is not None:
    #     wandb.log({"Test Loss": avg_loss}, step=step)
    return avg_loss


# ---------------------------------------
# 5. 基于全局迭代次数的训练
# ---------------------------------------
def train_with_iterations(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        total_iters=1000,
        eval_interval=100
):
    """
    使用 "global iteration" 的形式进行训练：
      - total_iters: 总训练迭代次数
      - eval_interval: 每隔多少次迭代评估一次
    """
    model.to(device)
    model.train()
    train_iter = iter(train_loader)  # 将 train_loader 变为迭代器

    for iteration in range(1, total_iters + 1):
        # 如果迭代器走到末尾，则重新创建迭代器（相当于又从头遍历训练集）
        try:
            batch_X, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_X, batch_y = next(train_iter)

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # 训练过程日志（可选）
        # wandb.log({"Train Loss": loss.item()}, step=iteration)

        # 到达 eval_interval 时评估
        if iteration % eval_interval == 0:
            evaluate_model(model, test_loader, criterion, device, step=iteration)


# ---------------------------------------
# 6. 超参数搜索函数 (基于全局迭代)
# ---------------------------------------
def hyperparameter_search_iteration(
        base_train_dataset,
        test_loader,
        input_dim,
        device,
        pruning_rate,
        total_iters,
        eval_interval,
        param_grid
):
    """
    在给定的 base_train_dataset 上做超参数搜索：
      - 会对 base_train_dataset 进行 prune_dataset
      - 使用 train_with_iterations 进行训练
      - 返回测试集最优损失对应的超参数

    param_grid 是一个 dict，例如：
        {
            'learning_rate': [0.001, 0.01],
            'dropout_rate': [0.1, 0.2, 0.3],
            'batch_size': [16, 32]
        }
    """
    best_params = None
    best_score = float('inf')
    criterion = nn.MSELoss()

    for lr in param_grid['learning_rate']:
        for dr in param_grid['dropout_rate']:
            for bs in param_grid['batch_size']:
                # 1) 设置随机种子并裁剪（保证可复现）
                set_seed(999)
                pruned_dataset = prune_dataset(base_train_dataset, pruning_rate)
                train_loader = DataLoader(pruned_dataset, batch_size=bs, shuffle=True, drop_last=False)

                # 2) 初始化模型与优化器
                model = BostonHousingModel(input_dim, hidden_layers=[64, 64], dropout_rate=dr).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # 3) 进行 total_iters 次训练
                train_with_iterations(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    total_iters=total_iters,
                    eval_interval=eval_interval  # 搜索时可以不太频繁评估
                )

                # 4) 评估
                final_loss = evaluate_model(model, test_loader, criterion, device=device, step=None)

                # 5) 更新最好结果
                if final_loss < best_score:
                    best_score = final_loss
                    best_params = {
                        'learning_rate': lr,
                        'dropout_rate': dr,
                        'batch_size': bs
                    }
                print(f"[pr={pruning_rate}, lr={lr}, dr={dr}, bs={bs}] final_loss={final_loss:.5f}")

    print(f"\n>>> Best params for pruning_rate={pruning_rate}: {best_params}, best_score={best_score:.5f}")
    return best_params, best_score


# ---------------------------------------
# 7. 多随机种子训练并取平均
# ---------------------------------------
def run_multiple_seeds_with_iterations(
        base_train_dataset,
        test_loader,
        best_params,
        config,
        device,
        seeds=5
):
    """
    在固定的 pruning_rate + 超参数 best_params 下，
    使用多个随机种子多次训练，记录并返回测试集平均损失。
    """
    all_test_losses = []
    criterion = nn.MSELoss()

    for seed_i in range(seeds):
        # 1) 设随机种子、裁剪
        set_seed(seed_i)
        pruned_dataset = prune_dataset(base_train_dataset, config.pruning_rate)
        train_loader = DataLoader(
            pruned_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            drop_last=False
        )

        # 2) 初始化模型
        model = BostonHousingModel(
            input_dim=config.input_dim,
            hidden_layers=[64, 64],
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        # 3) 训练
        train_with_iterations(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            total_iters=config.total_iters,
            eval_interval=config.eval_interval
        )

        # 4) 最终评估
        final_loss = evaluate_model(model, test_loader, criterion, device=device, step=None)
        all_test_losses.append(final_loss)

    # 计算平均
    avg_test_loss = np.mean(all_test_losses)
    # 记录到 wandb
    wandb.log({"Avg Test Loss over Seeds": avg_test_loss})
    return avg_test_loss


# ---------------------------------------
# 8. 主函数
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Boston Housing with global iteration & HP search & multi-seed")

    # 迭代次数相关
    parser.add_argument("--total_iters", type=int, default=1000, help="Total training iterations.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate every X iterations.")

    # 默认的初始学习率等
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate.")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="boston-iteration", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity.")

    # 是否启用超参数搜索
    parser.add_argument("--enable_search", action="store_true", help="Enable hyperparameter search.")
    # 多种子数量
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for run_multiple_seeds.")
    args = parser.parse_args()

    config = SimpleNamespace(**vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 全局固定分割数据的随机种子
    set_seed(42)

    # ---------------------------------
    # 8.1 读取数据 & 分割
    # ---------------------------------
    data = pd.read_csv("boston.csv")
    X_data = data.iloc[:, :-1].values
    y_data = data.iloc[:, -1].values

    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)
    y_data = np.expand_dims(y_data, axis=-1)

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    config.input_dim = X_data.shape[1]  # 记录特征维度

    # ---------------------------------
    # 8.2 定义搜索网格
    # ---------------------------------
    param_grid = {
        'learning_rate': [0.001, 0.01],
        'dropout_rate': [0.15, 0.2, ],
        'batch_size': [16, 32, 64]
    }

    # ---------------------------------
    # 8.3 遍历 pruning_rate
    # ---------------------------------
    for pr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\n=== [Iter-based training] Start with pruning_rate={pr} ===")

        # 如果启用超参数搜索
        if config.enable_search:
            best_params, best_score = hyperparameter_search_iteration(
                base_train_dataset=train_dataset,
                test_loader=test_loader,
                input_dim=config.input_dim,
                device=device,
                pruning_rate=pr,
                total_iters=config.total_iters,
                eval_interval=config.eval_interval,
                param_grid=param_grid
            )
        else:
            # 否则使用命令行中给定的值
            best_params = {
                'learning_rate': config.learning_rate,
                'dropout_rate': config.dropout_rate,
                'batch_size': config.batch_size
            }

        # 启动一个 wandb run
        run_name = f"PR{pr}_iter{config.total_iters}_lr{best_params['learning_rate']}_bs{best_params['batch_size']}_dr{best_params['dropout_rate']}"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config={
                "pruning_rate": pr,
                "total_iters": config.total_iters,
                "eval_interval": config.eval_interval,
                "learning_rate": best_params['learning_rate'],
                "batch_size": best_params['batch_size'],
                "dropout_rate": best_params['dropout_rate'],
                "seeds": config.seeds
            }
        )

        # 将当前 pruning_rate 也放到 config，以便多种子函数使用
        config.pruning_rate = pr

        # 使用多随机种子进行训练并记录平均测试损失
        avg_loss = run_multiple_seeds_with_iterations(
            base_train_dataset=train_dataset,
            test_loader=test_loader,
            best_params=best_params,
            config=config,
            device=device,
            seeds=config.seeds
        )

        print(f"[pruning_rate={pr}] Average Test Loss over {config.seeds} seeds = {avg_loss:.5f}")
        wandb.finish()


if __name__ == "__main__":
    main()