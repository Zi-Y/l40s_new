import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random, os
from types import SimpleNamespace
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

# =================================
# 3. 定义模型
# =================================
class MNISTModel(nn.Module):
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
        super(MNISTModel, self).__init__()
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
    评估模型在给定 loader 上的分类准确率和损失。
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
def train_model(model, train_loader, test_loader, optimizer, criterion, device, iterations, seed=None):
    """
    对 model 进行若干次迭代的训练，并在每个 epoch（完整遍历训练集）结束后在 test_loader 上做评估。
    如果传入了 seed，则在本函数内再次固定随机种子，保证可复现性。
    返回每个 epoch 的测试集损失和准确率列表。
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
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 10 == 0:
                test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
                test_results.append((test_loss, test_accuracy))

            if iteration >= iterations:
                break

    return test_results

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
        # # 'learning_rate': [0.01, 0.001, 0.0001],
        # # 'batch_size': [256, 512, 1024, 2048, 4096],
        'batch_size': [1024],
        # # 'batch_size': [32,],
        # # 'dropout_rate': [0.0, 0.05, 0.1, 0.15, 0.2]
        'dropout_rate': [0.1]
        # 'learning_rate': 0.01,
        # 'batch_size': 1024,
        # 'dropout_rate': 0.1
    }

    best_params = None
    best_score = 0

    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dr in param_grid['dropout_rate']:
                set_seed(42)
                pruned_dataset = prune_dataset(base_train_dataset, config.pruning_rate)
                train_loader = DataLoader(pruned_dataset, batch_size=bs, shuffle=True, drop_last=False)

                model = MNISTModel(config.input_dim, config.hidden_layers, dr).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                iterations = config.iterations
                results = train_model(model, train_loader, test_loader, optimizer, criterion,
                                      device=device,
                                      iterations=iterations,
                                      seed=None)

                final_accuracy = results[-1][1] if results else 0
                print(f"[lr={lr}, bs={bs}, dr={dr}] -> final_accuracy = {final_accuracy:.6f}")

                if final_accuracy > best_score:
                    best_score = final_accuracy
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'dropout_rate': dr}

    print(f"Best parameters: {best_params} with Accuracy: {best_score:.6f}")
    return best_params

# =================================
# 7. 多次运行并记录平均损失
# =================================
def run_multiple_seeds(base_train_dataset, test_loader, best_params, config, device, seeds=10):
    """
    使用找到的 best_params 多次训练（不同随机种子），
    并将每个 epoch 的平均测试损失和准确率记录到 wandb。
    """
    all_test_results = []

    for seed_i in range(seeds):
        set_seed(seed_i)
        pruned_dataset = prune_dataset(base_train_dataset, config.pruning_rate)
        train_loader = DataLoader(pruned_dataset,
                                  batch_size=best_params['batch_size'],
                                  shuffle=True, drop_last=False)

        model = MNISTModel(config.input_dim, config.hidden_layers, best_params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        test_results = train_model(model, train_loader, test_loader, optimizer, criterion,
                                   device=device,
                                   iterations=config.iterations,
                                   seed=None)
        all_test_results.append(test_results)

    avg_test_results = np.mean(all_test_results, axis=0)
    for epoch, (avg_loss, avg_accuracy) in enumerate(avg_test_results, start=1):
        wandb.log({"Iteration": epoch * 10, "Avg Test Loss": avg_loss, "Avg Test Accuracy": avg_accuracy})

# =================================
# 8. 主流程（main 函数）
# =================================
def main():
    parser = argparse.ArgumentParser(description="MNIST Classification Task")

    parser.add_argument('--iterations', type=int, default=1200, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--pruning_rate', type=float, default=0.1, help='Dataset pruning rate')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 64], help='Sizes of hidden layers')
    parser.add_argument('--wandb_project', type=str, default='m-classification', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='fixed_random', help='WandB run name')
    parser.add_argument('--seeds', type=int, default=1, help='Number of random seeds for multiple runs')
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