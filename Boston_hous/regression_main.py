import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
from types import SimpleNamespace

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# for pruning_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
# 初始化wandb
pruning_rate = 0.1  # 默认裁剪比例


cfg={
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "hidden_layers": [64, 64],
    "evaluate_interval": 1,
    "pruning_rate": pruning_rate
}

config = SimpleNamespace(**cfg)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset from local file...")
data = pd.read_csv("boston.csv")

# 提取特征和目标
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Preprocessing data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.expand_dims(y.to_numpy(), axis=-1)

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

print("Splitting dataset...")
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 实现裁剪功能
def prune_dataset(dataset, pruning_rate):
    num_samples = len(dataset)
    num_to_keep = int(num_samples * (1 - pruning_rate))
    indices = torch.randperm(num_samples)[:num_to_keep]
    pruned_dataset = torch.utils.data.Subset(dataset, indices)
    return pruned_dataset

train_dataset = prune_dataset(train_dataset, config.pruning_rate)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

# 构建模型
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

# 评估模型
def evaluate_model(model, test_loader, criterion, epoch=None):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_loader.dataset)

    if epoch is not None:
        wandb.log({"Epoch": epoch, "Test Loss": test_loss})

    return test_loss

# 超参数搜索
def hyperparameter_search():
    print("Starting hyperparameter optimization...")
    param_grid = {
        'learning_rate': [0.001, 0.01,],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
    best_params = None
    best_score = float('inf')

    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dr in param_grid['dropout_rate']:
                print(f"Testing lr={lr}, bs={bs}, dr={dr}...")
                set_seed(42)
                model = BostonHousingModel(X.shape[1], config.hidden_layers, dr).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                # Train and evaluate
                train_loader = DataLoader(prune_dataset(train_dataset, config.pruning_rate), batch_size=bs, shuffle=True)
                losses = []
                for epoch in range(config.epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    # Evaluate
                    test_loss = evaluate_model(model, test_loader, criterion)
                    losses.append(test_loss)

                best_test_loss = min(losses)
                print(f"Best Test Loss for lr={lr}, bs={bs}, dr={dr}: {best_test_loss}")
                if best_test_loss < best_score:
                    best_score = best_test_loss
                    best_params = {'learning_rate': lr, 'batch_size': bs, 'dropout_rate': dr}

    print(f"Best parameters: {best_params} with Avg Test Loss: {best_score}")
    return best_params

# 多次运行并记录平均损失
def run_multiple_seeds(best_params):
    seeds = range(10)
    all_test_losses = []

    for seed in seeds:
        set_seed(seed)
        model = BostonHousingModel(X.shape[1], config.hidden_layers, best_params['dropout_rate']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()
        train_loader = DataLoader(prune_dataset(train_dataset, config.pruning_rate), batch_size=best_params['batch_size'], shuffle=True)

        test_losses = []
        for epoch in range(config.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            test_loss = evaluate_model(model, test_loader, criterion)
            test_losses.append(test_loss)

        all_test_losses.append(test_losses)
    avg_test_losses = np.mean(all_test_losses, axis=0)
    for epoch, avg_loss in enumerate(avg_test_losses):
            wandb.log({"Epoch": epoch + 1, "Avg Test Loss": avg_loss})

best_params = hyperparameter_search()
run_name = (f"pruning_rate_{pruning_rate}_lr_"
            f"{best_params['learning_rate']}_bs{best_params['batch_size']}_"
            f"dr{best_params['dropout_rate']}")
wandb.init(project="boston-prediction", name=run_name, config={
    "epochs": 50,
    "batch_size": best_params.get('batch_size', 32),
    "learning_rate": best_params.get('learning_rate', 0.001),
    "dropout_rate": best_params.get('dropout_rate', 0.2),
    "hidden_layers": [64, 64],
    "evaluate_interval": 1,
    "pruning_rate": pruning_rate
})
run_multiple_seeds(best_params)

print("Saving final model...")
model = BostonHousingModel(X.shape[1], config.hidden_layers, best_params['dropout_rate']).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.MSELoss()
train_loader = DataLoader(prune_dataset(train_dataset, config.pruning_rate), batch_size=best_params['batch_size'], shuffle=True)

for epoch in range(config.epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

evaluate_model(model, test_loader, criterion)
torch.save(model.state_dict(), "boston_price_prediction_model.pth")
print("Model saved!")
