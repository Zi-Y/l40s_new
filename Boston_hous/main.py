import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 初始化wandb
wandb.init(project="boston-house-price-prediction", config={
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "hidden_layers": [64, 64]
})
config = wandb.config

# 加载波士顿房价数据集
boston = fetch_openml(name="Boston", version=1, as_frame=True)
X, y = boston.data, boston.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = np.expand_dims(y.to_numpy(), axis=-1)  # 确保目标是二维

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 拆分数据集
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
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
        layers.append(nn.Linear(in_dim, 1))  # 输出层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 初始化模型
model = BostonHousingModel(X.shape[1], config.hidden_layers, config.dropout_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# 训练模型
def train_model():
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_loader.dataset)

        # 记录到wandb
        wandb.log({"Epoch": epoch + 1, "Train Loss": epoch_loss})

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {epoch_loss:.4f}")

# 测试模型
def evaluate_model():
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
    test_loss /= len(test_loader.dataset)

    # 记录到wandb
    wandb.log({"Test Loss": test_loss})

    print(f"Test Loss: {test_loss:.4f}")

# 运行训练和测试
train_model()
evaluate_model()

# 保存模型
torch.save(model.state_dict(), "boston_price_prediction_model.pth")
