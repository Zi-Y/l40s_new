import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import wandb, os
import matplotlib.pyplot as plt

# Step 1: 定义 argparse 函数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model for Air Quality Prediction")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model")
    parser.add_argument("--project_name", type=str, default="air-quality-prediction", help="WandB project name")
    parser.add_argument("--entity", type=str, default="your_username", help="WandB entity or username")
    return parser.parse_args()


# Step 2: 下载数据集
# Step 2: 手动提供本地路径
def get_air_quality_dataset(file_path="AirQualityUCI.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}. Please download and extract it manually.")
    return file_path

# 替换之前的下载函数
data_file = get_air_quality_dataset()

# Step 3: 数据预处理
data_all = pd.read_csv(data_file, sep=';', decimal=',', na_values=-200)
# data = data.dropna()
features = ['CO(GT)', 'T', 'RH', 'AH']
data = data_all[features].dropna()

X = data[['T', 'RH', 'AH']].values
y = data['CO(GT)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))


class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Step 4: 定义深度学习模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)


# Step 5: 主训练逻辑
def main(args):
    # 初始化 WandB
    wandb.init(project=args.project_name)
    wandb.config.update({
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "model": "RegressionModel"
    })

    # 数据集准备
    train_dataset = AirQualityDataset(X_train, y_train)
    test_dataset = AirQualityDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型、损失函数和优化器
    model = RegressionModel(input_dim=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    def train_model():
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}")

    train_model()

    # 测试模型
    def evaluate_model():
        model.eval()
        predictions = []
        actuals = []
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                predictions.append(outputs.numpy())
                actuals.append(y_batch.numpy())

        avg_loss = total_loss / len(test_loader)
        wandb.log({"test_loss": avg_loss})
        print(f"Test Loss: {avg_loss:.4f}")
        return np.concatenate(predictions), np.concatenate(actuals)

    predictions, actuals = evaluate_model()

    # 反标准化数据
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)

    # 可视化并记录到 WandB


    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("CO Concentration")
    plt.title("Air Quality Prediction")
    plt.savefig("prediction_plot.png")
    wandb.log({"Prediction Plot": wandb.Image("prediction_plot.png")})
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)