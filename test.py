from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer

import os

# 设置环境变量
os.environ["NCCL_DEBUG"] = "INFO"  # 开启 NCCL 调试信息
os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # 指定分布式后端为 gloo
# os.environ["NCCL_P2P_DISABLE"] = "1"  # 指定分布式后端为 gloo
# os.environ["NCCL_SOCKET_IFNAME"] = "ens1f1"  # 指定分布式后端为 gloo
os.environ["NCCL_P2P_LEVEL"] = "NVL"

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # 定义模型
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        # 前向传播
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # 一个训练步骤的逻辑
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        print("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # 定义优化器
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def train_dataloader(self):
        # 定义训练数据加载器
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=32)

def main():
    model = MyModel()
    trainer = Trainer(strategy="ddp",
                      accelerator="gpu",
                      devices=4,)
    trainer.fit(model)

if __name__ == "__main__":
    main()

