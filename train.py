# train.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import config
from utils.dataset import MSTAR_ASC_Dataset
from utils.loss import ASCLoss
from model.asc_net import ASCNetComplex


def main():
    # 1. 创建数据集和DataLoader
    print("Loading dataset...")
    dataset = MSTAR_ASC_Dataset(sar_root=config.SAR_RAW_ROOT, label_root=config.LABEL_SAVE_ROOT)
    if len(dataset) == 0:
        print("Dataset is empty. Please run step4_label_preprocess.py first.")
        return

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 2. 初始化模型、损失函数和优化器
    print(f"Using device: {config.DEVICE}")
    model = ASCNetComplex(n_channels=1, n_params=2).to(config.DEVICE)
    criterion = ASCLoss(weight_nonzero=config.LOSS_WEIGHT_NONZERO)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 确保保存模型的目录存在
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 3. 训练循环
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        loop = tqdm(loader, leave=True)
        epoch_loss = 0.0

        for i, (sar_image, true_params) in enumerate(loop):
            sar_image = sar_image.to(config.DEVICE)
            true_params = true_params.to(config.DEVICE)

            # 前向传播
            predicted_params = model(sar_image)

            # 计算损失
            loss = criterion(predicted_params, true_params)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Average Loss: {avg_epoch_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
        print(f"Checkpoint saved to {os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)}")

    print("Training finished!")


if __name__ == "__main__":
    main()
