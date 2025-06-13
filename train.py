# train.py (修正版)

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import config
from utils.dataset import MSTAR_ASC_Dataset
from utils.loss import FocusedASCLoss
from model.asc_net import ASCNetComplex_v2


def main():
    # 1. 初始化TensorBoard
    writer = SummaryWriter(log_dir="runs/asc_experiment_1")

    # 2. 创建数据集并划分为训练集和验证集
    print("Loading and splitting dataset...")
    full_dataset = MSTAR_ASC_Dataset(sar_root=config.SAR_RAW_ROOT, label_root=config.LABEL_SAVE_ROOT)

    if len(full_dataset) == 0:
        print("Dataset is empty. Please run preprocess.py first.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset loaded: {train_size} training samples, {val_size} validation samples.")

    # 3. 初始化模型、损失函数、优化器和学习率调度器
    print(f"Using device: {config.DEVICE}")
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)
    criterion = FocusedASCLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 核心修改：移除 verbose=True 参数 ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    # 4. 训练与验证循环
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Training", leave=False)

        for sar_image, true_params in loop:
            sar_image = sar_image.to(config.DEVICE)
            true_params = true_params.to(config.DEVICE)

            optimizer.zero_grad()
            predicted_params = model(sar_image)
            loss = criterion(predicted_params, true_params)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sar_image, true_params in tqdm(
                val_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Validation", leave=False
            ):
                sar_image = sar_image.to(config.DEVICE)
                true_params = true_params.to(config.DEVICE)
                predicted_params = model(sar_image)
                loss = criterion(predicted_params, true_params)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- 学习率调度与模型保存 ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # --- 可视化 ---
        if epoch % 5 == 0:
            sar_image_viz, true_params_viz = next(iter(val_loader))
            sar_image_viz = sar_image_viz.to(config.DEVICE)
            with torch.no_grad():
                predicted_params_viz = model(sar_image_viz)

            writer.add_images("Validation/Input_SAR_mag", sar_image_viz.abs(), epoch, dataformats="NCHW")
            writer.add_images("Validation/True_Amplitude", true_params_viz[:, 0:1], epoch, dataformats="NCHW")
            writer.add_images("Validation/Pred_Amplitude", predicted_params_viz[:, 0:1], epoch, dataformats="NCHW")
            writer.add_images("Validation/True_Alpha", true_params_viz[:, 1:2], epoch, dataformats="NCHW")
            writer.add_images("Validation/Pred_Alpha", predicted_params_viz[:, 1:2], epoch, dataformats="NCHW")

    writer.close()
    print("Training finished!")


if __name__ == "__main__":
    main()

"""
tensorboard --logdir=runs
python train.py
"""
