# utils/config.py

import torch
import numpy as np

# --- 数据集路径 ---
# 原始MSTAR数据处理后得到的ASC .mat文件根目录
ASC_MAT_ROOT = r"datasets\SAR_ASC_Project\tmp_Training_ASC"

# 对应的SAR图像.raw文件根目录 (用于模型输入)
SAR_RAW_ROOT = r"datasets\SAR_ASC_Project\tmp_Data_Processed_raw"

# 预处理后，用于训练的参数图像标签的保存路径
LABEL_SAVE_ROOT = r"datasets\SAR_ASC_Project\tmp_MSTAR_ASC_LABELS"

# --- 模型和训练参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5  # <<< CHANGED: Lowered the learning rate
WEIGHT_DECAY = 1e-5  # <<< ADDED: L2 Regularization
NUM_EPOCHS = 50
CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = "best_model.pth"

# --- 图像和模型物理参数 ---
# MSTAR图像尺寸
IMG_HEIGHT = 128
IMG_WIDTH = 128
# K-space处理网格大小，用于坐标变换
P_GRID_SIZE = 84

# --- 损失函数权重 ---
# 论文中没有明确给出，我们假设权重为1。你可以根据训练情况调整
LOSS_WEIGHT_NONZERO = 1.0


# --- 物理和重建参数 ---
# 这些参数对于从ASC重建图像至关重要
C = 3e8  # 光速 (m/s)
F_C = 9.6e9  # 雷达中心频率 (Hz) - MSTAR通常使用9.6GHz
BANDWIDTH = 591e6  # 雷达带宽 (Hz)
PHI_M = 4 * np.pi / 180  # 成像角度范围 (弧度)

# MSTAR数据集的标称分辨率，用于像素坐标到物理坐标的转换
PIXEL_SPACING = 0.3  # 像素间距 (m)
