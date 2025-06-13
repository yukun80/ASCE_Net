# predict.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import config
from model.asc_net import ASCNetComplex
from utils.dataset import read_sar_complex_tensor


def predict_single_image(model_path, sar_image_path):
    # 加载模型
    model = ASCNetComplex(n_channels=1, n_params=2)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # 加载和预处理SAR图像 (使用与训练时相同的函数)
    sar_tensor = read_sar_complex_tensor(sar_image_path, config.IMG_HEIGHT, config.IMG_WIDTH)
    if sar_tensor is None:
        print(f"Failed to read image {sar_image_path}")
        return

    sar_tensor = sar_tensor.to(config.DEVICE)  # .unsqueeze(0) is already done in read_sar_complex_tensor

    # 预测
    with torch.no_grad():
        predicted_params = model(sar_tensor)

    # 将结果转换回numpy数组
    predicted_params_np = predicted_params.squeeze(0).cpu().numpy()
    pred_A = predicted_params_np[0, :, :]
    pred_alpha = predicted_params_np[1, :, :]

    # 用于可视化的原始图像幅度
    # 从复数张量中获取幅度
    sar_magnitude = sar_tensor.abs().squeeze().cpu().numpy()

    # 可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(sar_magnitude, cmap="gray")
    axes[0].set_title("Original SAR Magnitude")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_A, cmap="hot")
    axes[1].set_title("Predicted Amplitude (A)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(pred_alpha, cmap="hot")
    axes[2].set_title(r"Predicted Freq. Dependence ($\alpha$)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2])

    # 简单后处理：找到预测的散射中心
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(pred_A.flatten(), height=0.1)  # height是阈值，需要调整
    peak_rows, peak_cols = np.unravel_index(peaks, pred_A.shape)

    axes[3].imshow(sar_magnitude, cmap="gray")
    axes[3].scatter(peak_cols, peak_rows, c="red", s=10, marker="x")
    axes[3].set_title("Overlayed Scatterer Peaks")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 使用示例
    # 请确保你有一个训练好的模型，并提供一张 .raw 图像的路径
    model_file = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)

    # 找一张图像作为例子
    # 你需要根据你的文件结构修改这个路径
    example_image_path = (
        "datasets/SAR_ASC_Project/02_Data_Processed_raw/test_15_deg/BMP2/SN_9563/HB03333.000.128x128.raw"
    )

    if os.path.exists(model_file) and os.path.exists(example_image_path):
        predict_single_image(model_path=model_file, sar_image_path=example_image_path)
    else:
        print("Model file or example image not found.")
        print(f"Looking for model at: {model_file}")
        print(f"Looking for image at: {example_image_path}")
