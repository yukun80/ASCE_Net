import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model.asc_net import ASCNetComplex_v2
from utils.dataset import MSTAR_ASC_Dataset
from utils import config
import os


def predict_and_visualize():
    """
    加载训练好的模型，对验证集中的一个样本进行推理，并可视化结果。
    """
    print("--- Starting Inference ---")

    # 1. 设置设备
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. 加载模型
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {model_path}")

    # 3. 加载数据集并获取一个验证样本
    # 确保使用与train.py中相同的随机种子，以获得相同的验证集
    dataset = MSTAR_ASC_Dataset(root_dir=config.MSTAR_DATA_DIR_AS_MAT)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 获取一个批次（即一个样本）
    try:
        sar_img_complex, true_params = next(iter(val_loader))
    except StopIteration:
        print("Validation set is empty.")
        return

    # 4. 执行推理
    with torch.no_grad():
        predicted_params = model(sar_img_complex.to(device))

    # 5. 将Tensor转为Numpy数组，并移至CPU
    sar_img_complex = sar_img_complex.squeeze(0).cpu().numpy()
    true_params = true_params.squeeze(0).cpu().numpy()
    predicted_params = predicted_params.squeeze(0).cpu().detach().numpy()

    # 提取SAR幅度图用于显示
    sar_img_mag = np.abs(sar_img_complex[0] + 1j * sar_img_complex[1])

    # 6. 逆变换并分离参数
    # 对真实标签进行逆变换以便比较
    true_A_vis = np.expm1(true_params[0, :, :])
    true_alpha = true_params[1, :, :]

    # 对预测结果进行逆变换
    pred_A_vis = np.expm1(predicted_params[0, :, :])
    pred_alpha = predicted_params[1, :, :]

    print("Inference complete. Generating visualization...")

    # 7. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Model Inference vs. Ground Truth", fontsize=20)

    # 第一行：幅度 A
    ax = axes[0, 0]
    im = ax.imshow(sar_img_mag, cmap="gray")
    ax.set_title("Input SAR Image (Magnitude)")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(true_A_vis, cmap="jet")
    ax.set_title("Ground Truth: Amplitude (A)")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(pred_A_vis, cmap="jet")
    ax.set_title("Prediction: Amplitude (A)")
    fig.colorbar(im, ax=ax)

    # 第二行：Alpha
    axes[1, 0].axis("off")  # 隐藏左下角的空图

    ax = axes[1, 1]
    im = ax.imshow(true_alpha, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Ground Truth: Alpha (α)")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.imshow(pred_alpha, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Prediction: Alpha (α)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    output_filename = "inference_result.png"
    plt.savefig(output_filename)
    print(f"Visualization saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    predict_and_visualize()
