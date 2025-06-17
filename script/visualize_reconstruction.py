# script/visualize_reconstruction.py (修改后)

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image  # --- 新增 --- 导入PIL库用于图像读取

# --- Setup Project Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model.asc_net import ASCNet_v3_5param
from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset, read_sar_complex_tensor
from utils.reconstruction import (
    extract_scatterers_from_mat,
    extract_scatterers_from_prediction_5ch,
    reconstruct_sar_image,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# --- 修改 --- 新增一个辅助函数，用于根据SAR文件路径找到对应的JPG文件
def find_corresponding_jpg_file(sar_path):
    """根据SAR .raw文件路径，找到对应的_v1.JPG预览图路径。"""
    # 定义JPG预览图的根目录
    jpg_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "02_Data_Processed_jpg_tmp", "test_15_deg", "T72", "SN_132")

    # 从SAR .raw文件路径中获取相对路径和基本文件名
    rel_path = os.path.relpath(os.path.dirname(sar_path), config.SAR_RAW_ROOT)
    base_name = os.path.basename(sar_path).replace(".128x128.raw", "_v1.JPG")  # 对应于step2脚本生成的v1版本

    return os.path.join(jpg_root, rel_path, base_name)


def find_corresponding_mat_file(sar_path):
    """Finds the original .mat label file corresponding to a SAR image path."""
    rel_path = os.path.relpath(os.path.dirname(sar_path), config.SAR_RAW_ROOT)
    base_name = os.path.basename(sar_path).replace(".128x128.raw", "_yang.mat")
    return os.path.join(config.ASC_MAT_ROOT, rel_path, base_name)


def main():
    print("Loading the trained model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reconstruction plots will be saved in: {output_dir}")

    print("Discovering dataset samples...")
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("No valid samples found.")
        return
    print(f"Found {len(dataset.samples)} samples to visualize.")

    for sample_info in tqdm(dataset.samples, desc="Generating Reconstructions"):
        sar_path = os.path.normpath(sample_info["sar"])
        base_name = os.path.basename(sar_path).replace(".128x128.raw", "")

        # --- 修改: 加载原始SAR图像部分 ---
        # a. 从JPG文件加载用于显示的原始SAR图像
        jpg_path = find_corresponding_jpg_file(sar_path)
        if not os.path.exists(jpg_path):
            print(f"Warning: Skipping {base_name}, JPG file not found at {jpg_path}")
            continue
        original_sar_image_display = Image.open(jpg_path)

        # b. 仍然加载原始复数数据用于模型推理
        sar_tensor_for_model = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
        if sar_tensor_for_model is None:
            continue

        # c. Reconstruct from GROUND TRUTH (using pristine .mat file)
        gt_mat_path = find_corresponding_mat_file(sar_path)
        gt_scatterers = extract_scatterers_from_mat(gt_mat_path)
        recon_gt_img = reconstruct_sar_image(gt_scatterers)

        # d. Reconstruct from PREDICTION
        with torch.no_grad():
            input_tensor = sar_tensor_for_model.unsqueeze(0).to(config.DEVICE)
            predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

        pred_scatterers = extract_scatterers_from_prediction_5ch(predicted_maps)
        recon_pred_img = reconstruct_sar_image(pred_scatterers)

        # e. Visualize and Save
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Image Reconstruction Comparison: {base_name}", fontsize=16)

        # --- 修改: 更新可视化部分 ---
        # 使用从JPG加载的图像进行显示，并计算vmax用于其他两个图
        vmax = np.percentile(np.abs(sar_tensor_for_model[0].numpy()), 99.9)

        axes[0].imshow(original_sar_image_display, cmap="gray")
        axes[0].set_title("Original SAR Image (from JPG)")
        axes[0].axis("off")

        axes[1].imshow(np.abs(recon_gt_img), cmap="gray", vmin=0, vmax=vmax)
        axes[1].set_title("Reconstruction from GT (.mat)")
        axes[1].axis("off")

        axes[2].imshow(np.abs(recon_pred_img), cmap="gray", vmin=0, vmax=vmax)
        axes[2].set_title("Reconstruction from Prediction")
        axes[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"reconstruction_{base_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"\nReconstruction and visualization complete. Check the '{output_dir}' folder.")


if __name__ == "__main__":
    main()
