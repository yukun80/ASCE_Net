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

from model.asc_net import ASCNet_v3_5param  # noqa: E402
from utils import config  # noqa: E402
from utils.dataset import MSTAR_ASC_5CH_Dataset, read_sar_complex_tensor  # noqa: E402
from utils.reconstruction import (  # noqa: E402
    extract_scatterers_from_mat,
    extract_scatterers_from_prediction_5ch,
    reconstruct_sar_image,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def find_corresponding_jpg_file(sar_path):
    """根据SAR .raw文件路径，找到对应的_v1.JPG预览图路径。"""
    # Get current paths based on configuration
    paths = config.get_current_paths()

    if config.USE_COMPLETE_DATASET:
        # For complete dataset, use the same relative path structure as SAR files
        rel_path = os.path.relpath(os.path.dirname(sar_path), paths["sar_raw_root"])
        base_name = os.path.basename(sar_path).replace(".128x128.raw", "_v1.jpg")

        # Define JPG root for complete dataset
        jpg_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "02_Data_Processed_jpg")
        jpg_path = os.path.join(jpg_root, rel_path, base_name)

        # If the JPG file doesn't exist in the expected location, search for it
        if not os.path.exists(jpg_path):
            # Extract class and split from relative path
            path_parts = rel_path.split(os.sep)
            if len(path_parts) >= 2:
                split_name = path_parts[0]  # e.g., "test_15_deg"
                class_name = path_parts[1]  # e.g., "T72"

                # Search through SN subdirectories in JPG folder
                jpg_class_dir = os.path.join(jpg_root, split_name, class_name)
                if os.path.exists(jpg_class_dir):
                    for sn_dir in os.listdir(jpg_class_dir):
                        sn_path = os.path.join(jpg_class_dir, sn_dir)
                        if os.path.isdir(sn_path):
                            potential_jpg = os.path.join(sn_path, base_name)
                            if os.path.exists(potential_jpg):
                                jpg_path = potential_jpg
                                break
    else:
        # Legacy path for testing
        jpg_root = os.path.join(
            project_root, "datasets", "SAR_ASC_Project", "02_Data_Processed_jpg_tmp", "test_15_deg", "T72", "SN_132"
        )
        base_name = os.path.basename(sar_path).replace(".128x128.raw", "_v1.JPG")
        jpg_path = os.path.join(jpg_root, base_name)

    return jpg_path


def find_corresponding_mat_file(sar_path):
    """Finds the original .mat label file corresponding to a SAR image path."""
    # Get current paths based on configuration
    paths = config.get_current_paths()

    rel_path = os.path.relpath(os.path.dirname(sar_path), paths["sar_raw_root"])
    base_name = os.path.basename(sar_path).replace(".128x128.raw", "_yang.mat")
    return os.path.join(paths["asc_mat_root"], rel_path, base_name)


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
    print(f"Dataset mode: {'Complete Dataset' if config.USE_COMPLETE_DATASET else 'Testing Dataset'}")

    # Load test dataset for visualization
    if config.USE_COMPLETE_DATASET:
        dataset = MSTAR_ASC_5CH_Dataset(split="test")  # Use test set for visualization
    else:
        dataset = MSTAR_ASC_5CH_Dataset()  # Use all available data

    if not dataset.samples:
        print("No valid samples found.")
        return
    print(f"Found {len(dataset.samples)} samples to visualize.")

    for i, sample_info in enumerate(tqdm(dataset.samples, desc="Generating Reconstructions")):
        sar_path = os.path.normpath(sample_info["sar"])
        base_name = os.path.basename(sar_path).replace(".128x128.raw", "")

        # Get sample metadata if using complete dataset
        if config.USE_COMPLETE_DATASET:
            class_name = sample_info["class"]
            split_name = sample_info["split"]
            display_name = f"{split_name}_{class_name}_{base_name}"
        else:
            display_name = base_name

        # --- 修改: 加载原始SAR图像部分 ---
        # a. 从JPG文件加载用于显示的原始SAR图像
        jpg_path = find_corresponding_jpg_file(sar_path)
        if not os.path.exists(jpg_path):
            print(f"Warning: Skipping {display_name}, JPG file not found at {jpg_path}")
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
        fig.suptitle(f"Image Reconstruction Comparison: {display_name}", fontsize=16)

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
        save_path = os.path.join(output_dir, f"reconstruction_{display_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

        # Optionally limit the number of visualizations
        if i >= 10:  # Limit to first 10 samples for quick testing
            print(f"Limiting visualization to first {i+1} samples for testing...")
            break

    print(f"\nReconstruction and visualization complete. Check the '{output_dir}' folder.")


if __name__ == "__main__":
    main()
