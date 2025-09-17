"""
脚本用途: 综合可视化与分析（ASC‑Net）
- 加载 ASC‑Net 训练权重，对数据集样本执行推理，提取散射中心并进行 SAR 重建。
- 生成包含以下内容的综合可视化：
  1) 原始 SAR 预览图
  2) 真值(来自 .mat)重建 与 预测重建 的对比
  3) 重建残差图 (GT - Pred)
  4) 模型输出的 5 个参数通道图 (heatmap, A_log1p/alpha/dx/dy)
  5) 原图叠加预测散射中心、参数分布散点、强度分布直方图
- 批量处理样本并将结果 PNG 输出到 `datasets/result_vis/comprehensive_results/`，同时打印简要统计汇总。

输入/配置:
- 从 `utils.config` 读取数据与模型路径（`SAR_RAW_ROOT`, `ASC_MAT_ROOT`, `CHECKPOINT_DIR`, `MODEL_NAME` 等）。

适用场景:
- 需要一次性查看“推理→散射中心提取→重建→多视图可视化”的全流程效果与统计。

使用示例:
- 直接运行：`python script/visualize_comprehensive.py`

设置数量：
- max_samples = min(1000, len(dataset.samples))
"""

# script/visualize_comprehensive.py
# 全面的ASCE_Net可视化脚本：包含重建对比和5参数预测可视化

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import matplotlib.patches as patches

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

"""
加载模型、对原始样本推理，并生成综合可视化与结果输出。
"""


def find_corresponding_jpg_file(sar_path):
    """根据SAR .raw文件路径，找到对应的JPG预览图路径。"""
    jpg_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Data_Processed_jpg")
    rel_path = os.path.relpath(os.path.dirname(sar_path), config.SAR_RAW_ROOT)
    base_name = os.path.basename(sar_path).replace(".raw", "_v2.JPG")
    return os.path.join(jpg_root, rel_path, base_name)


def find_corresponding_mat_file(sar_path):
    """找到对应的.mat标注文件。"""
    rel_path = os.path.relpath(os.path.dirname(sar_path), config.SAR_RAW_ROOT)
    base_name = os.path.basename(sar_path).replace(".raw", ".mat")
    return os.path.join(config.ASC_MAT_ROOT, rel_path, base_name)


def visualize_parameter_maps(prediction_maps, title_prefix="Predicted"):
    """可视化5个参数通道"""
    param_names = ["Heatmap", "Amplitude (A)", "Alpha (α)", "X-offset (dx)", "Y-offset (dy)"]
    colormaps = ["hot", "viridis_r", "RdBu", "RdBu_r", "RdBu_r"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"{title_prefix} Parameter Maps", fontsize=16)

    for i in range(5):
        param_map = prediction_maps[i]
        im = axes[i].imshow(param_map, cmap=colormaps[i])
        axes[i].set_title(f"{param_names[i]}")
        axes[i].axis("off")

        # 添加颜色条
        plt.colorbar(im, ax=axes[i], orientation="horizontal", pad=0.1, shrink=0.8)

        # 添加统计信息
        stats_text = f"Min: {param_map.min():.3f}\nMax: {param_map.max():.3f}\nMean: {param_map.mean():.3f}"
        axes[i].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def visualize_scatterer_detection(original_image, prediction_maps, detected_scatterers, title="Scatterer Detection"):
    """可视化散射中心检测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # 1. 原始图像 + 检测到的散射中心
    axes[0].imshow(original_image, cmap="gray")
    if detected_scatterers:
        # 将物理坐标转换回像素坐标进行显示
        for scatterer in detected_scatterers:
            # 这里需要逆向转换回像素坐标，暂时使用简化方法
            # 实际应该使用pixel_to_model的逆函数
            x_pixel = int(scatterer["x"] * 32 + 64)  # 简化的坐标转换
            y_pixel = int(64 - scatterer["y"] * 32)  # 简化的坐标转换

            if 0 <= x_pixel < 128 and 0 <= y_pixel < 128:
                circle = patches.Circle((x_pixel, y_pixel), radius=2, color="red", fill=False, linewidth=2)
                axes[0].add_patch(circle)
                # 添加幅度标注
                axes[0].text(
                    x_pixel + 3, y_pixel - 3, f'A:{scatterer["A"]:.2f}', color="red", fontsize=8, weight="bold"
                )

    axes[0].set_title(f"Original + Detected Scatterers\n({len(detected_scatterers)} found)")
    axes[0].axis("off")

    # 2. 热力图 + 检测峰值
    heatmap = prediction_maps[0]
    axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title("Heatmap + Detection Peaks")
    axes[1].axis("off")

    # 3. 幅度图 + 检测结果
    amplitude_map = prediction_maps[1]
    im = axes[2].imshow(amplitude_map, cmap="viridis")
    axes[2].set_title("Amplitude Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], orientation="horizontal", pad=0.1)

    plt.tight_layout()
    return fig


def enhance_contrast(image, gamma=1.2, alpha=1.1):
    """增强图像对比度"""
    # 伽马校正和线性增强
    enhanced = np.power(image / np.max(image), gamma) * alpha
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced


def create_comprehensive_visualization(sample_info, model):
    """创建单个样本的全面可视化"""
    sar_path = os.path.normpath(sample_info["sar"])
    base_name = os.path.basename(sar_path).replace(".raw", "")

    # 1. 加载数据
    jpg_path = find_corresponding_jpg_file(sar_path)
    if not os.path.exists(jpg_path):
        print(f"Warning: JPG file not found at {jpg_path}")
        return None

    original_sar_image_display = Image.open(jpg_path)
    sar_tensor_for_model = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)

    if sar_tensor_for_model is None:
        print(f"Warning: Failed to load SAR tensor for {base_name}")
        return None

    # 2. 模型推理
    with torch.no_grad():
        input_tensor = sar_tensor_for_model.unsqueeze(0).to(config.DEVICE)
        predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

    # 3. 提取散射中心
    pred_scatterers = extract_scatterers_from_prediction_5ch(predicted_maps)

    # 4. 重建图像
    recon_pred_img = reconstruct_sar_image(pred_scatterers)

    # 5. 加载真值对比
    gt_mat_path = find_corresponding_mat_file(sar_path)
    gt_scatterers = extract_scatterers_from_mat(gt_mat_path)
    recon_gt_img = reconstruct_sar_image(gt_scatterers)

    # 6. 创建综合可视化 - 修复布局问题
    # 使用4x4的统一网格布局
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(f"Comprehensive Analysis: {base_name}", fontsize=20, y=0.95)

    # 计算显示范围和对比度增强
    original_array = np.array(original_sar_image_display)
    if len(original_array.shape) == 3:
        original_array = np.mean(original_array, axis=2)  # 转为灰度

    vmax = np.percentile(np.abs(sar_tensor_for_model[0].numpy()), 99.9)

    # 对重建图像进行对比度增强
    recon_gt_enhanced = enhance_contrast(np.abs(recon_gt_img) / vmax)
    recon_pred_enhanced = enhance_contrast(np.abs(recon_pred_img) / vmax)

    # 第一行：原始图像、真值重建、预测重建、残差图
    ax1 = plt.subplot(4, 4, 1)
    plt.imshow(original_array, cmap="gray")
    plt.title("Original SAR Image", fontsize=12)
    plt.axis("off")

    ax2 = plt.subplot(4, 4, 2)
    plt.imshow(recon_gt_enhanced, cmap="gray", vmin=0, vmax=1)
    plt.title(f"GT Reconstruction\n({len(gt_scatterers)} scatterers)", fontsize=12)
    plt.axis("off")

    ax3 = plt.subplot(4, 4, 3)
    plt.imshow(recon_pred_enhanced, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Predicted Reconstruction\n({len(pred_scatterers)} scatterers)", fontsize=12)
    plt.axis("off")

    # 添加残差图可视化
    ax4 = plt.subplot(4, 4, 4)
    if gt_scatterers and pred_scatterers:
        residual = np.abs(recon_gt_img) - np.abs(recon_pred_img)
        residual_normalized = residual / vmax
        im_residual = plt.imshow(residual_normalized, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        plt.title("Reconstruction Residual\n(GT - Predicted)", fontsize=12)
        plt.axis("off")
        plt.colorbar(im_residual, ax=ax4, shrink=0.8)

        # 添加误差统计
        mse = np.mean(residual**2)
        mae = np.mean(np.abs(residual))
        plt.text(
            0.02,
            0.98,
            f"MSE: {mse:.4f}\nMAE: {mae:.4f}",
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )
    else:
        plt.text(0.5, 0.5, "Residual\nNot Available", ha="center", va="center", transform=ax4.transAxes)
        plt.title("Reconstruction Residual", fontsize=12)

    # 第二行：5个参数图，但只用4个位置，将第5个放到下一行
    param_names = ["Heatmap", "Amplitude", "Alpha", "X-offset"]
    colormaps = ["hot", "viridis_r", "RdBu", "RdBu_r"]

    for i in range(4):
        ax = plt.subplot(4, 4, 5 + i)
        param_map = predicted_maps[i]
        im = plt.imshow(param_map, cmap=colormaps[i])
        plt.title(f"{param_names[i]}", fontsize=12)
        plt.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 第三行：Y-offset参数图和其他分析
    ax9 = plt.subplot(4, 4, 9)
    param_map = predicted_maps[4]  # Y-offset
    im = plt.imshow(param_map, cmap="RdBu_r")
    plt.title("Y-offset", fontsize=12)
    plt.axis("off")
    plt.colorbar(im, ax=ax9, shrink=0.8)

    # 检测结果图
    ax10 = plt.subplot(4, 4, 10)
    plt.imshow(original_array, cmap="gray")

    # 在原图上标注检测到的散射中心
    if pred_scatterers:
        for i, scatterer in enumerate(pred_scatterers[:10]):  # 只显示前10个
            x_pixel = int(scatterer["x"] * 32 + 64)
            y_pixel = int(64 - scatterer["y"] * 32)

            if 0 <= x_pixel < 128 and 0 <= y_pixel < 128:
                circle = patches.Circle((x_pixel, y_pixel), radius=3, color="red", fill=False, linewidth=2)
                ax10.add_patch(circle)
                plt.text(x_pixel + 4, y_pixel - 4, f"{i+1}", color="red", fontsize=8, weight="bold")

    plt.title("Detected Scatterers", fontsize=12)
    plt.axis("off")

    # 参数分布散点图
    ax11 = plt.subplot(4, 4, 11)
    if pred_scatterers:
        amplitudes = [s["A"] for s in pred_scatterers]
        alphas = [s["alpha"] for s in pred_scatterers]
        plt.scatter(alphas, amplitudes, c="blue", alpha=0.7, s=50)
        plt.xlabel("Alpha (α)", fontsize=10)
        plt.ylabel("Amplitude (A)", fontsize=10)
        plt.title("Parameter Distribution", fontsize=12)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No scatterers\ndetected", ha="center", va="center", transform=ax11.transAxes)
        plt.title("Parameter Distribution", fontsize=12)

    # 重建质量对比图
    ax12 = plt.subplot(4, 4, 12)
    if gt_scatterers and pred_scatterers:
        # 创建重建质量对比的直方图
        gt_intensity = np.abs(recon_gt_img).flatten()
        pred_intensity = np.abs(recon_pred_img).flatten()

        # 移除零值以更好地显示分布
        gt_nonzero = gt_intensity[gt_intensity > np.percentile(gt_intensity, 5)]
        pred_nonzero = pred_intensity[pred_intensity > np.percentile(pred_intensity, 5)]

        plt.hist(gt_nonzero, bins=50, alpha=0.5, label="GT", color="blue", density=True)
        plt.hist(pred_nonzero, bins=50, alpha=0.5, label="Predicted", color="red", density=True)
        plt.xlabel("Intensity", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.title("Intensity Distribution", fontsize=12)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Distribution\nNot Available", ha="center", va="center", transform=ax12.transAxes)
        plt.title("Intensity Distribution", fontsize=12)

    # 调整布局
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    return fig, {
        "base_name": base_name,
        "pred_scatterers": pred_scatterers,
        "gt_scatterers": gt_scatterers,
        "prediction_maps": predicted_maps,
    }


def main():
    print("Loading the trained ASCE_Net model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # 创建输出目录
    output_dir = os.path.join(project_root, "datasets", "result_vis", "comprehensive_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 加载数据集
    print("Discovering dataset samples...")
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("No valid samples found.")
        return
    print(f"Found {len(dataset.samples)} samples to visualize.")

    # 选择要可视化的样本数量（避免生成过多图片）
    max_samples = min(10, len(dataset.samples))
    selected_samples = dataset.samples[:max_samples]

    print(f"Processing {max_samples} samples for comprehensive visualization...")

    results_summary = []

    for sample_info in tqdm(selected_samples, desc="Generating Comprehensive Visualizations"):
        try:
            result = create_comprehensive_visualization(sample_info, model)

            if result is not None:
                fig, info = result

                # 保存图片
                save_path = os.path.join(output_dir, f"comprehensive_{info['base_name']}.png")
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # 收集统计信息
                results_summary.append(
                    {
                        "name": info["base_name"],
                        "pred_count": len(info["pred_scatterers"]),
                        "gt_count": len(info["gt_scatterers"]),
                    }
                )

                print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # 生成总结报告
    if results_summary:
        print("\n" + "=" * 60)
        print("COMPREHENSIVE VISUALIZATION SUMMARY")
        print("=" * 60)
        print(f"{'Sample Name':<30} {'Predicted':<10} {'Ground Truth':<12}")
        print("-" * 60)

        total_pred = 0
        total_gt = 0

        for result in results_summary:
            print(f"{result['name']:<30} {result['pred_count']:<10} {result['gt_count']:<12}")
            total_pred += result["pred_count"]
            total_gt += result["gt_count"]

        print("-" * 60)
        print(f"{'TOTAL':<30} {total_pred:<10} {total_gt:<12}")
        print(f"{'AVERAGE':<30} {total_pred/len(results_summary):<10.1f} {total_gt/len(results_summary):<12.1f}")
        print("=" * 60)

    print("\nComprehensive visualization complete!")
    print(f"All results saved in: {output_dir}")
    print(f"Generated visualizations for {len(results_summary)} samples.")


if __name__ == "__main__":
    main()
