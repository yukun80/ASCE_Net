# script/run_enhanced_visualization_fixed_layout.py - 修复布局问题版本

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import maximum_filter
import cv2

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
    pixel_to_model,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# --- 定义像素间距常量 ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
P_GRID_SIZE = 84
C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
PIXEL_SPACING = 0.1


def find_corresponding_jpg_file_final(sar_path):
    """查找JPG文件"""
    jpg_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Data_Processed_jpg")
    base_name = os.path.basename(sar_path).replace(".raw", "")

    possible_files = [
        os.path.join(jpg_root, f"{base_name}_v2.JPG"),
        os.path.join(jpg_root, f"{base_name}_v2.jpg"),
        os.path.join(jpg_root, f"{base_name}_v1.JPG"),
        os.path.join(jpg_root, f"{base_name}_v1.jpg"),
    ]

    for jpg_path in possible_files:
        if os.path.exists(jpg_path):
            return jpg_path
    return None


def find_corresponding_mat_file_final(sar_path):
    """查找MAT文件"""
    mat_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")
    base_name = os.path.basename(sar_path).replace(".raw", "")
    return os.path.join(mat_root, f"{base_name}.mat")


class EnhancedVisualizationProcessor:
    """增强版SAR可视化处理器"""

    def __init__(self):
        # 检测配置
        self.detection_configs = {
            "conservative": {
                "low_threshold": 0.3,
                "high_threshold": 0.6,
                "min_distance": 5,
                "amplitude_weight_low": 0.8,
            },
            "moderate": {"low_threshold": 0.1, "high_threshold": 0.5, "min_distance": 3, "amplitude_weight_low": 0.5},
            "aggressive": {
                "low_threshold": 0.05,
                "high_threshold": 0.3,
                "min_distance": 2,
                "amplitude_weight_low": 0.3,
            },
        }

        # 增强配置
        self.enhancement_configs = {
            "conservative": {"amplitude_boost": 1.5, "weak_boost": 2.0, "contrast_method": "gamma", "gamma": 0.7},
            "moderate": {"amplitude_boost": 2.5, "weak_boost": 3.5, "contrast_method": "adaptive", "gamma": 0.5},
            "aggressive": {"amplitude_boost": 4.0, "weak_boost": 6.0, "contrast_method": "log", "gamma": 0.3},
        }

    def enhanced_extract_scatterers_fixed(self, prediction_maps, config_name="moderate"):
        """修复版散射中心提取函数"""
        config_params = self.detection_configs[config_name]

        heatmap, pred_A_log1p, pred_alpha, pred_dx, pred_dy = [prediction_maps[i] for i in range(5)]
        pred_A = np.expm1(pred_A_log1p)

        # 非最大值抑制
        neighborhood_size = 5
        local_max = maximum_filter(heatmap, size=neighborhood_size) == heatmap

        # 高阈值检测
        high_candidates = (heatmap > config_params["high_threshold"]) & local_max
        high_peak_coords = np.argwhere(high_candidates)

        # 低阈值检测
        low_candidates = (heatmap > config_params["low_threshold"]) & local_max
        low_peak_coords = np.argwhere(low_candidates)

        # 合并结果，避免重复
        all_coords = []
        for coord in high_peak_coords:
            all_coords.append((coord[0], coord[1], "high"))

        for coord in low_peak_coords:
            is_duplicate = False
            for existing_coord in high_peak_coords:
                if np.linalg.norm(coord - existing_coord) < config_params["min_distance"]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_coords.append((coord[0], coord[1], "low"))

        scatterers = []
        for r, c, strength in all_coords:
            amplitude_weight = 1.0 if strength == "high" else config_params["amplitude_weight_low"]

            x_base, y_base = pixel_to_model(r, c)
            dx = pred_dx[r, c] * PIXEL_SPACING
            dy = pred_dy[r, c] * PIXEL_SPACING
            x_final = x_base + dx
            y_final = y_base - dy

            A_final = pred_A[r, c] * amplitude_weight
            alpha_final = pred_alpha[r, c]

            scatterers.append(
                {
                    "A": A_final,
                    "alpha": alpha_final,
                    "x": x_final,
                    "y": y_final,
                    "gamma": 0,
                    "L": 0,
                    "phi_prime": 0,
                    "strength": strength,
                }
            )

        return scatterers

    def enhanced_reconstruct(self, scatterers, config_name="moderate"):
        """增强重建"""
        config_params = self.enhancement_configs[config_name]

        boosted_scatterers = []
        for s in scatterers:
            s_copy = s.copy()
            boost = config_params["amplitude_boost"]
            if s.get("strength") == "low":
                boost *= config_params["weak_boost"] / config_params["amplitude_boost"]
            s_copy["A"] = s["A"] * boost
            boosted_scatterers.append(s_copy)

        return reconstruct_sar_image(boosted_scatterers)

    def apply_contrast_enhancement(self, img_complex, config_name="moderate"):
        """应用对比度增强"""
        config_params = self.enhancement_configs[config_name]
        img_magnitude = np.abs(img_complex)

        if config_params["contrast_method"] == "adaptive":
            img_uint8 = ((img_magnitude / img_magnitude.max()) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
            return enhanced.astype(np.float32) / 255.0 * img_magnitude.max()

        elif config_params["contrast_method"] == "gamma":
            gamma = config_params["gamma"]
            return np.power(img_magnitude / img_magnitude.max(), gamma) * img_magnitude.max()

        elif config_params["contrast_method"] == "log":
            return np.log1p(img_magnitude * 10) / np.log1p(10)

        return img_magnitude


def create_enhanced_comparison_fixed_layout(sample_info, model, processor, output_dir):
    """创建修复布局问题的增强对比可视化"""
    sar_path = os.path.normpath(sample_info["sar"])
    base_name = os.path.basename(sar_path).replace(".raw", "")

    print(f"\n--- 处理样本: {base_name} ---")

    # 查找文件
    jpg_path = find_corresponding_jpg_file_final(sar_path)
    mat_path = find_corresponding_mat_file_final(sar_path)

    if not jpg_path or not os.path.exists(mat_path):
        print(f"  ✗ 文件缺失")
        return False

    print(f"  ✓ JPG: {os.path.basename(jpg_path)}")
    print(f"  ✓ MAT: {os.path.basename(mat_path)}")

    # 加载数据
    try:
        original_sar_image = Image.open(jpg_path)
        sar_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
        gt_scatterers = extract_scatterers_from_mat(mat_path)

        if sar_tensor is None or not gt_scatterers:
            print(f"  ✗ 数据加载失败")
            return False

        print(f"  ✓ GT散射中心: {len(gt_scatterers)} 个")

    except Exception as e:
        print(f"  ✗ 数据加载错误: {e}")
        return False

    # GT重建
    recon_gt = reconstruct_sar_image(gt_scatterers)

    # 模型预测 - 确保heatmap来自ASC-Net模型
    print(f"  ✓ 开始ASC-Net模型推理...")
    with torch.no_grad():
        input_tensor = sar_tensor.unsqueeze(0).to(config.DEVICE)
        predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

    # 确认predicted_maps[0]是模型生成的heatmap
    model_heatmap = predicted_maps[0]  # 这是ASC-Net模型预测的散射中心概率热力图
    print(f"  ✓ 模型预测热力图范围: [{model_heatmap.min():.3f}, {model_heatmap.max():.3f}]")

    # 不同配置的结果
    configs = ["conservative", "moderate", "aggressive"]  # 确保包含所有三个配置
    results = {}

    for cfg in configs:
        scatterers = processor.enhanced_extract_scatterers_fixed(predicted_maps, cfg)
        recon = processor.enhanced_reconstruct(scatterers, cfg)
        enhanced = processor.apply_contrast_enhancement(recon, cfg)

        results[cfg] = {
            "scatterers": scatterers,
            "reconstruction": recon,
            "enhanced": enhanced,
            "count": len(scatterers),
        }
        print(f"  ✓ {cfg.title()}: {len(scatterers)} 个散射中心")

    # 修复布局问题：使用更合理的subplot布局
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"Enhanced SAR Reconstruction: {base_name}", fontsize=18, y=0.95)

    # 创建网格布局：3行5列
    gs = fig.add_gridspec(
        3, 5, height_ratios=[1, 1, 0.3], width_ratios=[1, 1, 1, 1, 1], hspace=0.25, wspace=0.15, top=0.9, bottom=0.25
    )

    # 第一行：原始图像 + GT + 三种配置的重建
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_sar_image, cmap="gray")
    ax1.set_title("Original SAR\n(v2 JPG)", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    gt_vmax = np.percentile(np.abs(recon_gt), 99.5)
    ax2.imshow(np.abs(recon_gt), cmap="gray", vmin=0, vmax=gt_vmax)
    ax2.set_title(f"GT Reconstruction\n({len(gt_scatterers)} scatterers)", fontsize=12)
    ax2.axis("off")

    # 确保显示所有三个配置
    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[0, i + 2])
        vmax = np.percentile(np.abs(results[cfg]["reconstruction"]), 99.5)
        ax.imshow(np.abs(results[cfg]["reconstruction"]), cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"{cfg.title()}\n({results[cfg]['count']} scatterers)", fontsize=12)
        ax.axis("off")

    # 第二行：模型预测热力图 + 三种配置的增强图像
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im_heatmap = ax_heatmap.imshow(model_heatmap, cmap="hot", interpolation="bilinear")
    ax_heatmap.set_title("ASC-Net Prediction\nHeatmap", fontsize=12)
    ax_heatmap.axis("off")
    plt.colorbar(im_heatmap, ax=ax_heatmap, fraction=0.046, pad=0.04)

    # 显示GT重建的增强版本
    ax_gt_enhanced = fig.add_subplot(gs[1, 1])
    # 对GT重建应用adaptive增强
    gt_magnitude = np.abs(recon_gt)
    gt_uint8 = ((gt_magnitude / gt_magnitude.max()) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gt_enhanced = clahe.apply(gt_uint8)
    gt_enhanced = gt_enhanced.astype(np.float32) / 255.0 * gt_magnitude.max()

    ax_gt_enhanced.imshow(gt_enhanced, cmap="gray")
    ax_gt_enhanced.set_title("Enhanced GT", fontsize=12)
    ax_gt_enhanced.axis("off")

    # 显示三种配置的增强图像
    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[1, i + 2])
        ax.imshow(results[cfg]["enhanced"], cmap="gray")
        ax.set_title(f"Enhanced {cfg.title()}", fontsize=12)
        ax.axis("off")

    # 第三行：配置参数表格（不重叠）
    ax_config = fig.add_subplot(gs[2, :])
    ax_config.axis("off")

    # 创建配置参数表格
    config_data = []
    headers = [
        "Configuration",
        "Low Threshold",
        "High Threshold",
        "Amplitude Boost",
        "Weak Boost",
        "Method",
        "Detected",
    ]

    for cfg in configs:
        det_cfg = processor.detection_configs[cfg]
        enh_cfg = processor.enhancement_configs[cfg]
        config_data.append(
            [
                cfg.title(),
                f"{det_cfg['low_threshold']:.2f}",
                f"{det_cfg['high_threshold']:.2f}",
                f"{enh_cfg['amplitude_boost']:.1f}x",
                f"{enh_cfg['weak_boost']:.1f}x",
                enh_cfg["contrast_method"],
                f"{results[cfg]['count']}",
            ]
        )

    # 创建表格
    table = ax_config.table(
        cellText=config_data, colLabels=headers, cellLoc="center", loc="center", bbox=[0.1, 0.1, 0.8, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(config_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f1f1f2")

    # 保存结果
    save_path = os.path.join(output_dir, f"enhanced_comparison_{base_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ 保存成功: {os.path.basename(save_path)}")

    # 输出详细对比
    print(f"  📊 ASC-Net模型效果对比:")
    for cfg in configs:
        improvement = results[cfg]["count"] / len(gt_scatterers) * 100
        print(f"     {cfg.title()}: {results[cfg]['count']}/{len(gt_scatterers)} " f"({improvement:.1f}% 检出率)")

    return True


def main():
    print("=== SAR重建可视化增强工具 (修复布局版) ===")
    print("修复内容:")
    print("1. 解决文本重叠问题")
    print("2. 确保显示所有三个配置")
    print("3. 确认heatmap来自ASC-Net模型")

    # 加载模型
    print("\n正在加载ASC-Net模型...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print("✓ ASC-Net模型加载成功")

    # 初始化处理器
    processor = EnhancedVisualizationProcessor()

    # 设置输出目录
    output_dir = os.path.join(project_root, "datasets", "result_vis", "comprehensive_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 结果保存目录: {output_dir}")

    # 加载数据集
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("✗ 未找到有效样本")
        return

    print(f"✓ 找到 {len(dataset.samples)} 个样本")

    # 处理样本
    num_samples = min(2000, len(dataset.samples))
    successful_samples = 0

    print(f"\n=== 开始处理 {num_samples} 个样本 ===")

    for i, sample_info in enumerate(dataset.samples[:num_samples]):
        try:
            if create_enhanced_comparison_fixed_layout(sample_info, model, processor, output_dir):
                successful_samples += 1
        except Exception as e:
            print(f"  ✗ 处理错误: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 总结
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {successful_samples}/{num_samples} 个样本")

    if successful_samples > 0:
        print(f"\n🎉 成功生成修复布局的对比图！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"\n✅ 修复内容确认:")
        print("  • 布局问题：使用网格布局，避免文本重叠")
        print("  • 显示完整：包含Conservative、Moderate、Aggressive三种配置")
        print("  • Heatmap来源：确认来自ASC-Net模型预测")
        print("  • 参数表格：清晰显示各配置参数")

        print(f"\n📊 可视化说明:")
        print("  • 第一行：原始图像、GT重建、三种配置重建")
        print("  • 第二行：ASC-Net预测热力图、三种配置增强图像")
        print("  • 第三行：详细参数配置表格")
    else:
        print("\n⚠️ 未能成功处理任何样本")


if __name__ == "__main__":
    main()
