#!/usr/bin/env python3
# script/visualize_scatterer_overlay.py - 散射中心叠加可视化工具

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
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
    pixel_to_model,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# --- 定义像素间距常量 ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
P_GRID_SIZE = 84
PIXEL_SPACING = 0.1


def model_to_pixel(x_model, y_model):
    """
    将物理模型坐标转换为像素坐标（pixel_to_model的逆函数）

    Parameters:
    -----------
    x_model, y_model : float
        物理坐标（米）

    Returns:
    --------
    row, col : float
        像素坐标
    """
    C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
    C2 = IMG_WIDTH / 2  # 64.0

    # 逆变换
    row = C2 - y_model * C1
    col = C2 + x_model * C1

    return row, col


def find_corresponding_jpg_file(sar_path):
    """查找对应的JPG文件"""
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


def find_corresponding_mat_file(sar_path):
    """查找对应的MAT文件"""
    mat_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")
    base_name = os.path.basename(sar_path).replace(".raw", "")
    return os.path.join(mat_root, f"{base_name}.mat")


class ScattererVisualizationProcessor:
    """散射中心叠加可视化处理器"""

    def __init__(self):
        # 检测配置
        self.detection_configs = {
            "conservative": {
                "low_threshold": 0.3,
                "high_threshold": 0.6,
                "min_distance": 5,
                "amplitude_weight_low": 0.8,
            },
            "moderate": {
                "low_threshold": 0.1,
                "high_threshold": 0.5,
                "min_distance": 3,
                "amplitude_weight_low": 0.5,
            },
            "aggressive": {
                "low_threshold": 0.05,
                "high_threshold": 0.3,
                "min_distance": 2,
                "amplitude_weight_low": 0.3,
            },
        }

        # 可视化样式配置 - 统一使用红点
        self.marker_styles = {
            "gt": {
                "color": "red",
                "marker": "o",
                "size": 60,
                "alpha": 0.8,
                "edgecolor": "darkred",
                "linewidth": 1.5,
                "label": "Ground Truth",
            },
            "pred_high": {
                "color": "red",
                "marker": "o",
                "size": 60,
                "alpha": 0.8,
                "edgecolor": "darkred",
                "linewidth": 1.5,
                "label": "High Confidence",
            },
            "pred_low": {
                "color": "red",
                "marker": "o",
                "size": 60,
                "alpha": 0.8,
                "edgecolor": "darkred",
                "linewidth": 1.5,
                "label": "Low Confidence",
            },
        }

    def enhanced_extract_scatterers(self, prediction_maps, config_name="moderate"):
        """增强版散射中心提取"""
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
                    "pixel_row": r,
                    "pixel_col": c,
                    "confidence": heatmap[r, c],
                }
            )

        return scatterers

    def create_overlay_visualization(
        self, sar_image, gt_scatterers, pred_scatterers, config_name="moderate", show_heatmap=True, prediction_maps=None
    ):
        """创建散射中心叠加可视化"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Scatterer Detection Overlay Visualization ({config_name.title()} Config)", fontsize=16, y=0.95)

        # 转换为numpy数组以便处理 - 确保正确的灰度显示
        if isinstance(sar_image, Image.Image):
            sar_array = np.array(sar_image)  # 保持原始数组格式
        else:
            sar_array = sar_image

        # 1. 原始SAR图像 + GT散射中心
        ax1 = axes[0]
        ax1.imshow(sar_array, cmap="gray", alpha=0.8)  # 使用灰度colormap确保正确显示

        # 绘制GT散射中心
        for scatterer in gt_scatterers:
            row, col = model_to_pixel(scatterer["x"], scatterer["y"])
            if 0 <= row < IMG_HEIGHT and 0 <= col < IMG_WIDTH:
                style = self.marker_styles["gt"]
                ax1.scatter(
                    col,
                    row,
                    c=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=style["alpha"],
                    edgecolors=style["edgecolor"],
                    linewidths=style["linewidth"],
                )

        ax1.set_title(f"Ground Truth Scatterers\n({len(gt_scatterers)} detected)", fontsize=12)
        ax1.axis("off")

        # 2. 原始SAR图像 + 预测散射中心
        ax2 = axes[1]
        ax2.imshow(sar_array, cmap="gray", alpha=0.8)  # 使用灰度colormap确保正确显示

        # 分别绘制高低置信度散射中心
        high_conf = [s for s in pred_scatterers if s["strength"] == "high"]
        low_conf = [s for s in pred_scatterers if s["strength"] == "low"]

        for scatterer in high_conf:
            row, col = model_to_pixel(scatterer["x"], scatterer["y"])
            if 0 <= row < IMG_HEIGHT and 0 <= col < IMG_WIDTH:
                style = self.marker_styles["pred_high"]
                ax2.scatter(
                    col,
                    row,
                    c=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=style["alpha"],
                    edgecolors=style["edgecolor"],
                    linewidths=style["linewidth"],
                )

        for scatterer in low_conf:
            row, col = model_to_pixel(scatterer["x"], scatterer["y"])
            if 0 <= row < IMG_HEIGHT and 0 <= col < IMG_WIDTH:
                style = self.marker_styles["pred_low"]
                ax2.scatter(
                    col,
                    row,
                    c=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=style["alpha"],
                    edgecolors=style["edgecolor"],
                    linewidths=style["linewidth"],
                )

        ax2.set_title(f"Predicted Scatterers\n({len(pred_scatterers)} detected)", fontsize=12)
        ax2.axis("off")

        # 添加图例 - 简化为统一的红点标记
        legend_elements = [
            plt.scatter(
                [],
                [],
                c="red",
                marker="o",
                s=60,
                alpha=0.8,
                edgecolors="darkred",
                linewidths=1.5,
                label="Detected Scatterers",
            )
        ]

        fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        return fig

    def calculate_detection_metrics(self, gt_scatterers, pred_scatterers, distance_threshold=0.5):
        """计算检测性能指标"""
        # 配对算法：找到距离最近的GT和预测散射中心
        matched_pairs = []
        unmatched_gt = list(range(len(gt_scatterers)))
        unmatched_pred = list(range(len(pred_scatterers)))

        for i, gt_sc in enumerate(gt_scatterers):
            best_match_idx = None
            min_distance = float("inf")

            for j, pred_sc in enumerate(pred_scatterers):
                if j in unmatched_pred:
                    distance = np.sqrt((gt_sc["x"] - pred_sc["x"]) ** 2 + (gt_sc["y"] - pred_sc["y"]) ** 2)
                    if distance < min_distance and distance < distance_threshold:
                        min_distance = distance
                        best_match_idx = j

            if best_match_idx is not None:
                matched_pairs.append((i, best_match_idx, min_distance))
                unmatched_gt.remove(i)
                unmatched_pred.remove(best_match_idx)

        # 计算指标
        true_positives = len(matched_pairs)
        false_positives = len(unmatched_pred)
        false_negatives = len(unmatched_gt)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "matched_pairs": matched_pairs,
            "avg_distance": np.mean([pair[2] for pair in matched_pairs]) if matched_pairs else 0,
        }


def create_scatterer_overlay_visualization(sample_info, model, processor, output_dir, config_name="moderate"):
    """为单个样本创建散射中心叠加可视化"""
    sar_path = os.path.normpath(sample_info["sar"])
    base_name = os.path.basename(sar_path).replace(".raw", "")

    print(f"\n--- 处理样本: {base_name} ---")

    # 查找文件
    jpg_path = find_corresponding_jpg_file(sar_path)
    mat_path = find_corresponding_mat_file(sar_path)

    if not jpg_path or not os.path.exists(mat_path):
        print(f"  ✗ 文件缺失")
        return False, None

    print(f"  ✓ JPG: {os.path.basename(jpg_path)}")
    print(f"  ✓ MAT: {os.path.basename(mat_path)}")

    # 加载数据
    try:
        original_sar_image = Image.open(jpg_path)
        sar_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
        gt_scatterers = extract_scatterers_from_mat(mat_path)

        if sar_tensor is None or not gt_scatterers:
            print(f"  ✗ 数据加载失败")
            return False, None

        print(f"  ✓ GT散射中心: {len(gt_scatterers)} 个")

    except Exception as e:
        print(f"  ✗ 数据加载错误: {e}")
        return False, None

    # 模型预测
    print(f"  ✓ 开始ASC-Net模型推理...")
    with torch.no_grad():
        input_tensor = sar_tensor.unsqueeze(0).to(config.DEVICE)
        predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

    # 提取预测散射中心
    pred_scatterers = processor.enhanced_extract_scatterers(predicted_maps, config_name)
    print(f"  ✓ 预测散射中心: {len(pred_scatterers)} 个")

    # 计算检测指标
    metrics = processor.calculate_detection_metrics(gt_scatterers, pred_scatterers)
    print(f"  📊 检测性能: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

    # 创建可视化
    fig = processor.create_overlay_visualization(
        original_sar_image,
        gt_scatterers,
        pred_scatterers,
        config_name,
        show_heatmap=True,
        prediction_maps=predicted_maps,
    )

    # 保存结果
    save_path = os.path.join(output_dir, f"scatterer_overlay_{config_name}_{base_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ 保存成功: {os.path.basename(save_path)}")

    return True, metrics


def main():
    print("=== 散射中心叠加可视化工具 (简化版) ===")
    print("功能:")
    print("1. 在原始SAR图像上叠加GT和预测散射中心")
    print("2. 提供直观的检测效果对比")
    print("3. 计算检测性能指标")
    print("4. 简化版: 只显示GT和预测两个子图")

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
    processor = ScattererVisualizationProcessor()

    # 设置输出目录
    output_dir = os.path.join(project_root, "datasets", "result_vis", "scatterer_overlay_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 结果保存目录: {output_dir}")

    # 加载数据集
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("✗ 未找到有效样本")
        return

    print(f"✓ 找到 {len(dataset.samples)} 个样本")

    # 选择处理配置
    configs_to_test = ["conservative", "moderate", "aggressive"]

    # 处理样本
    num_samples = min(500, len(dataset.samples))  # 处理前50个样本

    print(f"\n=== 开始处理 {num_samples} 个样本，{len(configs_to_test)} 种配置 ===")

    all_metrics = {config: [] for config in configs_to_test}
    successful_samples = 0

    for i, sample_info in enumerate(tqdm(dataset.samples[:num_samples], desc="处理样本")):
        sample_success = False

        for config_name in configs_to_test:
            try:
                success, metrics = create_scatterer_overlay_visualization(
                    sample_info, model, processor, output_dir, config_name
                )
                if success:
                    all_metrics[config_name].append(metrics)
                    sample_success = True

            except Exception as e:
                print(f"  ✗ {config_name}配置处理错误: {e}")
                continue

        if sample_success:
            successful_samples += 1

    # 计算平均性能指标
    print(f"\n=== 性能统计 ===")
    print(f"成功处理: {successful_samples}/{num_samples} 个样本")

    for config_name in configs_to_test:
        metrics_list = all_metrics[config_name]
        if metrics_list:
            avg_precision = np.mean([m["precision"] for m in metrics_list])
            avg_recall = np.mean([m["recall"] for m in metrics_list])
            avg_f1 = np.mean([m["f1_score"] for m in metrics_list])
            avg_distance = np.mean([m["avg_distance"] for m in metrics_list])

            print(f"\n{config_name.title()} 配置平均性能:")
            print(f"  Precision: {avg_precision:.3f}")
            print(f"  Recall: {avg_recall:.3f}")
            print(f"  F1-Score: {avg_f1:.3f}")
            print(f"  Avg Match Distance: {avg_distance:.3f}m")

    if successful_samples > 0:
        print(f"\n🎉 散射中心叠加可视化完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"\n✅ 可视化内容:")
        print("  • 左：原始SAR + GT散射中心（红色圆点）")
        print("  • 右：原始SAR + 预测散射中心（红色圆点）")
        print(f"\n📊 生成了 {len(configs_to_test)} 种检测配置的对比结果")
    else:
        print("\n⚠️ 未能成功处理任何样本")


if __name__ == "__main__":
    main()
