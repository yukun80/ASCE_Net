#!/usr/bin/env python
# script/diagnose_metrics.py - 诊断定量指标计算问题

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model.asc_net import ASCNet_v3_5param
from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset, read_sar_complex_tensor
from utils.reconstruction import reconstruct_sar_image, pixel_to_model
from script.quantitative_analysis import QuantitativeAnalyzer


class MetricsDiagnosis:
    """定量指标诊断工具"""

    def __init__(self):
        self.analyzer = QuantitativeAnalyzer()

    def diagnose_single_sample(self, sar_path, model):
        """深度诊断单个样本"""
        base_name = os.path.basename(sar_path).replace(".raw", "")
        print(f"\n🔬 深度诊断样本: {base_name}")
        print("=" * 60)

        # 1. 加载原始SAR图像
        original_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
        if original_tensor is None:
            print("❌ 无法加载原始SAR图像")
            return

        original_complex = original_tensor.numpy()
        original_magnitude = np.abs(original_complex)

        print(f"📊 原始图像分析:")
        print(f"  形状: {original_complex.shape}")
        print(f"  幅度范围: {original_magnitude.min():.6f} ~ {original_magnitude.max():.6f}")
        print(f"  平均幅度: {original_magnitude.mean():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(original_magnitude**2)):.6f}")

        # 2. 模型预测
        with torch.no_grad():
            input_tensor = original_tensor.unsqueeze(0).to(config.DEVICE)
            predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

        print(f"\n🤖 模型预测分析:")
        for i, name in enumerate(["heatmap", "A_log1p", "alpha", "dx", "dy"]):
            pred_map = predicted_maps[i]
            print(f"  {name}: 范围[{pred_map.min():.3f}, {pred_map.max():.3f}], 均值{pred_map.mean():.3f}")

        # 3. 提取散射中心 (Aggressive模式)
        predicted_scatterers = self.analyzer.extract_scatterers_aggressive(predicted_maps)
        print(f"\n🎯 散射中心提取 (Aggressive):")
        print(f"  检出数量: {len(predicted_scatterers)}")

        if predicted_scatterers:
            amplitudes = [s["A"] for s in predicted_scatterers]
            print(f"  幅度范围: {min(amplitudes):.4f} ~ {max(amplitudes):.4f}")
            print(f"  平均幅度: {np.mean(amplitudes):.4f}")

        # 4. 重建SAR图像
        reconstructed_complex = reconstruct_sar_image(predicted_scatterers)
        reconstructed_magnitude = np.abs(reconstructed_complex)

        print(f"\n🔧 重建图像分析:")
        print(f"  形状: {reconstructed_complex.shape}")
        print(f"  幅度范围: {reconstructed_magnitude.min():.6f} ~ {reconstructed_magnitude.max():.6f}")
        print(f"  平均幅度: {reconstructed_magnitude.mean():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(reconstructed_magnitude**2)):.6f}")

        # 5. MSE计算诊断
        print(f"\n📐 MSE计算诊断:")

        # 方法1：当前使用的相对MSE
        original_norm = np.linalg.norm(original_complex)
        diff_norm = np.linalg.norm(reconstructed_complex - original_complex)
        current_mse = diff_norm / original_norm
        print(f"  当前方法 (相对RMSE): {current_mse:.6f}")

        # 方法2：传统MSE
        traditional_mse = np.mean(np.abs(reconstructed_complex - original_complex) ** 2)
        print(f"  传统MSE: {traditional_mse:.6f}")

        # 方法3：归一化MSE
        normalized_mse = traditional_mse / np.mean(np.abs(original_complex) ** 2)
        print(f"  归一化MSE: {normalized_mse:.6f}")

        # 方法4：对数域MSE
        log_original = np.log1p(original_magnitude)
        log_reconstructed = np.log1p(reconstructed_magnitude)
        log_mse = np.mean((log_reconstructed - log_original) ** 2)
        print(f"  对数域MSE: {log_mse:.6f}")

        # 6. ENT计算诊断
        print(f"\n🌊 ENT计算诊断:")

        # 当前方法：Energy归一化
        g = np.sqrt(np.sum(reconstructed_magnitude**2))
        print(f"  Energy归一化因子 g: {g:.6f}")

        if g > 1e-10:
            normalized_magnitude = reconstructed_magnitude / g
            entropy_values = normalized_magnitude**2 * np.log(normalized_magnitude**2 + 1e-12)
            current_ent = -np.sum(entropy_values)
            print(f"  当前ENT (Energy归一化): {current_ent:.6f}")
        else:
            print(f"  当前ENT: -inf (能量太小)")

        # 替代方法1：直接归一化
        if reconstructed_magnitude.max() > 0:
            direct_normalized = reconstructed_magnitude / reconstructed_magnitude.max()
            prob = direct_normalized**2 / np.sum(direct_normalized**2)
            prob = prob[prob > 1e-12]  # 避免log(0)
            direct_ent = -np.sum(prob * np.log(prob))
            print(f"  直接归一化ENT: {direct_ent:.6f}")

        # 替代方法2：传统图像熵
        magnitude_uint8 = ((reconstructed_magnitude / reconstructed_magnitude.max()) * 255).astype(np.uint8)
        hist, _ = np.histogram(magnitude_uint8, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        traditional_ent = -np.sum(hist * np.log2(hist))
        print(f"  传统图像熵: {traditional_ent:.6f}")

        # 7. GT对比（如果存在）
        mat_path = self.analyzer.find_corresponding_mat_file(sar_path)
        if os.path.exists(mat_path):
            print(f"\n📋 GT对比分析:")
            gt_scatterers = self.analyzer.extract_scatterers_from_mat_fixed(mat_path)

            if gt_scatterers:
                gt_reconstructed = reconstruct_sar_image(gt_scatterers)
                gt_magnitude = np.abs(gt_reconstructed)

                print(f"  GT散射中心数量: {len(gt_scatterers)}")
                print(f"  GT重建图像幅度范围: {gt_magnitude.min():.6f} ~ {gt_magnitude.max():.6f}")

                # GT vs 原始图像的MSE
                gt_vs_original = np.linalg.norm(gt_reconstructed - original_complex) / np.linalg.norm(original_complex)
                print(f"  GT vs 原始图像 MSE: {gt_vs_original:.6f}")

                # 预测 vs GT的MSE
                pred_vs_gt = np.linalg.norm(reconstructed_complex - gt_reconstructed) / np.linalg.norm(gt_reconstructed)
                print(f"  预测 vs GT MSE: {pred_vs_gt:.6f}")

        # 8. 可视化对比
        self.create_diagnostic_visualization(original_complex, reconstructed_complex, base_name, predicted_scatterers)

        print(f"\n✅ 诊断完成: {base_name}")

    def create_diagnostic_visualization(self, original, reconstructed, sample_name, scatterers):
        """创建诊断可视化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Metrics Diagnosis: {sample_name}", fontsize=14)

        original_mag = np.abs(original)
        reconstructed_mag = np.abs(reconstructed)
        difference = reconstructed_mag - original_mag

        # 原始图像
        im1 = axes[0, 0].imshow(original_mag, cmap="gray")
        axes[0, 0].set_title("Original SAR")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 重建图像
        im2 = axes[0, 1].imshow(reconstructed_mag, cmap="gray")
        axes[0, 1].set_title("Reconstructed SAR")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        # 差异图
        im3 = axes[0, 2].imshow(difference, cmap="seismic", vmin=-difference.max(), vmax=difference.max())
        axes[0, 2].set_title("Difference (Recon - Original)")
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

        # 散射中心位置
        axes[1, 0].imshow(original_mag, cmap="gray")
        if scatterers:
            for s in scatterers:
                # 将物理坐标转换回像素坐标进行显示
                pixel_x = (s["x"] * 128 / (0.3 * 84)) + 65
                pixel_y = 65 - (s["y"] * 128 / (0.3 * 84))
                axes[1, 0].plot(pixel_x, pixel_y, "r+", markersize=8, markeredgewidth=2)
        axes[1, 0].set_title(f"Scatterers ({len(scatterers)})")
        axes[1, 0].axis("off")

        # 幅度直方图对比
        axes[1, 1].hist(original_mag.flatten(), bins=50, alpha=0.7, label="Original", density=True)
        axes[1, 1].hist(reconstructed_mag.flatten(), bins=50, alpha=0.7, label="Reconstructed", density=True)
        axes[1, 1].set_title("Magnitude Histograms")
        axes[1, 1].legend()
        axes[1, 1].set_xlabel("Magnitude")
        axes[1, 1].set_ylabel("Density")

        # 误差分析
        error_stats = [
            f"Max Original: {original_mag.max():.4f}",
            f"Max Reconstructed: {reconstructed_mag.max():.4f}",
            f"Mean Original: {original_mag.mean():.4f}",
            f"Mean Reconstructed: {reconstructed_mag.mean():.4f}",
            f"RMS Difference: {np.sqrt(np.mean(difference**2)):.4f}",
        ]

        axes[1, 2].text(
            0.05,
            0.95,
            "\n".join(error_stats),
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 2].set_title("Error Statistics")
        axes[1, 2].axis("off")

        plt.tight_layout()

        # 保存
        output_dir = os.path.join(project_root, "datasets", "result_vis", "diagnosis")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"diagnosis_{sample_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    💾 诊断图保存: {save_path}")


def main():
    print("🔬 定量指标计算诊断工具")
    print("=" * 50)

    # 加载模型
    print("正在加载ASC-Net模型...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print("✅ 模型加载成功")

    # 加载数据集
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("❌ 未找到数据集样本")
        return

    print(f"✅ 找到 {len(dataset.samples)} 个样本")

    # 诊断工具
    diagnoser = MetricsDiagnosis()

    # 诊断前5个样本
    print(f"\n🔍 开始诊断前5个样本...")

    for i, sample_info in enumerate(dataset.samples[:5]):
        try:
            sar_path = sample_info["sar"]
            diagnoser.diagnose_single_sample(sar_path, model)
        except Exception as e:
            print(f"❌ 诊断样本 {i+1} 时出错: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n🎉 诊断完成！")
    print(f"📁 结果保存在: datasets/result_vis/diagnosis/")


if __name__ == "__main__":
    main()
