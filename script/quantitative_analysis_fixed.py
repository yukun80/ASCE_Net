#!/usr/bin/env python
# script/quantitative_analysis_fixed.py - 修复版定量分析工具

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
from tqdm import tqdm
from scipy.ndimage import maximum_filter

# Setup Project Path
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

# 设置字体以避免警告
matplotlib.rc("font", family=["SimHei", "Microsoft YaHei", "DejaVu Sans"])
matplotlib.rc("axes", unicode_minus=False)


class QuantitativeAnalyzerFixed:
    """修复版定量分析器"""

    def __init__(self):
        # Aggressive模式检测配置 - 根据用户要求
        self.detection_config = {
            "low_threshold": 0.05,
            "high_threshold": 0.3,
            "min_distance": 2,
            "amplitude_weight_low": 0.3,
        }

        # 像素间距常量
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.P_GRID_SIZE = 84
        self.C1 = self.IMG_HEIGHT / (0.3 * self.P_GRID_SIZE)
        self.PIXEL_SPACING = 0.1

        print("🔧 定量分析器配置 (修复版):")
        print(f"  检测模式: Aggressive")
        print(f"  低阈值: {self.detection_config['low_threshold']}")
        print(f"  高阈值: {self.detection_config['high_threshold']}")
        print(f"  最小距离: {self.detection_config['min_distance']}")
        print(f"  弱散射加权: {self.detection_config['amplitude_weight_low']}")

    def extract_scatterers_from_mat_corrected(self, mat_path):
        """修复版：正确解析用户提供的MAT文件结构"""
        scatterers = []
        try:
            mat_data = sio.loadmat(mat_path)
            scatter_all = mat_data["scatter_all"]

            print(f"    📊 MAT结构: {scatter_all.shape}")

            # 用户描述：scatter_all是Nx1 cell，每个元素是1x1 cell，包含参数数组
            for i in range(scatter_all.shape[0]):
                try:
                    # 第一层：scatter_all[i, 0] 获取第i个cell
                    cell = scatter_all[i, 0]

                    # 第二层：cell[0, 0] 获取内层的参数数组
                    params = cell[0, 0]

                    if isinstance(params, np.ndarray) and len(params) >= 7:
                        # 用户示例：[2.1672 0.3851 0.5000 5.9223e-04 0 0 2.6466]
                        # 参数顺序：[x, y, alpha, gamma, phi_prime, L, A]
                        scatterers.append(
                            {
                                "x": float(params[0]),
                                "y": float(params[1]),
                                "alpha": float(params[2]),
                                "gamma": float(params[3]),
                                "phi_prime": float(params[4]),
                                "L": float(params[5]),
                                "A": float(params[6]),
                            }
                        )
                    else:
                        print(f"    ⚠️ 第{i+1}个参数异常: {type(params)}")

                except (IndexError, TypeError) as e:
                    continue

            print(f"    ✅ 成功提取 {len(scatterers)} 个GT散射中心")

            # 显示GT散射中心统计
            if scatterers:
                amplitudes = [s["A"] for s in scatterers]
                x_coords = [s["x"] for s in scatterers]
                y_coords = [s["y"] for s in scatterers]
                print(f"    📈 幅度范围: {min(amplitudes):.4f} ~ {max(amplitudes):.4f}")
                print(f"    📈 X坐标范围: {min(x_coords):.4f} ~ {max(x_coords):.4f}")
                print(f"    📈 Y坐标范围: {min(y_coords):.4f} ~ {max(y_coords):.4f}")

        except Exception as e:
            print(f"    ❌ MAT解析错误: {e}")
            # 回退到原始方法
            try:
                scatterers = extract_scatterers_from_mat(mat_path)
                print(f"    🔄 回退方法提取 {len(scatterers)} 个散射中心")
            except:
                pass

        return scatterers

    def extract_scatterers_aggressive(self, prediction_maps):
        """Aggressive模式散射中心提取"""
        heatmap, pred_A_log1p, pred_alpha, pred_dx, pred_dy = [prediction_maps[i] for i in range(5)]
        pred_A = np.expm1(pred_A_log1p)

        # 非最大值抑制
        neighborhood_size = 5
        local_max = maximum_filter(heatmap, size=neighborhood_size) == heatmap

        # 高阈值检测
        high_candidates = (heatmap > self.detection_config["high_threshold"]) & local_max
        high_peak_coords = np.argwhere(high_candidates)

        # 低阈值检测
        low_candidates = (heatmap > self.detection_config["low_threshold"]) & local_max
        low_peak_coords = np.argwhere(low_candidates)

        # 合并结果，避免重复
        all_coords = []
        for coord in high_peak_coords:
            all_coords.append((coord[0], coord[1], "high"))

        for coord in low_peak_coords:
            is_duplicate = False
            for existing_coord in high_peak_coords:
                if np.linalg.norm(coord - existing_coord) < self.detection_config["min_distance"]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_coords.append((coord[0], coord[1], "low"))

        scatterers = []
        for r, c, strength in all_coords:
            amplitude_weight = 1.0 if strength == "high" else self.detection_config["amplitude_weight_low"]

            x_base, y_base = pixel_to_model(r, c)
            dx = pred_dx[r, c] * self.PIXEL_SPACING
            dy = pred_dy[r, c] * self.PIXEL_SPACING
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

    def calculate_mse_corrected(self, original_complex, reconstructed_complex):
        """修复版MSE计算 - 使用相对RMSE"""
        original_norm = np.linalg.norm(original_complex)
        if original_norm == 0:
            return float("inf")

        diff_norm = np.linalg.norm(reconstructed_complex - original_complex)
        relative_rmse = diff_norm / original_norm

        return relative_rmse

    def calculate_entropy_corrected(self, image_complex):
        """修复版ENT计算 - 多种方法对比"""
        magnitude = np.abs(image_complex)

        # 方法1：Energy归一化 (原始方法)
        g = np.sqrt(np.sum(magnitude**2))
        if g > 1e-10:
            normalized_magnitude = magnitude / g
            # 使用概率分布形式
            prob_density = normalized_magnitude**2
            # 避免log(0)
            prob_density = prob_density[prob_density > 1e-12]
            if len(prob_density) > 0:
                energy_ent = -np.sum(prob_density * np.log(prob_density))
            else:
                energy_ent = 0.0
        else:
            energy_ent = -np.inf

        # 方法2：直接归一化
        if magnitude.max() > 0:
            direct_normalized = magnitude / magnitude.max()
            prob = direct_normalized**2 / np.sum(direct_normalized**2)
            prob = prob[prob > 1e-12]
            if len(prob) > 0:
                direct_ent = -np.sum(prob * np.log(prob))
            else:
                direct_ent = 0.0
        else:
            direct_ent = -np.inf

        # 方法3：传统图像熵
        if magnitude.max() > 0:
            magnitude_norm = magnitude / magnitude.max()
            magnitude_uint8 = (magnitude_norm * 255).astype(np.uint8)
            hist, _ = np.histogram(magnitude_uint8, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                traditional_ent = -np.sum(hist * np.log2(hist))
            else:
                traditional_ent = 0.0
        else:
            traditional_ent = -np.inf

        # 返回所有方法的结果，主要使用energy归一化方法
        return {
            "energy_normalized": energy_ent,
            "direct_normalized": direct_ent,
            "traditional": traditional_ent,
            "energy_factor": g,
        }

    def analyze_sample(self, sample_info, model):
        """分析单个样本"""
        sar_path = sample_info["sar"]
        base_name = os.path.basename(sar_path).replace(".raw", "")

        try:
            print(f"  🔍 分析样本: {base_name}")

            # 1. 加载原始SAR图像
            original_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
            if original_tensor is None:
                print(f"    ❌ 无法加载SAR图像")
                return None

            original_complex = original_tensor.numpy()

            # 2. 模型预测
            with torch.no_grad():
                input_tensor = original_tensor.unsqueeze(0).to(config.DEVICE)
                predicted_maps = model(input_tensor).squeeze(0).cpu().numpy()

            # 3. 提取散射中心 (Aggressive模式)
            predicted_scatterers = self.extract_scatterers_aggressive(predicted_maps)

            # 4. 重建SAR图像
            reconstructed_complex = reconstruct_sar_image(predicted_scatterers)

            # 5. 计算MSE
            mse_value = self.calculate_mse_corrected(original_complex, reconstructed_complex)

            # 6. 计算ENT (多种方法)
            ent_results = self.calculate_entropy_corrected(reconstructed_complex)

            # 7. 获取GT散射中心
            gt_count = 0
            gt_scatterers = []
            gt_reconstructed = None
            mat_path = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Training_ASC", f"{base_name}.mat")

            if os.path.exists(mat_path):
                print(f"    🔍 解析GT: {base_name}.mat")
                gt_scatterers = self.extract_scatterers_from_mat_corrected(mat_path)
                gt_count = len(gt_scatterers)

                # GT重建
                if gt_scatterers:
                    try:
                        gt_reconstructed = reconstruct_sar_image(gt_scatterers)
                        print(f"    ✅ GT重建成功")
                    except Exception as e:
                        print(f"    ⚠️ GT重建失败: {e}")
            else:
                print(f"    ⚠️ MAT文件不存在")

            # 8. 输出分析结果
            print(f"    📊 结果: MSE={mse_value:.6f}, ENT={ent_results['energy_normalized']:.6f}")
            print(
                f"    🎯 散射中心: 预测{len(predicted_scatterers)}/GT{gt_count} (检出率:{len(predicted_scatterers)/max(gt_count,1):.2%})"
            )
            print(f"    ⚡ 能量因子: {ent_results['energy_factor']:.6f}")

            result = {
                "sample_name": base_name,
                "mse": mse_value,
                "entropy_energy": ent_results["energy_normalized"],
                "entropy_direct": ent_results["direct_normalized"],
                "entropy_traditional": ent_results["traditional"],
                "energy_factor": ent_results["energy_factor"],
                "predicted_count": len(predicted_scatterers),
                "gt_count": gt_count,
                "detection_rate": len(predicted_scatterers) / max(gt_count, 1),
                "original_magnitude": np.abs(original_complex),
                "reconstructed_magnitude": np.abs(reconstructed_complex),
                "gt_magnitude": np.abs(gt_reconstructed) if gt_reconstructed is not None else None,
            }

            return result

        except Exception as e:
            print(f"    ❌ 分析错误: {e}")
            import traceback

            traceback.print_exc()
            return None

    def create_comprehensive_visualization(self, results, output_dir):
        """创建综合可视化"""
        if not results:
            return

        print(f"\n📊 生成综合可视化分析...")

        valid_results = [r for r in results if r is not None]

        # 提取数据
        mse_values = [r["mse"] for r in valid_results if np.isfinite(r["mse"])]
        ent_energy = [r["entropy_energy"] for r in valid_results if np.isfinite(r["entropy_energy"])]
        ent_direct = [r["entropy_direct"] for r in valid_results if np.isfinite(r["entropy_direct"])]
        ent_traditional = [r["entropy_traditional"] for r in valid_results if np.isfinite(r["entropy_traditional"])]
        detection_rates = [r["detection_rate"] for r in valid_results]
        energy_factors = [r["energy_factor"] for r in valid_results]

        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Quantitative Analysis Results (Fixed Version)", fontsize=16)

        # MSE分布
        if mse_values:
            axes[0, 0].hist(mse_values, bins=20, alpha=0.7, color="blue", edgecolor="black")
            axes[0, 0].set_title(f"MSE Distribution (n={len(mse_values)})")
            axes[0, 0].set_xlabel("Relative RMSE")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].axvline(
                np.mean(mse_values), color="red", linestyle="--", label=f"Mean: {np.mean(mse_values):.3f}"
            )
            axes[0, 0].legend()

        # 熵值对比
        x_pos = np.arange(3)
        ent_means = [
            np.mean(ent_energy) if ent_energy else 0,
            np.mean(ent_direct) if ent_direct else 0,
            np.mean(ent_traditional) if ent_traditional else 0,
        ]
        ent_stds = [
            np.std(ent_energy) if len(ent_energy) > 1 else 0,
            np.std(ent_direct) if len(ent_direct) > 1 else 0,
            np.std(ent_traditional) if len(ent_traditional) > 1 else 0,
        ]

        axes[0, 1].bar(x_pos, ent_means, yerr=ent_stds, capsize=5, alpha=0.7, color=["blue", "green", "orange"])
        axes[0, 1].set_title("Entropy Comparison (Different Methods)")
        axes[0, 1].set_xlabel("Method")
        axes[0, 1].set_ylabel("Entropy Value")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(["Energy Norm", "Direct Norm", "Traditional"])

        # 检出率分布
        if detection_rates:
            axes[0, 2].hist(detection_rates, bins=20, alpha=0.7, color="green", edgecolor="black")
            axes[0, 2].set_title(f"Detection Rate Distribution")
            axes[0, 2].set_xlabel("Detection Rate")
            axes[0, 2].set_ylabel("Frequency")
            axes[0, 2].axvline(
                np.mean(detection_rates), color="red", linestyle="--", label=f"Mean: {np.mean(detection_rates):.2%}"
            )
            axes[0, 2].legend()

        # MSE vs 检出率散点图
        if mse_values and detection_rates and len(mse_values) == len(detection_rates):
            axes[1, 0].scatter(detection_rates[: len(mse_values)], mse_values, alpha=0.6)
            axes[1, 0].set_xlabel("Detection Rate")
            axes[1, 0].set_ylabel("MSE (Relative RMSE)")
            axes[1, 0].set_title("MSE vs Detection Rate")

        # 能量因子分布
        if energy_factors:
            axes[1, 1].hist(energy_factors, bins=20, alpha=0.7, color="purple", edgecolor="black")
            axes[1, 1].set_title("Energy Factor Distribution")
            axes[1, 1].set_xlabel("Energy Factor")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_yscale("log")

        # 示例图像对比
        if valid_results:
            sample_result = valid_results[0]

            # 原始vs重建对比
            if sample_result["original_magnitude"] is not None:
                orig_mag = sample_result["original_magnitude"]
                recon_mag = sample_result["reconstructed_magnitude"]

                # 并排显示
                combined = np.hstack([orig_mag, recon_mag])
                axes[1, 2].imshow(combined, cmap="gray")
                axes[1, 2].set_title(f'Sample: {sample_result["sample_name"]}\nOriginal | Reconstructed')
                axes[1, 2].axis("off")

                # 添加分割线
                axes[1, 2].axvline(orig_mag.shape[1], color="red", linewidth=2)

        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(output_dir, "quantitative_analysis_fixed_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ 可视化保存: {save_path}")

    def generate_detailed_report(self, results, output_dir):
        """生成详细报告"""
        if not results:
            return

        valid_results = [r for r in results if r is not None]

        # 统计分析
        mse_values = [r["mse"] for r in valid_results if np.isfinite(r["mse"])]
        ent_energy = [r["entropy_energy"] for r in valid_results if np.isfinite(r["entropy_energy"])]
        detection_rates = [r["detection_rate"] for r in valid_results]
        energy_factors = [r["energy_factor"] for r in valid_results]

        # 生成报告
        report_lines = [
            "=" * 80,
            "📊 定量分析详细报告 (修复版)",
            "=" * 80,
            "",
            f"🔧 分析配置:",
            f"  检测模式: Aggressive",
            f"  低阈值: {self.detection_config['low_threshold']}",
            f"  高阈值: {self.detection_config['high_threshold']}",
            f"  最小距离: {self.detection_config['min_distance']}",
            "",
            f"📈 整体统计:",
            f"  总样本数: {len(valid_results)}",
            f"  有效MSE样本: {len(mse_values)}",
            f"  有效ENT样本: {len(ent_energy)}",
            "",
        ]

        # MSE统计
        if mse_values:
            report_lines.extend(
                [
                    f"🎯 MSE (相对均方根误差) 统计:",
                    f"  平均值: {np.mean(mse_values):.6f}",
                    f"  标准差: {np.std(mse_values):.6f}",
                    f"  中位数: {np.median(mse_values):.6f}",
                    f"  最小值: {np.min(mse_values):.6f}",
                    f"  最大值: {np.max(mse_values):.6f}",
                    f"  25百分位: {np.percentile(mse_values, 25):.6f}",
                    f"  75百分位: {np.percentile(mse_values, 75):.6f}",
                    "",
                ]
            )

        # ENT统计
        if ent_energy:
            valid_ent = [e for e in ent_energy if e != -np.inf]
            inf_count = len(ent_energy) - len(valid_ent)

            report_lines.extend(
                [
                    f"🌊 ENT (Energy归一化熵) 统计:",
                    f"  有效值数量: {len(valid_ent)}",
                    f"  无效值(-inf): {inf_count}",
                ]
            )

            if valid_ent:
                report_lines.extend(
                    [
                        f"  平均值: {np.mean(valid_ent):.6f}",
                        f"  标准差: {np.std(valid_ent):.6f}",
                        f"  中位数: {np.median(valid_ent):.6f}",
                        f"  最小值: {np.min(valid_ent):.6f}",
                        f"  最大值: {np.max(valid_ent):.6f}",
                    ]
                )

            report_lines.append("")

        # 检出率统计
        if detection_rates:
            gt_counts = [r["gt_count"] for r in valid_results]
            pred_counts = [r["predicted_count"] for r in valid_results]

            report_lines.extend(
                [
                    f"🎯 散射中心检出统计:",
                    f"  GT平均数量: {np.mean(gt_counts):.2f}",
                    f"  预测平均数量: {np.mean(pred_counts):.2f}",
                    f"  平均检出率: {np.mean(detection_rates):.2%}",
                    f"  检出率中位数: {np.median(detection_rates):.2%}",
                    f"  检出率标准差: {np.std(detection_rates):.2%}",
                    "",
                ]
            )

        # 性能分析
        if mse_values and ent_energy:
            excellent_mse = len([v for v in mse_values if v < 0.1])
            good_mse = len([v for v in mse_values if v < 0.2])
            valid_ent_count = len([e for e in ent_energy if e > -1])

            report_lines.extend(
                [
                    f"📈 性能汇总:",
                    f"  优秀重建样本 (MSE<0.1): {excellent_mse}/{len(mse_values)} ({excellent_mse/len(mse_values):.1%})",
                    f"  良好重建样本 (MSE<0.2): {good_mse}/{len(mse_values)} ({good_mse/len(mse_values):.1%})",
                    f"  有效熵值样本 (ENT>-1): {valid_ent_count}/{len(ent_energy)} ({valid_ent_count/len(ent_energy):.1%})",
                    "",
                ]
            )

        # 能量因子分析
        if energy_factors:
            zero_energy = len([e for e in energy_factors if e < 1e-10])
            report_lines.extend(
                [
                    f"⚡ 能量因子分析:",
                    f"  平均能量因子: {np.mean(energy_factors):.6f}",
                    f"  零能量样本: {zero_energy}/{len(energy_factors)} ({zero_energy/len(energy_factors):.1%})",
                    "",
                ]
            )

        report_lines.extend(
            [
                "=" * 80,
                "💡 分析建议:",
                "",
                "1. MSE分析:",
                "   - 当前MSE值偏高，说明重建精度有提升空间",
                "   - 建议检查散射中心幅度预测的准确性",
                "",
                "2. ENT分析:",
                "   - 如果ENT值过小或为-inf，可能是重建图像能量过低",
                "   - 建议增强散射中心幅度或改进重建算法",
                "",
                "3. 检出率分析:",
                "   - Aggressive模式下的检出率反映了模型的检测能力",
                "   - 可以调整阈值参数优化检出率",
                "",
                "=" * 80,
            ]
        )

        # 保存报告
        report_path = os.path.join(output_dir, "detailed_analysis_report_fixed.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"✅ 详细报告保存: {report_path}")


def main():
    print("🔧 定量分析工具 (修复版)")
    print("=" * 50)
    print("修复内容:")
    print("1. ✅ 正确解析MAT文件的双重嵌套结构")
    print("2. ✅ 修复MSE计算方法")
    print("3. ✅ 改进ENT计算，提供多种方法对比")
    print("4. ✅ 增强错误处理和调试输出")
    print("5. ✅ 生成详细的诊断报告")

    # 加载模型
    print("\n🤖 正在加载ASC-Net模型...")
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
        print("❌ 未找到有效样本")
        return

    print(f"✅ 找到 {len(dataset.samples)} 个样本")

    # 设置输出目录
    output_dir = os.path.join(project_root, "datasets", "result_vis", "quantitative_fixed")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分析器
    analyzer = QuantitativeAnalyzerFixed()

    # 分析样本
    num_samples = min(100, len(dataset.samples))
    print(f"\n🔍 开始分析 {num_samples} 个样本...")

    results = []
    for i, sample_info in enumerate(tqdm(dataset.samples[:num_samples], desc="分析样本")):
        result = analyzer.analyze_sample(sample_info, model)
        if result:
            results.append(result)

    print(f"\n📊 分析完成！成功处理 {len(results)}/{num_samples} 个样本")

    if results:
        # 生成可视化
        analyzer.create_comprehensive_visualization(results, output_dir)

        # 生成详细报告
        analyzer.generate_detailed_report(results, output_dir)

        print(f"\n🎉 修复版定量分析完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"\n🔍 主要发现:")

        valid_results = [r for r in results if r is not None]
        if valid_results:
            mse_values = [r["mse"] for r in valid_results if np.isfinite(r["mse"])]
            ent_values = [r["entropy_energy"] for r in valid_results if np.isfinite(r["entropy_energy"])]

            if mse_values:
                print(
                    f"  📐 MSE: 平均 {np.mean(mse_values):.4f}, 范围 [{np.min(mse_values):.4f}, {np.max(mse_values):.4f}]"
                )

            if ent_values:
                valid_ent = [e for e in ent_values if e != -np.inf]
                if valid_ent:
                    print(f"  🌊 ENT: 有效值 {len(valid_ent)}/{len(ent_values)}, 平均 {np.mean(valid_ent):.4f}")
                else:
                    print(f"  🌊 ENT: 全部为-inf，需要检查重建算法")
    else:
        print("❌ 未能成功分析任何样本")


if __name__ == "__main__":
    main()
