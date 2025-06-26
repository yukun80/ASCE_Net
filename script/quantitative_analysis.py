# script/quantitative_analysis.py - SAR重建定量分析工具 (Aggressive模式)

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import maximum_filter

# 设置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

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
import scipy.io as sio

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class QuantitativeAnalyzer:
    """SAR图像重建定量分析器"""

    def __init__(self, model_path=None):
        """初始化分析器"""
        # 加载模型
        self.model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        self.model.eval()
        print(f"✓ 模型加载成功: {model_path}")

        # 设置aggressive检测配置（与EnhancedVisualizationProcessor一致）
        self.detection_config = {
            "low_threshold": 0.05,
            "high_threshold": 0.3,
            "min_distance": 2,
            "amplitude_weight_low": 0.3,
        }

        # 像素间距常量
        self.PIXEL_SPACING = 0.1

        # 存储结果
        self.results = {
            "mse_values": [],
            "ent_values": [],
            "sample_names": [],
            "original_images": [],
            "reconstructed_images": [],
            "gt_scatterer_counts": [],
            "pred_scatterer_counts": [],
        }

    def calculate_mse(self, original_complex, reconstructed_complex):
        """
        计算重建误差MSE

        公式: MSE = sqrt(1/N * sum(||x_hat - x||_2^2 / ||x||_2^2))

        Args:
            original_complex: 原始SAR复数图像 (H, W)
            reconstructed_complex: 重建SAR复数图像 (H, W)

        Returns:
            mse: 相对均方根误差
        """
        # 确保是numpy数组
        if torch.is_tensor(original_complex):
            original_complex = original_complex.cpu().numpy()
        if torch.is_tensor(reconstructed_complex):
            reconstructed_complex = reconstructed_complex.cpu().numpy()

        # 计算L2范数的平方
        diff_norm_sq = np.sum(np.abs(reconstructed_complex - original_complex) ** 2)
        original_norm_sq = np.sum(np.abs(original_complex) ** 2)

        # 避免除零
        if original_norm_sq == 0:
            return 0.0

        # 相对误差
        relative_error_sq = diff_norm_sq / original_norm_sq
        return np.sqrt(relative_error_sq)

    def calculate_entropy(self, image_complex, normalization_method="energy"):
        """
        计算图像熵ENT

        公式: ENT = -||w/g||_2^2 * ln(||w/g||_2^2)

        Args:
            image_complex: SAR复数图像 (H, W)
            normalization_method: 归一化方法 ('max', 'energy', 'sum')

        Returns:
            entropy: 图像熵值
        """
        # 确保是numpy数组
        if torch.is_tensor(image_complex):
            image_complex = image_complex.cpu().numpy()

        # 获取幅度
        magnitude = np.abs(image_complex)

        # 检查是否为全零图像
        if np.all(magnitude == 0):
            print("  ⚠️ 警告: 重建图像为全零，ENT设为-inf")
            return float("-inf")

        # 归一化
        if normalization_method == "max":
            g = np.max(magnitude)
        elif normalization_method == "energy":
            g = np.sqrt(np.sum(magnitude**2))
        elif normalization_method == "sum":
            g = np.sum(magnitude)
        else:
            raise ValueError(f"未知的归一化方法: {normalization_method}")

        # 检查归一化因子
        if g <= 1e-12:  # 使用更严格的阈值
            print(f"  ⚠️ 警告: 归一化因子过小 g={g:.2e}，ENT设为-inf")
            return float("-inf")

        # 归一化权重
        w_normalized = magnitude / g

        # 计算L2范数的平方
        l2_norm_sq = np.sum(w_normalized**2)

        # 检查L2范数
        if l2_norm_sq <= 1e-12:
            print(f"  ⚠️ 警告: L2范数过小 {l2_norm_sq:.2e}，ENT设为-inf")
            return float("-inf")

        # 计算熵
        entropy = -l2_norm_sq * np.log(l2_norm_sq)

        # 检查结果是否合理
        if np.isnan(entropy) or np.isinf(entropy):
            print(f"  ⚠️ 警告: ENT计算异常 {entropy}，L2范数={l2_norm_sq:.6f}")
            return float("-inf")

        return entropy

    def find_corresponding_mat_file(self, sar_path):
        """查找对应的MAT文件"""
        mat_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")
        base_name = os.path.basename(sar_path).replace(".raw", "")
        return os.path.join(mat_root, f"{base_name}.mat")

    def extract_scatterers_from_mat_fixed(self, mat_path):
        """修复版：正确解析双重嵌套的MAT文件结构"""
        scatterers = []
        try:
            mat_data = sio.loadmat(mat_path)
            scatter_all = mat_data["scatter_all"]

            print(f"    📊 MAT结构: {scatter_all.shape}, 类型: {type(scatter_all)}")

            # 根据用户描述：scatter_all是Nx1 cell，每个元素是1x1 cell，包含参数数组
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
                        print(f"    ⚠️ 第{i+1}个散射中心参数异常: {type(params)}, {params}")

                except (IndexError, TypeError) as e:
                    print(f"    ⚠️ 解析第{i+1}个散射中心错误: {e}")
                    continue

            print(f"    ✅ 正确提取 {len(scatterers)} 个散射中心")

            # 显示散射中心统计
            if scatterers:
                amplitudes = [s["A"] for s in scatterers]
                print(f"    📈 幅度范围: {min(amplitudes):.4f} ~ {max(amplitudes):.4f}")
                print(f"    📈 平均幅度: {np.mean(amplitudes):.4f}")

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
        """
        使用aggressive配置提取散射中心

        Args:
            prediction_maps: 模型预测的5通道输出 [heatmap, A_log1p, alpha, dx, dy]

        Returns:
            scatterers: 散射中心列表
        """
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

    def analyze_sample(self, sample_info, threshold=0.5):
        """
        分析单个样本

        Args:
            sample_info: 样本信息字典
            threshold: 散射中心检测阈值

        Returns:
            result_dict: 分析结果字典
        """
        sar_path = sample_info["sar"]
        base_name = os.path.basename(sar_path).replace(".raw", "")

        try:
            # 1. 加载原始SAR数据
            original_sar_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)
            if original_sar_tensor is None:
                print(f"  ✗ 无法加载SAR数据: {base_name}")
                return None

            # 移除batch维度，获取复数数据
            original_complex = original_sar_tensor.squeeze(0)  # [H, W] complex tensor

            # 2. 模型预测
            with torch.no_grad():
                input_tensor = original_sar_tensor.unsqueeze(0).to(config.DEVICE)
                predicted_maps = self.model(input_tensor).squeeze(0).cpu().numpy()

            # 3. 使用aggressive配置提取预测的散射中心
            predicted_scatterers = self.extract_scatterers_aggressive(predicted_maps)

            # 4. 重建SAR图像
            if predicted_scatterers:
                reconstructed_complex = reconstruct_sar_image(predicted_scatterers)
                recon_energy = np.sum(np.abs(reconstructed_complex) ** 2)
                print(f"    重建图像能量: {recon_energy:.6f}")
            else:
                print(f"    ⚠️ 未检测到散射中心，重建图像为全零")
                reconstructed_complex = np.zeros_like(original_complex.cpu().numpy())

            # 5. 计算MSE
            mse_value = self.calculate_mse(original_complex, reconstructed_complex)

            # 6. 计算ENT (对重建图像，使用energy归一化)
            ent_value = self.calculate_entropy(reconstructed_complex, normalization_method="energy")

            # 7. 获取GT散射中心（使用修复的解析方法）
            gt_count = 0
            gt_scatterers = []
            mat_path = self.find_corresponding_mat_file(sar_path)
            if os.path.exists(mat_path):
                print(f"    🔍 解析GT散射中心: {os.path.basename(mat_path)}")
                gt_scatterers = self.extract_scatterers_from_mat_fixed(mat_path)
                gt_count = len(gt_scatterers)

                # 如果修复方法失败，尝试原始方法
                if gt_count == 0:
                    try:
                        gt_scatterers = extract_scatterers_from_mat(mat_path)
                        gt_count = len(gt_scatterers)
                        print(f"    🔄 使用原始方法获取 {gt_count} 个GT散射中心")
                    except:
                        print(f"    ❌ 无法解析GT散射中心")
            else:
                print(f"    ⚠️ MAT文件不存在: {os.path.basename(mat_path)}")

            # 8. 可选：使用GT散射中心重建进行对比
            gt_reconstructed = None
            if gt_scatterers:
                try:
                    gt_reconstructed = reconstruct_sar_image(gt_scatterers)
                    print(f"    ✅ GT重建图像生成成功")
                except Exception as e:
                    print(f"    ⚠️ GT重建失败: {e}")

            result = {
                "sample_name": base_name,
                "mse": mse_value,
                "entropy": ent_value,
                "original_image": original_complex.cpu().numpy(),
                "reconstructed_image": reconstructed_complex,
                "gt_scatterer_count": gt_count,
                "pred_scatterer_count": len(predicted_scatterers),
                "reconstruction_ratio": len(predicted_scatterers) / max(gt_count, 1),
            }

            print(
                f"  ✓ {base_name}: MSE={mse_value:.6f}, ENT={ent_value:.6f}, "
                f"散射中心={len(predicted_scatterers)}/{gt_count} (检出率:{len(predicted_scatterers)/max(gt_count,1):.2%})"
            )

            return result

        except Exception as e:
            print(f"  ✗ 分析错误 {base_name}: {e}")
            return None

    def batch_analyze(self, dataset, max_samples=None):
        """
        批量分析数据集

        Args:
            dataset: MSTAR数据集
            max_samples: 最大分析样本数
        """
        print(f"\n=== 开始批量定量分析 ===")
        print(f"📋 分析配置:")
        print(f"  • 散射中心检测: Aggressive模式")
        print(f"    - 低阈值: {self.detection_config['low_threshold']}")
        print(f"    - 高阈值: {self.detection_config['high_threshold']}")
        print(f"    - 最小距离: {self.detection_config['min_distance']}")
        print(f"    - 弱散射中心权重: {self.detection_config['amplitude_weight_low']}")
        print(f"  • MSE计算: 相对均方根误差")
        print(f"  • ENT计算: Energy归一化方法")

        samples_to_analyze = len(dataset.samples)
        if max_samples:
            samples_to_analyze = min(max_samples, samples_to_analyze)

        print(f"\n分析样本数: {samples_to_analyze}")

        successful_count = 0

        for i, sample_info in enumerate(tqdm(dataset.samples[:samples_to_analyze], desc="分析进度")):
            result = self.analyze_sample(sample_info)

            if result:
                # 存储结果
                self.results["mse_values"].append(result["mse"])
                self.results["ent_values"].append(result["entropy"])
                self.results["sample_names"].append(result["sample_name"])
                self.results["original_images"].append(result["original_image"])
                self.results["reconstructed_images"].append(result["reconstructed_image"])
                self.results["gt_scatterer_counts"].append(result["gt_scatterer_count"])
                self.results["pred_scatterer_counts"].append(result["pred_scatterer_count"])

                successful_count += 1

        print(f"\n=== 分析完成 ===")
        print(f"成功分析: {successful_count}/{samples_to_analyze} 个样本")

        if successful_count > 0:
            self.generate_statistics()

    def generate_statistics(self):
        """生成统计结果"""
        mse_values = np.array(self.results["mse_values"])
        ent_values = np.array(self.results["ent_values"])

        print(f"\n📊 定量分析统计结果 (Aggressive模式):")
        print(f"{'='*60}")

        print(f"🎯 MSE (重建误差) - 相对均方根误差:")
        print(f"  样本数量:     {len(mse_values)}")
        print(f"  平均值:       {np.mean(mse_values):.6f}")
        print(f"  标准差:       {np.std(mse_values):.6f}")
        print(f"  中位数:       {np.median(mse_values):.6f}")
        print(f"  最小值:       {np.min(mse_values):.6f}")
        print(f"  最大值:       {np.max(mse_values):.6f}")
        print(f"  25百分位:     {np.percentile(mse_values, 25):.6f}")
        print(f"  75百分位:     {np.percentile(mse_values, 75):.6f}")

        # 处理ENT值中的-inf
        valid_ent = ent_values[~np.isinf(ent_values)]
        inf_count = np.sum(np.isinf(ent_values))

        print(f"\n🌊 ENT (图像熵) - Energy归一化:")
        print(f"  样本数量:     {len(ent_values)}")
        print(f"  有效样本:     {len(valid_ent)}")
        print(f"  无效样本(-inf):{inf_count}")

        if len(valid_ent) > 0:
            print(f"  平均值:       {np.mean(valid_ent):.6f}")
            print(f"  标准差:       {np.std(valid_ent):.6f}")
            print(f"  中位数:       {np.median(valid_ent):.6f}")
            print(f"  最小值:       {np.min(valid_ent):.6f}")
            print(f"  最大值:       {np.max(valid_ent):.6f}")
            print(f"  25百分位:     {np.percentile(valid_ent, 25):.6f}")
            print(f"  75百分位:     {np.percentile(valid_ent, 75):.6f}")
        else:
            print(f"  ⚠️ 所有ENT值均为-inf (重建图像能量过低)")

        # 散射中心检出率统计
        gt_counts = np.array(self.results["gt_scatterer_counts"])
        pred_counts = np.array(self.results["pred_scatterer_counts"])
        detection_ratios = pred_counts / np.maximum(gt_counts, 1)

        print(f"\n🎯 散射中心检出统计 (Aggressive模式):")
        print(f"  GT平均数量:   {np.mean(gt_counts):.2f}")
        print(f"  预测平均数量: {np.mean(pred_counts):.2f}")
        print(f"  平均检出率:   {np.mean(detection_ratios):.2%}")
        print(f"  检出率中位数: {np.median(detection_ratios):.2%}")
        print(f"  检出率标准差: {np.std(detection_ratios):.2%}")
        print(
            f"  完美检出样本: {np.sum(detection_ratios >= 0.95)}/{len(detection_ratios)} ({np.sum(detection_ratios >= 0.95)/len(detection_ratios):.1%})"
        )
        print(
            f"  良好检出样本: {np.sum(detection_ratios >= 0.8)}/{len(detection_ratios)} ({np.sum(detection_ratios >= 0.8)/len(detection_ratios):.1%})"
        )

        print(f"\n📈 性能汇总:")
        print(
            f"  MSE < 0.1的样本: {np.sum(mse_values < 0.1)}/{len(mse_values)} ({np.sum(mse_values < 0.1)/len(mse_values):.1%})"
        )
        print(
            f"  MSE < 0.2的样本: {np.sum(mse_values < 0.2)}/{len(mse_values)} ({np.sum(mse_values < 0.2)/len(mse_values):.1%})"
        )
        print(
            f"  ENT > -1的样本:  {np.sum(ent_values > -1)}/{len(ent_values)} ({np.sum(ent_values > -1)/len(ent_values):.1%})"
        )
        print(f"{'='*60}")

    def save_results(self, output_dir):
        """保存分析结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存统计数据
        stats_file = os.path.join(output_dir, "quantitative_statistics.txt")
        with open(stats_file, "w", encoding="utf-8") as f:
            mse_values = np.array(self.results["mse_values"])
            ent_values = np.array(self.results["ent_values"])

            f.write("SAR图像重建定量分析结果 (Aggressive模式)\n")
            f.write("=" * 60 + "\n\n")

            f.write("分析配置:\n")
            f.write(f"  散射中心检测: Aggressive模式\n")
            f.write(f"  ENT归一化方法: Energy\n\n")

            f.write("MSE (重建误差) 统计 - 相对均方根误差:\n")
            f.write(f"  样本数量: {len(mse_values)}\n")
            f.write(f"  平均值: {np.mean(mse_values):.6f}\n")
            f.write(f"  标准差: {np.std(mse_values):.6f}\n")
            f.write(f"  中位数: {np.median(mse_values):.6f}\n")
            f.write(f"  最小值: {np.min(mse_values):.6f}\n")
            f.write(f"  最大值: {np.max(mse_values):.6f}\n")
            f.write(f"  25百分位: {np.percentile(mse_values, 25):.6f}\n")
            f.write(f"  75百分位: {np.percentile(mse_values, 75):.6f}\n\n")

            # 处理ENT值中的-inf
            valid_ent = ent_values[~np.isinf(ent_values)]
            inf_count = np.sum(np.isinf(ent_values))

            f.write("ENT (图像熵) 统计 - Energy归一化:\n")
            f.write(f"  样本数量: {len(ent_values)}\n")
            f.write(f"  有效样本: {len(valid_ent)}\n")
            f.write(f"  无效样本(-inf): {inf_count}\n")

            if len(valid_ent) > 0:
                f.write(f"  平均值: {np.mean(valid_ent):.6f}\n")
                f.write(f"  标准差: {np.std(valid_ent):.6f}\n")
                f.write(f"  中位数: {np.median(valid_ent):.6f}\n")
                f.write(f"  最小值: {np.min(valid_ent):.6f}\n")
                f.write(f"  最大值: {np.max(valid_ent):.6f}\n")
                f.write(f"  25百分位: {np.percentile(valid_ent, 25):.6f}\n")
                f.write(f"  75百分位: {np.percentile(valid_ent, 75):.6f}\n")
            else:
                f.write("  ⚠️ 所有ENT值均为-inf (重建图像能量过低)\n")

        # 2. 保存详细数据
        detailed_file = os.path.join(output_dir, "detailed_results.txt")
        with open(detailed_file, "w", encoding="utf-8") as f:
            f.write("样本名称\tMSE\tENT\tGT散射中心\t预测散射中心\t检出率\n")
            for i, name in enumerate(self.results["sample_names"]):
                gt_count = self.results["gt_scatterer_counts"][i]
                pred_count = self.results["pred_scatterer_counts"][i]
                ratio = pred_count / max(gt_count, 1)

                f.write(
                    f"{name}\t{self.results['mse_values'][i]:.6f}\t"
                    f"{self.results['ent_values'][i]:.6f}\t{gt_count}\t"
                    f"{pred_count}\t{ratio:.3f}\n"
                )

        # 3. 生成可视化图表
        self.plot_statistics(output_dir)

        print(f"\n✓ 结果已保存到: {output_dir}")

    def plot_statistics(self, output_dir):
        """生成统计图表"""
        mse_values = np.array(self.results["mse_values"])
        ent_values = np.array(self.results["ent_values"])

        # 处理ENT中的-inf值
        valid_ent = ent_values[~np.isinf(ent_values)]
        if len(valid_ent) == 0:
            # 如果所有ENT都是-inf，用一个很小的负数替代以便绘图
            ent_values = np.where(np.isinf(ent_values), -10, ent_values)

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("SAR Image Reconstruction Quantitative Analysis Results", fontsize=16)

        # MSE直方图
        axes[0, 0].hist(mse_values, bins=30, alpha=0.7, color="blue", edgecolor="black")
        axes[0, 0].set_title("MSE Distribution")
        axes[0, 0].set_xlabel("MSE Value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # ENT直方图
        axes[0, 1].hist(ent_values, bins=30, alpha=0.7, color="green", edgecolor="black")
        axes[0, 1].set_title("ENT Distribution")
        axes[0, 1].set_xlabel("ENT Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # MSE vs ENT散点图
        axes[1, 0].scatter(mse_values, ent_values, alpha=0.6, color="red")
        axes[1, 0].set_title("MSE vs ENT Relationship")
        axes[1, 0].set_xlabel("MSE Value")
        axes[1, 0].set_ylabel("ENT Value")
        axes[1, 0].grid(True, alpha=0.3)

        # 散射中心检出率
        gt_counts = np.array(self.results["gt_scatterer_counts"])
        pred_counts = np.array(self.results["pred_scatterer_counts"])

        axes[1, 1].scatter(gt_counts, pred_counts, alpha=0.6, color="purple")
        axes[1, 1].plot([0, max(gt_counts)], [0, max(gt_counts)], "r--", label="Ideal Line")
        axes[1, 1].set_title("Scatterer Detection Comparison")
        axes[1, 1].set_xlabel("GT Scatterer Count")
        axes[1, 1].set_ylabel("Predicted Scatterer Count")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "quantitative_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 SAR图像重建定量分析工具 - Aggressive模式")
    print("=" * 70)
    print("📋 分析指标:")
    print("  • MSE: 重建误差 (相对均方根误差)")
    print("  • ENT: 图像熵 (Energy归一化)")
    print("  • 散射中心检出率统计")
    print("-" * 70)

    # 初始化分析器
    analyzer = QuantitativeAnalyzer()

    # 加载数据集
    dataset = MSTAR_ASC_5CH_Dataset()
    if not dataset.samples:
        print("✗ 未找到有效样本")
        return

    print(f"✓ 找到 {len(dataset.samples)} 个样本")

    # 进行批量分析（可以设置max_samples限制样本数）
    analyzer.batch_analyze(dataset, max_samples=100)  # 分析前100个样本

    # 保存结果
    output_dir = os.path.join(project_root, "datasets", "result_vis", "quantitative_analysis")
    analyzer.save_results(output_dir)

    print(f"\n🎉 定量分析完成！")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📊 生成文件:")
    print(f"  • quantitative_statistics.txt - 详细统计数据")
    print(f"  • detailed_results.txt - 每个样本的详细结果")
    print(f"  • quantitative_analysis.png - 可视化图表")
    print("=" * 70)


if __name__ == "__main__":
    main()
