"""
批量OMP处理器
=============

用于批量处理MSTAR数据集，比较不同OMP配置的性能，
并生成详细的分析报告。
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from traditional_method.omp_asc_extractor import OMPASCExtractor
from traditional_method.omp_config import get_config, list_available_configs
from utils.reconstruction import extract_scatterers_from_mat


class BatchOMPProcessor:
    """批量OMP处理器"""

    def __init__(self, data_root=None, output_root=None):
        """
        初始化批量处理器

        Parameters:
        -----------
        data_root : str, optional
            MSTAR数据根目录
        output_root : str, optional
            输出结果根目录
        """
        if data_root is None:
            self.data_root = os.path.join(project_root, "datasets", "SAR_ASC_Project")
        else:
            self.data_root = data_root

        if output_root is None:
            self.output_root = os.path.join(project_root, "traditional_method", "batch_results")
        else:
            self.output_root = output_root

        # 创建输出目录
        os.makedirs(self.output_root, exist_ok=True)

        self.mat_root = os.path.join(self.data_root, "01_Data_Processed_mat_part-tmp")
        self.raw_root = os.path.join(self.data_root, "02_Data_Processed_raw")
        self.gt_asc_root = os.path.join(self.data_root, "03_Training_ASC")

        print(f"🎯 批量OMP处理器初始化完成")
        print(f"   数据根目录: {self.data_root}")
        print(f"   输出根目录: {self.output_root}")

    def find_test_files(self, max_files=None, file_pattern=None):
        """
        查找测试文件

        Parameters:
        -----------
        max_files : int, optional
            最大处理文件数
        file_pattern : str, optional
            文件名模式过滤

        Returns:
        --------
        list
            测试文件信息列表
        """
        test_files = []

        if not os.path.exists(self.mat_root):
            print(f"❌ MAT数据目录不存在: {self.mat_root}")
            return test_files

        for root, dirs, files in os.walk(self.mat_root):
            for file in files:
                if file.endswith(".mat"):
                    if file_pattern and file_pattern not in file:
                        continue

                    mat_path = os.path.join(root, file)

                    # 查找对应的GT ASC文件
                    rel_path = os.path.relpath(root, self.mat_root)
                    base_name = os.path.splitext(file)[0]

                    gt_asc_path = None
                    if os.path.exists(self.gt_asc_root):
                        gt_asc_candidate = os.path.join(self.gt_asc_root, rel_path, base_name + ".mat")
                        if os.path.exists(gt_asc_candidate):
                            gt_asc_path = gt_asc_candidate

                    test_files.append(
                        {"mat_path": mat_path, "gt_asc_path": gt_asc_path, "base_name": base_name, "rel_path": rel_path}
                    )

                    if max_files and len(test_files) >= max_files:
                        break

            if max_files and len(test_files) >= max_files:
                break

        print(f"📁 找到 {len(test_files)} 个测试文件")
        if gt_asc_path:
            gt_count = sum(1 for f in test_files if f["gt_asc_path"] is not None)
            print(f"   其中 {gt_count} 个文件有对应的GT ASC标注")

        return test_files

    def process_single_file(self, file_info, config_name, extractor=None):
        """
        处理单个文件

        Parameters:
        -----------
        file_info : dict
            文件信息
        config_name : str
            配置名称
        extractor : OMPASCExtractor, optional
            OMP提取器实例

        Returns:
        --------
        dict
            处理结果
        """
        if extractor is None:
            config = get_config(config_name)
            omp_params = config["omp_params"]

            extractor = OMPASCExtractor(
                img_size=config["image_params"]["img_size"],
                search_region=omp_params["search_region"],
                grid_resolution=omp_params["grid_resolution"],
                amplitude_range=(0.1, 10.0),
                alpha_range=omp_params["alpha_range"],
                n_nonzero_coefs=omp_params["n_nonzero_coefs"],
                cross_validation=omp_params["cross_validation"],
            )

        result = {
            "file_info": file_info,
            "config_name": config_name,
            "success": False,
            "error_message": None,
            "processing_time": 0,
            "n_detected_scatterers": 0,
            "n_gt_scatterers": 0,
            "reconstruction_metrics": {},
            "detected_asc": [],
            "gt_asc": [],
        }

        try:
            start_time = time.time()

            # 提取ASC
            asc_list = extractor.extract_asc_from_mat(file_info["mat_path"])

            processing_time = time.time() - start_time

            result.update(
                {
                    "success": True,
                    "processing_time": processing_time,
                    "n_detected_scatterers": len(asc_list),
                    "detected_asc": asc_list,
                }
            )

            # 读取GT ASC（如果存在）
            if file_info["gt_asc_path"]:
                try:
                    gt_asc = extract_scatterers_from_mat(file_info["gt_asc_path"])
                    result["gt_asc"] = gt_asc
                    result["n_gt_scatterers"] = len(gt_asc)
                except Exception as e:
                    print(f"⚠️ 读取GT ASC失败: {e}")

            # 计算重建质量指标
            if asc_list:
                try:
                    import scipy.io as sio

                    mat_data = sio.loadmat(file_info["mat_path"])
                    original_img = mat_data["Img"] * np.exp(1j * mat_data["phase"])

                    recon_img = extractor.reconstruct_image_from_asc(asc_list)
                    metrics = extractor.evaluate_reconstruction(original_img, recon_img)
                    result["reconstruction_metrics"] = metrics
                except Exception as e:
                    print(f"⚠️ 重建评估失败: {e}")

        except Exception as e:
            result.update({"success": False, "error_message": str(e)})

        return result

    def batch_process(self, config_names=["standard"], max_files=10, parallel=False):
        """
        批量处理多个配置

        Parameters:
        -----------
        config_names : list
            配置名称列表
        max_files : int
            最大处理文件数
        parallel : bool
            是否并行处理

        Returns:
        --------
        dict
            批量处理结果
        """
        print(f"🚀 开始批量处理")
        print(f"   配置方案: {config_names}")
        print(f"   最大文件数: {max_files}")
        print(f"   并行处理: {parallel}")

        # 查找测试文件
        test_files = self.find_test_files(max_files=max_files)
        if not test_files:
            print("❌ 未找到测试文件")
            return {}

        all_results = {}

        for config_name in config_names:
            print(f"\n📊 处理配置: {config_name}")
            print("-" * 40)

            config_results = []

            # 串行处理
            for file_info in tqdm(test_files, desc=f"处理{config_name}"):
                result = self.process_single_file(file_info, config_name)
                config_results.append(result)

            all_results[config_name] = config_results

            # 打印配置摘要
            self._print_config_summary(config_name, config_results)

        # 保存结果
        self._save_batch_results(all_results)

        # 生成比较报告
        if len(all_results) > 1:
            self._generate_comparison_report(all_results)

        return all_results

    def _print_config_summary(self, config_name, results):
        """打印配置处理摘要"""
        successful_results = [r for r in results if r["success"]]
        n_success = len(successful_results)
        n_total = len(results)

        if n_success > 0:
            avg_time = np.mean([r["processing_time"] for r in successful_results])
            avg_scatterers = np.mean([r["n_detected_scatterers"] for r in successful_results])

            # 重建质量统计
            rmse_values = [
                r["reconstruction_metrics"].get("relative_rmse", float("inf"))
                for r in successful_results
                if r["reconstruction_metrics"]
            ]
            avg_rmse = np.mean(rmse_values) if rmse_values else float("inf")

            print(f"✅ 配置 {config_name} 处理完成:")
            print(f"   成功率: {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")
            print(f"   平均处理时间: {avg_time:.2f}秒")
            print(f"   平均检测散射中心数: {avg_scatterers:.1f}")
            print(f"   平均重建RMSE: {avg_rmse:.4f}")
        else:
            print(f"❌ 配置 {config_name} 全部处理失败")

    def _save_batch_results(self, all_results):
        """保存批量处理结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存摘要统计（CSV格式）
        summary_data = []
        for config_name, results in all_results.items():
            successful_results = [r for r in results if r["success"]]

            if successful_results:
                summary_data.append(
                    {
                        "config_name": config_name,
                        "n_files": len(results),
                        "n_success": len(successful_results),
                        "success_rate": len(successful_results) / len(results),
                        "avg_processing_time": np.mean([r["processing_time"] for r in successful_results]),
                        "std_processing_time": np.std([r["processing_time"] for r in successful_results]),
                        "avg_detected_scatterers": np.mean([r["n_detected_scatterers"] for r in successful_results]),
                        "std_detected_scatterers": np.std([r["n_detected_scatterers"] for r in successful_results]),
                        "avg_rmse": np.mean(
                            [
                                r["reconstruction_metrics"].get("relative_rmse", float("inf"))
                                for r in successful_results
                                if r["reconstruction_metrics"]
                            ]
                        ),
                        "avg_correlation": np.mean(
                            [
                                r["reconstruction_metrics"].get("correlation", 0)
                                for r in successful_results
                                if r["reconstruction_metrics"]
                            ]
                        ),
                    }
                )

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_root, f"batch_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"📊 摘要统计已保存到: {summary_file}")

    def _generate_comparison_report(self, all_results):
        """生成比较分析报告"""
        if len(all_results) < 2:
            print("⚠️ 需要至少2个配置才能生成比较报告")
            return

        print("\n📈 生成比较分析报告...")

        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("OMP配置性能比较分析", fontsize=16, fontweight="bold")

        config_names = list(all_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(config_names)))

        # 1. 处理时间比较
        ax = axes[0, 0]
        processing_times = []
        labels = []
        for config_name in config_names:
            successful_results = [r for r in all_results[config_name] if r["success"]]
            times = [r["processing_time"] for r in successful_results]
            if times:
                processing_times.append(times)
                labels.append(config_name)

        if processing_times:
            ax.boxplot(processing_times, labels=labels)
            ax.set_title("处理时间分布")
            ax.set_ylabel("时间 (秒)")
            ax.tick_params(axis="x", rotation=45)

        # 2. 检测散射中心数比较
        ax = axes[0, 1]
        scatterer_counts = []
        for config_name in config_names:
            successful_results = [r for r in all_results[config_name] if r["success"]]
            counts = [r["n_detected_scatterers"] for r in successful_results]
            if counts:
                scatterer_counts.append(counts)

        if scatterer_counts:
            ax.boxplot(scatterer_counts, labels=labels)
            ax.set_title("检测散射中心数分布")
            ax.set_ylabel("散射中心数")
            ax.tick_params(axis="x", rotation=45)

        # 3. 成功率比较
        ax = axes[1, 0]
        success_rates = []
        for config_name in config_names:
            results = all_results[config_name]
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            success_rates.append(success_rate)

        bars = ax.bar(config_names, success_rates, color=colors)
        ax.set_title("处理成功率")
        ax.set_ylabel("成功率")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=45)

        # 4. 处理时间 vs 散射中心数
        ax = axes[1, 1]
        for i, config_name in enumerate(config_names):
            successful_results = [r for r in all_results[config_name] if r["success"]]
            times = [r["processing_time"] for r in successful_results]
            counts = [r["n_detected_scatterers"] for r in successful_results]

            if times and counts:
                ax.scatter(times, counts, label=config_name, color=colors[i], alpha=0.7)

        ax.set_xlabel("处理时间 (秒)")
        ax.set_ylabel("检测散射中心数")
        ax.set_title("处理时间 vs 检测数量")
        ax.legend()

        plt.tight_layout()

        # 保存图表
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_root, f"comparison_report_{timestamp}.png")
        plt.savefig(report_file, dpi=300, bbox_inches="tight")
        print(f"📊 比较报告已保存到: {report_file}")

        plt.show()


def main():
    """主函数：批量处理演示"""
    print("🚀 批量OMP处理器演示")
    print("=" * 50)

    # 初始化批量处理器
    processor = BatchOMPProcessor()

    # 列出可用配置
    print("\n🔧 可用配置:")
    list_available_configs()

    # 批量处理多个配置
    config_names = ["debug", "fast", "standard"]  # 选择几个配置进行比较
    max_files = 5  # 限制文件数以减少演示时间

    print(f"\n🎯 开始批量处理演示")
    print(f"   选择配置: {config_names}")
    print(f"   最大文件数: {max_files}")

    try:
        results = processor.batch_process(
            config_names=config_names,
            max_files=max_files,
            parallel=False,  # 演示时使用串行处理
        )

        if results:
            print("\n🎉 批量处理完成！")
            print("   详细结果已保存到输出目录")
        else:
            print("\n❌ 批量处理失败，请检查数据路径")

    except Exception as e:
        print(f"\n❌ 批量处理出错: {e}")


if __name__ == "__main__":
    main()
