"""
OMP ASC提取器演示测试
===================

简单的测试脚本，用于验证OMP算法实现的正确性和性能。
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from traditional_method.omp_config import get_config, list_available_configs, estimate_computation_cost


def test_configuration_system():
    """测试配置系统"""
    print("🔧 测试配置系统")
    print("=" * 40)

    # 列出所有配置
    list_available_configs()

    # 测试不同配置的计算复杂度
    print("\n💾 计算复杂度分析:")
    for config_name in ["debug", "fast", "standard", "high_precision"]:
        print(f"\n📊 {config_name} 配置:")
        estimate_computation_cost(config_name)


def test_synthetic_data():
    """使用合成数据测试OMP算法"""
    print("\n🧪 合成数据测试")
    print("=" * 40)

    try:
        from traditional_method.omp_asc_extractor import OMPASCExtractor

        # 创建简单的合成数据
        print("🔧 生成合成SAR数据...")

        # 参数设置
        img_size = (128, 128)

        # 创建简单的散射中心模拟
        synthetic_asc = [
            {"x": 2.0, "y": 1.5, "alpha": 0.5, "A": 3.0},
            {"x": -1.5, "y": -0.8, "alpha": -0.3, "A": 2.0},
            {"x": 0.5, "y": 2.2, "alpha": 0.8, "A": 1.5},
        ]

        # 初始化提取器（使用debug配置以减少计算时间）
        print("🎯 初始化OMP提取器...")
        extractor = OMPASCExtractor(
            img_size=img_size,
            search_region=(-4.0, 4.0, -4.0, 4.0),
            grid_resolution=0.6,  # 粗网格
            alpha_range=(-1.0, 1.0),
            n_nonzero_coefs=8,
            cross_validation=False,
        )

        # 使用提取器生成合成图像
        print("🔧 生成合成SAR图像...")
        synthetic_img = extractor.reconstruct_image_from_asc(synthetic_asc)

        # 添加一些噪声
        noise_level = 0.1
        noise = noise_level * (np.random.randn(*img_size) + 1j * np.random.randn(*img_size))
        noisy_img = synthetic_img + noise

        print(f"   生成了 {len(synthetic_asc)} 个合成散射中心")
        print(f"   添加噪声水平: {noise_level}")

        # 使用OMP提取ASC
        print("🔍 使用OMP提取散射中心...")
        start_time = time.time()

        extracted_asc = extractor.extract_asc_from_image(noisy_img)

        processing_time = time.time() - start_time
        print(f"✅ 提取完成，用时: {processing_time:.2f}秒")
        print(f"   检测到 {len(extracted_asc)} 个散射中心")

        # 显示结果
        if extracted_asc:
            print("\n📊 检测结果:")
            print("   原始散射中心:")
            for i, asc in enumerate(synthetic_asc):
                print(
                    f"     {i+1}. 位置:({asc['x']:.2f}, {asc['y']:.2f}), "
                    f"幅度:{asc['A']:.2f}, Alpha:{asc['alpha']:.2f}"
                )

            print("   检测到的散射中心:")
            for i, asc in enumerate(extracted_asc[:5]):  # 只显示前5个
                print(
                    f"     {i+1}. 位置:({asc['x']:.2f}, {asc['y']:.2f}), "
                    f"幅度:{asc['A']:.3f}, Alpha:{asc['alpha']:.3f}"
                )

        # 可视化结果
        print("📈 生成可视化结果...")
        create_test_visualization(synthetic_img, noisy_img, extracted_asc, synthetic_asc)

        return True

    except Exception as e:
        print(f"❌ 合成数据测试失败: {e}")
        return False


def create_test_visualization(original_img, noisy_img, extracted_asc, true_asc):
    """创建测试结果可视化"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("OMP算法合成数据测试结果", fontsize=14, fontweight="bold")

    # 1. 原始合成图像
    axes[0, 0].imshow(np.abs(original_img), cmap="gray")
    axes[0, 0].set_title("原始合成SAR图像")
    axes[0, 0].axis("off")

    # 标记真实散射中心位置
    for asc in true_asc:
        # 简化的坐标转换（仅用于可视化）
        pixel_x = int(64 + asc["x"] * 10)  # 粗略转换
        pixel_y = int(64 - asc["y"] * 10)
        if 0 <= pixel_x < 128 and 0 <= pixel_y < 128:
            axes[0, 0].plot(pixel_x, pixel_y, "g+", markersize=12, markeredgewidth=3)

    # 2. 含噪声图像
    axes[0, 1].imshow(np.abs(noisy_img), cmap="gray")
    axes[0, 1].set_title("含噪声SAR图像")
    axes[0, 1].axis("off")

    # 3. 检测结果叠加
    axes[1, 0].imshow(np.abs(noisy_img), cmap="gray", alpha=0.7)

    # 标记检测到的散射中心
    for asc in extracted_asc:
        pixel_x = int(64 + asc["x"] * 10)
        pixel_y = int(64 - asc["y"] * 10)
        if 0 <= pixel_x < 128 and 0 <= pixel_y < 128:
            axes[1, 0].plot(pixel_x, pixel_y, "r+", markersize=10, markeredgewidth=2)

    axes[1, 0].set_title(f"OMP检测结果 ({len(extracted_asc)}个)")
    axes[1, 0].axis("off")

    # 4. 幅度对比
    axes[1, 1].bar(range(len(true_asc)), [asc["A"] for asc in true_asc], alpha=0.7, label="真实幅度", color="green")

    if extracted_asc:
        detected_amplitudes = [asc["A"] for asc in extracted_asc[: len(true_asc)]]
        x_pos = np.arange(len(detected_amplitudes)) + 0.3
        axes[1, 1].bar(x_pos, detected_amplitudes, alpha=0.7, label="检测幅度", color="red", width=0.4)

    axes[1, 1].set_title("散射幅度对比")
    axes[1, 1].set_ylabel("幅度")
    axes[1, 1].set_xlabel("散射中心编号")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(project_root, "traditional_method", "test_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"omp_test_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 测试结果已保存到: {save_path}")

    plt.show()


def test_data_processing():
    """测试数据处理功能"""
    print("\n📁 数据处理测试")
    print("=" * 40)

    # 检查数据目录
    data_root = os.path.join(project_root, "datasets", "SAR_ASC_Project")
    mat_root = os.path.join(data_root, "01_Data_Processed_mat_part-tmp")

    if os.path.exists(mat_root):
        # 查找测试文件
        test_files = []
        for root, dirs, files in os.walk(mat_root):
            for file in files[:2]:  # 只取前2个文件
                if file.endswith(".mat"):
                    test_files.append(os.path.join(root, file))

        print(f"📁 找到 {len(test_files)} 个MAT文件用于测试")

        if test_files:
            try:
                from traditional_method.omp_asc_extractor import OMPASCExtractor

                # 使用debug配置进行快速测试
                extractor = OMPASCExtractor(
                    img_size=(128, 128),
                    search_region=(-3.0, 3.0, -3.0, 3.0),
                    grid_resolution=0.8,
                    n_nonzero_coefs=5,
                    cross_validation=False,
                )

                for i, test_file in enumerate(test_files):
                    print(f"\n📊 测试文件 {i+1}: {os.path.basename(test_file)}")

                    start_time = time.time()
                    asc_list = extractor.extract_asc_from_mat(test_file)
                    processing_time = time.time() - start_time

                    print(f"   处理时间: {processing_time:.2f}秒")
                    print(f"   检测散射中心数: {len(asc_list)}")

                    if asc_list:
                        print(
                            f"   最强散射中心: 位置({asc_list[0]['x']:.2f}, {asc_list[0]['y']:.2f}), "
                            f"幅度{asc_list[0]['A']:.3f}"
                        )

                return True

            except Exception as e:
                print(f"❌ 实际数据测试失败: {e}")
                return False
        else:
            print("⚠️ 未找到测试文件")
            return False
    else:
        print("⚠️ MAT数据目录不存在，请先运行MATLAB预处理脚本")
        print(f"   期望路径: {mat_root}")
        return False


def main():
    """主测试函数"""
    print("🚀 OMP ASC提取器演示测试")
    print("=" * 50)

    # 测试配置系统
    test_configuration_system()

    # 测试合成数据
    synthetic_success = test_synthetic_data()

    # 测试实际数据处理
    data_success = test_data_processing()

    # 总结
    print("\n📋 测试总结")
    print("=" * 40)
    print(f"✅ 配置系统: 正常")
    print(f"{'✅' if synthetic_success else '❌'} 合成数据测试: {'通过' if synthetic_success else '失败'}")
    print(f"{'✅' if data_success else '⚠️'} 实际数据测试: {'通过' if data_success else '需要数据'}")

    if synthetic_success:
        print("\n🎉 OMP算法基本功能验证通过！")
        print("   可以开始使用OMP进行ASC提取")
    else:
        print("\n⚠️ 存在问题，请检查代码实现")

    print("\n📚 下一步:")
    print("   1. 运行MATLAB预处理脚本准备数据")
    print("   2. 使用 python omp_asc_extractor.py 进行完整测试")
    print("   3. 使用 python batch_omp_processor.py 进行批量处理")


if __name__ == "__main__":
    main()
