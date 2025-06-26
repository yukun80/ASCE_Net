#!/usr/bin/env python
# script/test_quantitative_fix.py - 测试修复效果的脚本

import sys
import os
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from quantitative_analysis import QuantitativeAnalyzer


def test_entropy_calculation():
    """测试ENT计算修复"""
    print("🧪 测试ENT计算修复...")

    analyzer = QuantitativeAnalyzer()

    # 测试案例1: 全零图像
    zero_image = np.zeros((128, 128), dtype=complex)
    ent1 = analyzer.calculate_entropy(zero_image)
    print(f"  测试1 - 全零图像ENT: {ent1}")

    # 测试案例2: 正常图像
    normal_image = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)
    ent2 = analyzer.calculate_entropy(normal_image)
    print(f"  测试2 - 随机图像ENT: {ent2:.6f}")

    # 测试案例3: 很小的值
    tiny_image = np.ones((128, 128), dtype=complex) * 1e-15
    ent3 = analyzer.calculate_entropy(tiny_image)
    print(f"  测试3 - 微小值图像ENT: {ent3}")

    # 测试案例4: 单点非零
    single_point = np.zeros((128, 128), dtype=complex)
    single_point[64, 64] = 1.0
    ent4 = analyzer.calculate_entropy(single_point)
    print(f"  测试4 - 单点图像ENT: {ent4:.6f}")


def test_font_fix():
    """测试字体修复"""
    print("\n🎨 测试matplotlib中文字体修复...")

    import matplotlib.pyplot as plt

    # 创建简单测试图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("Font Test - English Title")  # 使用英文标题
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    # 保存测试图
    test_dir = os.path.join(project_root, "datasets", "result_vis", "test")
    os.makedirs(test_dir, exist_ok=True)
    save_path = os.path.join(test_dir, "font_test.png")

    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ 字体测试图保存成功: {save_path}")
    except Exception as e:
        print(f"  ❌ 字体测试失败: {e}")


def main():
    print("🔧 定量分析修复测试")
    print("=" * 50)

    try:
        test_entropy_calculation()
        test_font_fix()

        print("\n✅ 所有测试完成！")
        print("\n📋 修复说明:")
        print("  1. ENT计算: 改进了对全零和极小值图像的处理")
        print("  2. 字体问题: 图表标题改为英文，避免中文字体警告")
        print("  3. 调试信息: 增加了更详细的诊断输出")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
