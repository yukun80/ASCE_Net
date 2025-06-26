#!/usr/bin/env python
# script/debug_mat_parser.py - 调试MAT文件解析

import sys
import os
import scipy.io as sio
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def debug_mat_structure(mat_path):
    """调试MAT文件的真实结构"""
    print(f"🔍 调试MAT文件: {os.path.basename(mat_path)}")

    try:
        mat_data = sio.loadmat(mat_path)
        print(f"📋 MAT文件键: {list(mat_data.keys())}")

        if "scatter_all" in mat_data:
            scatter_all = mat_data["scatter_all"]
            print(f"📐 scatter_all形状: {scatter_all.shape}")
            print(f"📐 scatter_all类型: {type(scatter_all)}")

            # 检查第一个元素的结构
            if len(scatter_all) > 0:
                first_element = scatter_all[0, 0]
                print(f"📐 第一个元素类型: {type(first_element)}")
                print(f"📐 第一个元素形状: {first_element.shape if hasattr(first_element, 'shape') else 'No shape'}")
                print(f"📐 第一个元素内容: {first_element}")

                # 如果是数组，直接使用
                if isinstance(first_element, np.ndarray) and first_element.ndim == 1:
                    print(f"✅ 找到参数数组: {first_element}")
                    return first_element

                # 如果还有嵌套，继续解析
                if hasattr(first_element, "__len__") and len(first_element) > 0:
                    inner = first_element[0]
                    print(f"📐 内层元素类型: {type(inner)}")
                    print(f"📐 内层元素内容: {inner}")

                    if hasattr(inner, "__len__") and len(inner) > 0:
                        deeper = inner[0]
                        print(f"📐 更深层元素类型: {type(deeper)}")
                        print(f"📐 更深层元素内容: {deeper}")

        return None
    except Exception as e:
        print(f"❌ 解析错误: {e}")
        return None


def extract_scatterers_from_mat_fixed(mat_path):
    """修复版：根据用户提供的真实结构解析"""
    scatterers = []
    try:
        mat_data = sio.loadmat(mat_path)
        scatter_all = mat_data["scatter_all"]

        print(f"  📊 MAT文件结构: {scatter_all.shape}")

        # 根据用户描述：scatter_all是36x1 cell，每个元素是1x1 cell，包含参数数组
        for i in range(scatter_all.shape[0]):
            try:
                # 第一层：scatter_all[i, 0] 获取第i个cell
                cell = scatter_all[i, 0]

                # 第二层：cell[0, 0] 获取内层的参数数组
                params = cell[0, 0]

                if isinstance(params, np.ndarray) and len(params) >= 7:
                    # 根据用户示例：[2.1672 0.3851 0.5000 5.9223e-04 0 0 2.6466]
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
                    print(f"  ⚠️ 第{i+1}个散射中心参数格式异常: {params}")

            except (IndexError, TypeError) as e:
                print(f"  ⚠️ 解析第{i+1}个散射中心时出错: {e}")
                continue

        print(f"  ✅ 成功提取 {len(scatterers)} 个散射中心")

        # 显示前几个散射中心的参数
        for i, s in enumerate(scatterers[:3]):
            print(f"    散射中心{i+1}: x={s['x']:.4f}, y={s['y']:.4f}, A={s['A']:.4f}, α={s['alpha']:.4f}")

    except Exception as e:
        print(f"  ❌ MAT文件解析错误: {e}")

    return scatterers


def main():
    print("🔧 MAT文件解析调试工具")
    print("=" * 50)

    # 查找一个MAT文件进行测试
    mat_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")

    if os.path.exists(mat_root):
        mat_files = [f for f in os.listdir(mat_root) if f.endswith(".mat")]

        if mat_files:
            test_file = os.path.join(mat_root, mat_files[0])
            print(f"📁 测试文件: {test_file}")

            # 调试结构
            debug_mat_structure(test_file)

            print("\n" + "=" * 50)

            # 测试修复的解析函数
            print("🧪 测试修复的解析函数:")
            scatterers = extract_scatterers_from_mat_fixed(test_file)

            if scatterers:
                print(f"\n📈 解析成功！共{len(scatterers)}个散射中心")
                print("📋 散射中心统计:")
                amplitudes = [s["A"] for s in scatterers]
                print(f"  幅度范围: {min(amplitudes):.4f} ~ {max(amplitudes):.4f}")
                print(f"  平均幅度: {np.mean(amplitudes):.4f}")
            else:
                print("❌ 未提取到散射中心")
        else:
            print("❌ 未找到MAT文件")
    else:
        print(f"❌ MAT目录不存在: {mat_root}")


if __name__ == "__main__":
    main()
