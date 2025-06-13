# preprocess.py

import sys
import os

"""
python step4_label_preprocess.py
这个脚本用于将ASC的.mat文件转换为.npy文件，用于训练和测试
"""

# --- 开始添加的代码 ---
# 获取当前脚本文件所在的目录 (ASCE_Net/script)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (ASCE_Net)
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(project_root)
# --- 结束添加的代码 ---
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils import config  # noqa: E402


def model_to_pixel(x_model, y_model):
    """将物理模型的米制坐标转换为128x128图像的像素坐标"""
    # 这个公式来源于你的matlab脚本 step35_Output_Difference.m
    # 必须保证这个转换是正确的，否则标签就是错误的
    col_pixel = (x_model * config.IMG_HEIGHT / (0.3 * config.P_GRID_SIZE)) + (config.IMG_WIDTH / 2 + 1)
    row_pixel = (config.IMG_HEIGHT / 2 + 1) - (y_model * config.IMG_HEIGHT / (0.3 * config.P_GRID_SIZE))
    return int(round(row_pixel)), int(round(col_pixel))


def process_asc_mat_file(mat_path):
    """处理单个 .mat 文件, 返回两个参数图像"""
    try:
        asc_mat = sio.loadmat(mat_path)
        scatter_all = asc_mat["scatter_all"][0]
    except Exception as e:
        print(f"Could not read {mat_path}: {e}")
        return None, None

    image_A = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)
    image_alpha = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)

    for scatter_i in scatter_all:
        params = scatter_i[0]
        # 根据你的 matlab 代码 extrac.m 和 visualize_results.m 推断参数顺序
        # scatter{sca_s,1}=[M(1) M(2) a M(3) 0 0 A]; -> [x, y, alpha, gamma, phi, L, A]
        x, y, alpha, _, _, _, A = params

        # 坐标转换
        row, col = model_to_pixel(x, y)

        # 检查边界
        if 0 <= row < config.IMG_HEIGHT and 0 <= col < config.IMG_WIDTH:
            image_A[row, col] = A
            image_alpha[row, col] = alpha
        else:
            print(f"Warning: Scatterer at ({x:.2f}, {y:.2f}) -> pixel ({row}, {col}) is out of bounds.")

    return image_A, image_alpha


def main():
    print("Starting preprocessing...")
    print(f"ASC MAT Root: {config.ASC_MAT_ROOT}")
    print(f"Label Save Root: {config.LABEL_SAVE_ROOT}")

    mat_files = []
    for dirpath, _, filenames in os.walk(config.ASC_MAT_ROOT):
        for filename in filenames:
            if filename.endswith("_yang.mat"):
                mat_files.append(os.path.join(dirpath, filename))

    if not mat_files:
        print("No _yang.mat files found. Please run the MATLAB scripts first.")
        return

    for mat_path in tqdm(mat_files, desc="Processing .mat files"):
        image_A, image_alpha = process_asc_mat_file(mat_path)

        if image_A is not None and image_alpha is not None:
            # 创建保存路径
            relative_path = os.path.relpath(os.path.dirname(mat_path), config.ASC_MAT_ROOT)
            save_dir = os.path.join(config.LABEL_SAVE_ROOT, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            # 获取文件名
            base_name = os.path.basename(mat_path).replace(".mat", "")

            # 保存 .npy 文件
            np.save(os.path.join(save_dir, f"{base_name}_A.npy"), image_A)
            np.save(os.path.join(save_dir, f"{base_name}_alpha.npy"), image_alpha)

    print("Preprocessing finished successfully!")


if __name__ == "__main__":
    main()
