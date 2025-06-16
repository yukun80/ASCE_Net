# script/step4_label_preprocess.py (重构版)

import os
import sys
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# --- 动态添加项目根目录到Python路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import config  # noqa: E402


def gaussian_2d(shape, sigma=1):
    """Generates a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def model_to_pixel(x_model, y_model):
    """
    使用最终修正的坐标中心进行转换。
    """
    C1 = config.IMG_HEIGHT / (0.3 * config.P_GRID_SIZE)
    # 修正：使用 (width - 1) / 2.0 作为最精确的中心
    C2_h = (config.IMG_HEIGHT - 1) / 2.0  # 63.5
    C2_w = (config.IMG_WIDTH - 1) / 2.0  # 63.5
    col_pixel = (x_model * C1) + C2_w
    row_pixel = C2_h - (y_model * C1)
    return row_pixel, col_pixel


def main():
    # 确保输出目录存在
    os.makedirs(config.LABEL_SAVE_ROOT_5CH, exist_ok=True)

    print(f"Reading .mat files from: {config.ASC_MAT_ROOT}")
    print(f"Saving 5-channel .npy labels to: {config.LABEL_SAVE_ROOT_5CH}")

    all_mat_files = []
    for root, _, fnames in os.walk(config.ASC_MAT_ROOT):
        for fname in fnames:
            if fname.endswith("_yang.mat"):
                all_mat_files.append(os.path.join(root, fname))

    for mat_path in tqdm(all_mat_files, desc="Generating Labels"):
        try:
            # 修正：正确处理Matlab元胞数组
            cell_array = sio.loadmat(mat_path)["scatter_all"]
            if cell_array.size == 0:
                print(f"Warning: Skipping empty .mat file: {os.path.basename(mat_path)}")
                continue
            scatter_data = np.vstack([cell[0] for cell in cell_array])
        except (KeyError, IndexError, FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not process {os.path.basename(mat_path)}. Error: {e}")
            continue

        # 初始化5通道标签图
        label_map = np.zeros((5, config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)
        gaussian_kernel = gaussian_2d((7, 7), sigma=1)

        for params in scatter_data:
            # params: [x, y, alpha, gamma, phi_prime, L, A]
            x_m, y_m, alpha, A = params[0], params[1], params[2], params[6]

            # 计算精确的浮点像素位置
            row_f, col_f = model_to_pixel(x_m, y_m)

            # 四舍五入到最近的整数像素作为峰值中心
            row_c, col_c = int(round(row_f)), int(round(col_f))

            if not (0 <= row_c < config.IMG_HEIGHT and 0 <= col_c < config.IMG_WIDTH):
                continue

            # 计算亚像素偏移量
            dy = row_f - row_c
            dx = col_f - col_c

            # 绘制高斯热力图
            k_size = gaussian_kernel.shape[0]
            k_half = k_size // 2

            # 计算高斯核在标签图上的放置范围，并处理边界情况
            r_start, r_end = max(0, row_c - k_half), min(config.IMG_HEIGHT, row_c + k_half + 1)
            c_start, c_end = max(0, col_c - k_half), min(config.IMG_WIDTH, col_c + k_half + 1)

            kr_start, kr_end = max(0, k_half - (row_c - r_start)), min(k_size, k_half + (r_end - row_c))
            kc_start, kc_end = max(0, k_half - (col_c - c_start)), min(k_size, k_half + (c_end - col_c))

            # 确保不会覆盖更强的峰值
            label_map[0, r_start:r_end, c_start:c_end] = np.maximum(
                label_map[0, r_start:r_end, c_start:c_end], gaussian_kernel[kr_start:kr_end, kc_start:kc_end]
            )

            # 在整数峰值位置填充回归通道的值
            label_map[1, row_c, col_c] = np.log1p(A)
            label_map[2, row_c, col_c] = alpha
            label_map[3, row_c, col_c] = dx
            label_map[4, row_c, col_c] = dy

        # 保存5通道的.npy文件
        output_filename = os.path.basename(mat_path).replace("_yang.mat", "_5ch.npy")
        relative_path = os.path.relpath(os.path.dirname(mat_path), config.ASC_MAT_ROOT)
        output_dir = os.path.join(config.LABEL_SAVE_ROOT_5CH, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, output_filename), label_map)

    print("\nLabel generation complete.")


if __name__ == "__main__":
    main()
