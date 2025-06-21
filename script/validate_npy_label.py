import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import random

# --- 配置参数 (可以从config文件导入，这里为保持独立性而保留) ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
P_GRID_SIZE = 84


# --- 坐标转换函数 (与旧版相同) ---
def model_to_pixel_final(x_model, y_model):
    """最终修正版坐标转换"""
    C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
    C2_h = (IMG_HEIGHT - 1) / 2.0
    C2_w = (IMG_WIDTH - 1) / 2.0
    col_pixel = (x_model * C1) + C2_w
    row_pixel = C2_h - (y_model * C1)
    return row_pixel, col_pixel


def pixel_to_model_final(row, col):
    """最终修正版反向坐标转换"""
    C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
    C2_h = (IMG_HEIGHT - 1) / 2.0
    C2_w = (IMG_WIDTH - 1) / 2.0
    y_model = (C2_h - row) / C1
    x_model = (col - C2_w) / C1
    return x_model, y_model


def decode_asc_from_npy(npy_data):
    """从5通道的npy标签中解码出ASC参数列表。"""
    heatmap = npy_data[0, :, :]
    amplitude_map = npy_data[1, :, :]
    alpha_map = npy_data[2, :, :]
    dx_map = npy_data[3, :, :]
    dy_map = npy_data[4, :, :]

    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(heatmap, size=5) == heatmap
    peaks = local_max & (heatmap > 0.1)
    rows, cols = np.where(peaks)

    scatterers = []
    for r, c in zip(rows, cols):
        amp = np.expm1(amplitude_map[r, c])
        alpha = alpha_map[r, c]
        dx = dx_map[r, c]
        dy = dy_map[r, c]
        precise_row = r + dy
        precise_col = c + dx
        x_model, y_model = pixel_to_model_final(precise_row, precise_col)
        scatterer = {"x": x_model, "y": y_model, "A": amp, "alpha": alpha}
        scatterers.append(scatterer)
    return scatterers


def find_all_samples(data_root):
    """
    遍历数据集目录，找到所有拥有完整文件的样本。
    返回一个字典列表，每个字典包含 'basename' 和 'rel_path'。
    """
    label_root = os.path.join(data_root, "04_MSTAR_ASC_LABELS")
    if not os.path.exists(label_root):
        print(f"错误: 标签目录不存在 {label_root}")
        return []

    samples = []
    print(f"正在从 {label_root} 搜索样本...")
    for dirpath, _, filenames in os.walk(label_root):
        for filename in filenames:
            if filename.endswith("_5ch.npy"):
                basename = filename.replace("_5ch.npy", "")
                rel_path = os.path.relpath(dirpath, label_root)
                samples.append({"basename": basename, "rel_path": rel_path})
    print(f"共找到 {len(samples)} 个有效样本。")
    return samples


def visualize_and_validate(basename, sar_jpg_path, asc_mat_path, label_npy_path, output_dir):
    """主可视化和验证函数"""
    if not all(os.path.exists(p) for p in [sar_jpg_path, asc_mat_path, label_npy_path]):
        print(f"警告：样本 {basename} 的文件不完整，跳过此样本。")
        print(f"  - JPG: {sar_jpg_path} (存在: {os.path.exists(sar_jpg_path)})")
        print(f"  - MAT: {asc_mat_path} (存在: {os.path.exists(asc_mat_path)})")
        print(f"  - NPY: {label_npy_path} (存在: {os.path.exists(label_npy_path)})")
        return

    # 加载数据
    sar_image = Image.open(sar_jpg_path)
    label_npy = np.load(label_npy_path)
    try:
        cell_array = sio.loadmat(asc_mat_path)["scatter_all"]
        asc_mat_data = np.vstack([cell[0] for cell in cell_array])
    except (ValueError, KeyError, TypeError):
        asc_mat_data = np.array([])

    # 解码散射点位置
    scatterers_from_npy = decode_asc_from_npy(label_npy)
    pixel_coords_from_npy = [model_to_pixel_final(s["x"], s["y"]) for s in scatterers_from_npy]
    pixel_coords_from_mat = (
        [model_to_pixel_final(x, y) for x, y in asc_mat_data[:, 0:2]] if asc_mat_data.size > 0 else []
    )

    # 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Validation for: {os.path.join(os.path.relpath(os.path.dirname(label_npy_path), os.path.dirname(os.path.dirname(label_npy_path))), basename)}",
        fontsize=14,
    )

    # 绘制散射点覆盖图
    axes[0, 0].imshow(sar_image, cmap="gray")
    if pixel_coords_from_mat:
        rows, cols = zip(*pixel_coords_from_mat)
        axes[0, 0].scatter(cols, rows, marker="+", c="lime", s=120, linewidths=2, label="GT from .mat")
    if pixel_coords_from_npy:
        rows, cols = zip(*pixel_coords_from_npy)
        axes[0, 0].scatter(cols, rows, marker="x", c="red", s=60, label="Decoded from .npy")
    axes[0, 0].set_title("Original SAR + Scatterer Overlay")
    axes[0, 0].legend()
    axes[0, 0].axis("off")

    # 绘制5个通道
    channel_titles = ["Heatmap", "Amplitude (log1p)", "Alpha", "Offset dx", "Offset dy"]
    plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, pos in enumerate(plot_positions):
        ax = axes[pos]
        im = ax.imshow(label_npy[i], cmap="viridis")
        ax.set_title(channel_titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"validation_output_{basename}.png")
    plt.savefig(output_filename)
    print(f"验证图像已保存至: {output_filename}")
    # plt.show() # 在批量处理时建议注释掉此行
    plt.close(fig)  # 批量处理时关闭图像以释放内存


def main():
    parser = argparse.ArgumentParser(description="Validate SAR ASC .npy labels against .mat ground truth.")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录 (e.g., 'datasets/SAR_ASC_Project')")
    parser.add_argument("--num_samples", type=int, default=5, help="要随机验证的样本数量")
    parser.add_argument(
        "--sample_name", type=str, default=None, help="指定要验证的单个样本的basename (e.g., 'HB03335.015')"
    )
    parser.add_argument("--output_dir", type=str, default="datasets/SAR_ASC_Project/validation_outputs", help="保存验证结果图像的目录")

    args = parser.parse_args()

    all_samples = find_all_samples(args.data_root)
    if not all_samples:
        return

    if args.sample_name:
        # 验证指定样本
        # 注意: 指定的sample_name不应包含'.128x128'
        selected_samples = [s for s in all_samples if s["basename"].startswith(args.sample_name)]
        if not selected_samples:
            print(f"错误: 找不到指定的样本 '{args.sample_name}'")
            return
    else:
        # 随机选择样本
        num_to_select = min(args.num_samples, len(all_samples))
        selected_samples = random.sample(all_samples, num_to_select)

    for sample in selected_samples:
        npy_basename = sample["basename"]  # e.g., 'HB03335.015.128x128'
        rel_path = sample["rel_path"]

        # --- 核心修正逻辑 ---
        # 1. 从.npy文件名中解析出真正的基础名 (去掉 .128x128)
        if ".128x128" in npy_basename:
            true_basename = npy_basename.replace(".128x128", "")
        else:
            true_basename = npy_basename

        # 2. 动态构建所有文件的路径
        label_npy_path = os.path.join(args.data_root, "04_MSTAR_ASC_LABELS", rel_path, f"{npy_basename}_5ch.npy")
        asc_mat_path = os.path.join(args.data_root, "03_Training_ASC", rel_path, f"{npy_basename}_yang.mat")

        # 3. 使用修正后的true_basename构建JPG路径，并添加_v1后缀
        jpg_root = os.path.join(args.data_root, "02_Data_Processed_jpg")
        sar_jpg_path = os.path.join(jpg_root, rel_path, f"{true_basename}_v1.jpg")

        # 为了兼容大小写，检查.jpg和.JPG
        if not os.path.exists(sar_jpg_path):
            sar_jpg_path_upper = os.path.join(jpg_root, rel_path, f"{true_basename}_v1.JPG")
            if os.path.exists(sar_jpg_path_upper):
                sar_jpg_path = sar_jpg_path_upper

        visualize_and_validate(npy_basename, sar_jpg_path, asc_mat_path, label_npy_path, args.output_dir)


if __name__ == "__main__":
    main()

"""
# 随机验证3个样本
python script/validate_npy_label.py --data_root "datasets/SAR_ASC_Project" --num_samples 3

# 验证一个特定的样本
python script/validate_npy_label.py --data_root "datasets/SAR_ASC_Project" --sample_name "HB04001.015"

# 指定输出目录
python script/validate_npy_label.py --data_root "datasets/SAR_ASC_Project" --num_samples 5 --output_dir "my_validation_images"


"""
