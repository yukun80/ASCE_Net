import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse

# --- 动态添加项目根目录到Python路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import config  # noqa: E402

# --- 配置参数 ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
P_GRID_SIZE = 84


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


def find_sample_files(basename, split="test_15_deg", class_name="T72"):
    """Find corresponding files for a given sample with SN subdirectory support"""
    # Get current paths based on configuration
    paths = config.get_current_paths()

    if config.USE_COMPLETE_DATASET:
        # For complete dataset, we need to search within SN subdirectories

        # Search for the mat file first to determine the correct SN directory
        mat_class_dir = os.path.join(paths["asc_mat_root"], split, class_name)
        asc_mat_path = None
        label_npy_path = None
        sar_jpg_path = None

        if os.path.exists(mat_class_dir):
            # Search through all SN subdirectories
            for sn_dir in os.listdir(mat_class_dir):
                sn_path = os.path.join(mat_class_dir, sn_dir)
                if os.path.isdir(sn_path):
                    potential_mat = os.path.join(sn_path, f"{basename}_yang.mat")
                    if os.path.exists(potential_mat):
                        asc_mat_path = potential_mat

                        # Find corresponding npy file
                        npy_sn_path = os.path.join(paths["label_save_root"], split, class_name, sn_dir)
                        potential_npy = os.path.join(npy_sn_path, f"{basename}_5ch.npy")
                        if os.path.exists(potential_npy):
                            label_npy_path = potential_npy

                        # Find corresponding JPG file (may need to adjust based on your JPG structure)
                        data_root = os.path.join(project_root, "datasets", "SAR_ASC_Project")
                        jpg_sn_path = os.path.join(data_root, "02_Data_Processed_jpg", split, class_name, sn_dir)
                        potential_jpg = os.path.join(jpg_sn_path, f"{basename}_v1.jpg")
                        if os.path.exists(potential_jpg):
                            sar_jpg_path = potential_jpg

                        break

        # If not found, provide default paths for error reporting
        if asc_mat_path is None:
            asc_mat_path = os.path.join(paths["asc_mat_root"], split, class_name, "SN_XXX", f"{basename}_yang.mat")
        if label_npy_path is None:
            label_npy_path = os.path.join(paths["label_save_root"], split, class_name, "SN_XXX", f"{basename}_5ch.npy")
        if sar_jpg_path is None:
            data_root = os.path.join(project_root, "datasets", "SAR_ASC_Project")
            sar_jpg_path = os.path.join(
                data_root, "02_Data_Processed_jpg", split, class_name, "SN_XXX", f"{basename}_v1.jpg"
            )

    else:
        # Legacy paths for testing
        data_root = os.path.join(project_root, "datasets", "SAR_ASC_Project")
        sar_jpg_path = os.path.join(
            data_root, "02_Data_Processed_jpg_tmp", "test_15_deg", "T72", "SN_132", f"{basename}_v1.jpg"
        )
        asc_mat_path = os.path.join(paths["asc_mat_root"], f"{basename}_yang.mat")
        label_npy_path = os.path.join(paths["label_save_root"], f"{basename}_5ch.npy")

    return sar_jpg_path, asc_mat_path, label_npy_path


def visualize_and_validate(basename, sar_jpg_path, asc_mat_path, label_npy_path):
    """主可视化和验证函数"""
    if not all(os.path.exists(p) for p in [sar_jpg_path, asc_mat_path, label_npy_path]):
        missing_files = [p for p in [sar_jpg_path, asc_mat_path, label_npy_path] if not os.path.exists(p)]
        print(f"错误：样本 {basename} 的文件不完整，缺失文件：")
        for missing in missing_files:
            print(f"  - {missing}")
        return

    # 加载数据
    sar_image = Image.open(sar_jpg_path)
    label_npy = np.load(label_npy_path)
    try:
        cell_array = sio.loadmat(asc_mat_path)["scatter_all"]
        asc_mat_data = np.vstack([cell[0] for cell in cell_array])
    except (ValueError, KeyError):
        asc_mat_data = np.array([])

    # 解码散射点位置
    scatterers_from_npy = decode_asc_from_npy(label_npy)

    pixel_coords_from_npy = [model_to_pixel_final(s["x"], s["y"]) for s in scatterers_from_npy]
    pixel_coords_from_mat = (
        [model_to_pixel_final(x, y) for x, y in asc_mat_data[:, 0:2]] if asc_mat_data.size > 0 else []
    )

    # 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Validation for: {basename}", fontsize=16)

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

    channel_titles = ["Heatmap", "Amplitude (log1p)", "Alpha", "Offset dx", "Offset dy"]
    channel_maps = [label_npy[i] for i in range(5)]

    # 重新安排子图以更好地显示5个通道
    plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, pos in enumerate(plot_positions):
        ax = axes[pos]
        im = ax.imshow(channel_maps[i], cmap="viridis")
        ax.set_title(channel_titles[i])
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = f"validation_output_{basename}.png"
    plt.savefig(output_filename)
    print(f"验证图像已保存至: {output_filename}")
    plt.show()


def list_available_samples(split="test_15_deg", class_name="T72", limit=10):
    """List available samples for validation"""
    paths = config.get_current_paths()

    if config.USE_COMPLETE_DATASET:
        mat_class_dir = os.path.join(paths["asc_mat_root"], split, class_name)
        if os.path.exists(mat_class_dir):
            samples = []
            for sn_dir in os.listdir(mat_class_dir):
                sn_path = os.path.join(mat_class_dir, sn_dir)
                if os.path.isdir(sn_path):
                    for file in os.listdir(sn_path):
                        if file.endswith("_yang.mat"):
                            basename = file.replace("_yang.mat", "")
                            samples.append(basename)
                            if len(samples) >= limit:
                                break
                if len(samples) >= limit:
                    break
            return samples
    else:
        # Legacy mode
        if os.path.exists(paths["asc_mat_root"]):
            samples = []
            for file in os.listdir(paths["asc_mat_root"]):
                if file.endswith("_yang.mat"):
                    basename = file.replace("_yang.mat", "")
                    samples.append(basename)
                    if len(samples) >= limit:
                        break
            return samples

    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate NPY labels against MAT files")
    parser.add_argument("--basename", type=str, default="HB03335.015", help="Base filename to validate")
    parser.add_argument("--split", type=str, default="test_15_deg", help="Dataset split (train_17_deg or test_15_deg)")
    parser.add_argument("--class_name", type=str, default="T72", help="Target class (BTR70, BMP2, T72)")
    parser.add_argument("--list_samples", action="store_true", help="List available samples for validation")

    args = parser.parse_args()

    print(f"Dataset mode: {'Complete Dataset' if config.USE_COMPLETE_DATASET else 'Testing Dataset'}")

    # If user wants to list samples
    if args.list_samples:
        print(f"\nAvailable samples in {args.split}/{args.class_name}:")
        samples = list_available_samples(args.split, args.class_name, limit=20)
        for i, sample in enumerate(samples, 1):
            print(f"  {i:2d}. {sample}")
        print("\nUse any of these samples with --basename option")
        print(
            f"Example: python script/validate_npy_label.py --basename {samples[0] if samples else 'SAMPLE_NAME'} --split {args.split} --class_name {args.class_name}"
        )
        exit(0)

    sar_jpg_path, asc_mat_path, label_npy_path = find_sample_files(args.basename, args.split, args.class_name)

    print(f"Validating sample: {args.basename}")
    print(f"SAR JPG: {sar_jpg_path}")
    print(f"ASC MAT: {asc_mat_path}")
    print(f"Label NPY: {label_npy_path}")

    visualize_and_validate(args.basename, sar_jpg_path, asc_mat_path, label_npy_path)

'''
# 列出测试集T72类别的可用样本
python script/validate_npy_label.py --list_samples --split test_15_deg --class_name T72

# 列出训练集BTR70类别的可用样本  
python script/validate_npy_label.py --list_samples --split train_17_deg --class_name BTR70

# 验证T72类别的样本
python script/validate_npy_label.py --basename HB04001.015.128x128 --split train_17_deg --class_name T72

# 验证BTR70类别的样本
python script/validate_npy_label.py --basename HB05000.004 --split test_15_deg --class_name BTR70
'''