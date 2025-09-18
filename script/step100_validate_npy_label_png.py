"""
脚本用途: 5通道标签验证与可视化
- 读取单个样本的 SAR 预览图、对应的 .mat 标注与 5 通道 .npy 标签。
- 从 .npy 中解码散射中心（基于 heatmap 峰与 dx/dy 亚像素偏移），并可与 .mat 中的GT散射中心对齐对比。
- 生成对比可视化图：原图+叠加散射中心，以及 5 个通道的热力图，并保存 PNG。

使用方法:
- 在脚本底部修改 `basename` 与 `data_dir` 后直接运行，或改造为命令行参数。
"""

import os, glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

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


def visualize_and_validate(basename, sar_jpg_path, asc_mat_path, label_npy_path):
    """主可视化和验证函数"""
    if not all(os.path.exists(p) for p in [sar_jpg_path, asc_mat_path, label_npy_path]):
        print(f"错误：样本 {basename} 的文件不完整，请检查路径。")
        # ... (错误处理) ...
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

    # ... (与之前相同的绘图逻辑, 但使用新的 pixel_coords*) ...
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


if __name__ == "__main__":
    basename = "HB03335.015"
    data_dir = r"E:\Document\paper_library\3rd_paper_250512\code\ASCE_Net\datasets\SAR_ASC_Project"
    print(
        "JPG:",
        os.path.exists(
            os.path.join(data_dir, "02_Data_Processed_jpg", "test_15_deg", "T72", "SN_132", f"{basename}_v1.jpg")
        ),
    )
    print("MAT:", os.path.exists(os.path.join(data_dir, "tmp_Training_ASC", f"{basename}.mat")))
    print("NPY:", os.path.exists(os.path.join(data_dir, "tmp_MSTAR_ASC_LABELS", f"{basename}.npy")))
    # Optional: locate a basename that exists
    print("Some 5ch npy:", glob.glob(os.path.join(data_dir, "tmp_MSTAR_ASC_LABELS", "HB*.npy"))[:5])

    sar_jpg_path = os.path.join(data_dir, "02_Data_Processed_jpg", "test_15_deg", "T72", "SN_132", f"{basename}_v1.jpg")
    asc_mat_path = os.path.join(data_dir, "tmp_Training_ASC", f"{basename}.mat")
    label_npy_path = os.path.join(data_dir, "tmp_MSTAR_ASC_LABELS", f"{basename}.npy")

    visualize_and_validate(basename, sar_jpg_path, asc_mat_path, label_npy_path)
