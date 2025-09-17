import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle

"""
单幅图像处理：
python script/step5_visualize_label.py --input datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS/HB03406.017.npy

批处理：
python script/step5_visualize_label.py --root datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS

自定义输出目录：
python script/step5_visualize_label.py --root datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS --out datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS_Visuallse

"""

# --- 动态添加项目根目录到Python路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import config  # noqa: E402


def robust_percentile_max(values: np.ndarray, percentile: float = 99.5) -> float:
    if values.size == 0:
        return 1.0
    vmax = np.percentile(values, percentile)
    if vmax <= 0 or not np.isfinite(vmax):
        return 1.0
    return float(vmax)


def save_channel_image(channel: np.ndarray, ch_index: int, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])  # 占满整个画布，避免外部边距

    def add_outer_right_colorbar(im_ref):
        cb = fig.colorbar(im_ref, ax=ax, pad=0.005, fraction=0.025, shrink=0.75)
        # 仅显示首尾刻度
        try:
            vmin = float(im_ref.norm.vmin)
            vmax = float(im_ref.norm.vmax)
            cb.set_ticks([vmin, vmax])
        except Exception:
            pass
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=14, length=0, pad=1)
        return cb

    if ch_index == 0:
        # Heatmap [0,1]
        im = ax.imshow(channel, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        add_outer_right_colorbar(im)

    elif ch_index == 1:
        # ln(1+A) 稀疏点，做鲁棒缩放并可选轻度模糊以提升可见度（仅可视化）
        nz = channel[channel > 0]
        vmax = robust_percentile_max(nz, 99.5)
        disp = channel.copy()
        if np.count_nonzero(nz) > 0:
            disp = gaussian_filter(disp, sigma=0.8)
        im = ax.imshow(disp, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
        add_outer_right_colorbar(im)

    elif ch_index == 2:
        # alpha 角度/形状参数：非零像素上色，零像素使用深色背景以凸显alpha
        nz = channel[channel != 0]
        if nz.size > 0:
            vmin = float(np.min(nz))
            vmax = float(np.max(nz))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -1.0, 1.0

        # 使用非循环色带，避免色带首尾颜色重复；不屏蔽0值
        cmap = plt.get_cmap("viridis")
        im = ax.imshow(channel, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        add_outer_right_colorbar(im)

    elif ch_index == 3:
        # dx 发散色图，固定范围
        im = ax.imshow(channel, cmap="seismic", vmin=-0.5, vmax=0.5, interpolation="nearest")
        add_outer_right_colorbar(im)

    elif ch_index == 4:
        # dy 发散色图，固定范围
        im = ax.imshow(channel, cmap="seismic", vmin=-0.5, vmax=0.5, interpolation="nearest")
        add_outer_right_colorbar(im)

    else:
        # 兜底：直接显示
        im = ax.imshow(channel, cmap="gray", interpolation="nearest")
        add_outer_right_colorbar(im)

    # 添加黑色边框（宽度=1pt）围绕图像区域
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="black", linewidth=1))
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_label_file(npy_path: str, out_root: str, keep_rel_from: str):
    # 读取
    label = np.load(npy_path)
    if label.ndim != 3 or label.shape[0] != 5:
        print(f"[WARN] Skip invalid shape {label.shape} at {npy_path}")
        return

    # 子目录结构
    rel_dir = os.path.relpath(os.path.dirname(npy_path), keep_rel_from)
    base = os.path.splitext(os.path.basename(npy_path))[0]

    # 目标输出目录
    out_dir = os.path.join(out_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 通道命名
    names = [
        "ch0_heatmap",
        "ch1_amp_ln1p",
        "ch2_alpha",
        "ch3_dx",
        "ch4_dy",
    ]

    # 保存每个通道
    for ch in range(5):
        ch_img_path = os.path.join(out_dir, f"{base}_{names[ch]}.png")
        save_channel_image(label[ch], ch, ch_img_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize 5-channel ASC labels and save per-channel PNGs.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single .npy label file to visualize. If not set, will scan --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=config.LABEL_SAVE_ROOT_5CH,
        help="Root directory containing 5-channel .npy labels (batch mode).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            os.path.dirname(project_root),  # default align with repository root if desired
            os.path.basename(project_root),
            "datasets",
            "SAR_ASC_Project",
            "tmp_MSTAR_ASC_LABELS_Visuallse",
        ),
        help="Output root directory for per-channel visualizations.",
    )
    args = parser.parse_args()

    # 兼容：如果 --out 未指定，就用项目内默认路径
    if args.out is None:
        args.out = os.path.join(project_root, "datasets", "SAR_ASC_Project", "tmp_MSTAR_ASC_LABELS_Visuallse")

    os.makedirs(args.out, exist_ok=True)

    if args.input is not None:
        # 单文件模式
        npy_path = os.path.normpath(args.input)
        visualize_label_file(npy_path, args.out, keep_rel_from=os.path.dirname(npy_path))
        print(f"[OK] Saved visualizations for: {os.path.basename(npy_path)}")
        return

    # 批量模式
    root = os.path.normpath(args.root)
    count_total = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".npy"):
                npy_path = os.path.join(dirpath, fname)
                visualize_label_file(npy_path, args.out, keep_rel_from=root)
                count_total += 1
    print(f"[DONE] Saved visualizations for {count_total} label files under: {args.out}")


if __name__ == "__main__":
    main()
