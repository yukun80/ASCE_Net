"""
Reconstruct SAR images directly from 2-channel ASC label maps (A, alpha).

Purpose:
- Verify dataset label correctness by rebuilding SAR images from ground-truth
  parameter maps, independent of the neural network prediction.

Usage:
  python A-ConvNets-FCN/reconstruct_from_label.py \
    --config A-ConvNets-FCN/config.yaml \
    [--labels datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS_2ch] \
    [--topk 5] [--nms-window 5] [--amp-pctl 99.5] [--amp-pctl-scale 1.0] \
    [--save-dir A-ConvNets-FCN/outputs/recon_from_labels] [--compare-raw]
    


python A-ConvNets-FCN/reconstruct_from_label.py --config config.yaml --labels datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS_2ch --topk 80 --nms-window 5 --amp-pctl 99.5 --amp-pctl-scale 1.0 --compare-raw --save-dir A-ConvNets-FCN/outputs/recon_from_labels

python A-ConvNets-FCN/reconstruct_from_label.py --config config.yaml --labels datasets/SAR_ASC_Project/tmp_MSTAR_ASC_LABELS_2ch --topk 80 --nz-only --min-dist 7 --nms-window 5 --amp-pctl 99.7 --amp-pctl-scale 1.0 --compare-raw --save-dir A-ConvNets-FCN/outputs/recon_from_labels
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from utils.io import ensure_dir, load_config  # noqa: E402
from utils.peaks import nms_topk_peaks  # noqa: E402
from dataset import read_sar_complex_tensor  # noqa: E402


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct SAR from 2-ch labels (A, alpha)")
    parser.add_argument("--config", default="A-ConvNets-FCN/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--labels",
        default=None,
        help="2-ch label root directory; defaults to data.label_2ch_root from config",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K peaks from A map for reconstruction")
    parser.add_argument("--nms-window", type=int, default=5, help="2D NMS window size (odd integer, >=3)")
    parser.add_argument("--amp-pctl", type=float, default=99.5, help="Percentile for adaptive threshold")
    parser.add_argument("--amp-pctl-scale", type=float, default=1.0, help="Scale factor for percentile threshold")
    parser.add_argument("--nz-only", action="store_true", help="Restrict peaks to strictly positive A pixels")
    parser.add_argument("--min-dist", type=int, default=3, help="Minimum pixel distance between selected peaks")
    parser.add_argument(
        "--save-dir",
        default=str(MODULE_ROOT / "outputs/recon_from_labels"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--compare-raw",
        action="store_true",
        help="If set, try to read corresponding raw and include in visualization",
    )
    return parser.parse_args()


def robust_vmax(values: np.ndarray, percentile: float = 99.5) -> float:
    if values.size == 0:
        return 1.0
    vmax = np.percentile(values, percentile)
    if vmax <= 0 or not np.isfinite(vmax):
        return 1.0
    return float(vmax)


def import_reconstruction_module(project_root: Path):
    import importlib

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return importlib.import_module("utils.reconstruction")


def collect_label_files(label_root: Path) -> List[Path]:
    if not label_root.exists() or not label_root.is_dir():
        return []
    return sorted(p for p in label_root.rglob("*.npy"))


def visualize_and_save(
    save_dir: Path,
    base_name: str,
    amp_map: np.ndarray,
    peaks: List[Tuple[int, int, float]],
    recon_mag: np.ndarray,
    raw_mag: Optional[np.ndarray] = None,
) -> None:
    save_dir = Path(ensure_dir(str(save_dir)))

    cols = 3 if raw_mag is not None else 2
    fig = plt.figure(figsize=(5 * cols, 5))

    idx = 1
    if raw_mag is not None:
        ax = plt.subplot(1, cols, idx)
        vmax = robust_vmax(raw_mag)
        ax.imshow(raw_mag, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title("Original |Z|", fontsize=10)
        ax.axis("off")
        idx += 1

    ax = plt.subplot(1, cols, idx)
    vmax = robust_vmax(amp_map)
    ax.imshow(amp_map, cmap="viridis", vmin=0, vmax=vmax)
    for r, c, _v in peaks:
        ax.plot(int(c), int(r), "r+", markersize=8, markeredgewidth=2)
    ax.set_title("Label A with peaks", fontsize=10)
    ax.axis("off")
    idx += 1

    ax = plt.subplot(1, cols, idx)
    vmax = robust_vmax(recon_mag)
    ax.imshow(recon_mag, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title("Reconstruction |Z|", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    out_path = save_dir / f"recon_from_label_{base_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve label root
    data_cfg = cfg.get("data", {})
    label_root = Path(args.labels) if args.labels else Path(str(data_cfg.get("label_2ch_root", "")))
    if not label_root.is_absolute():
        label_root = (Path(cfg["project_root"]) / label_root).resolve()

    # Try to import reconstruction module (with corrected pixel_to_model)
    try:
        rec_mod = import_reconstruction_module(Path(cfg["project_root"]))
        reconstruct_sar_image = rec_mod.reconstruct_sar_image
        pixel_to_model = rec_mod.pixel_to_model
    except Exception:
        if str(MODULE_ROOT) not in sys.path:
            sys.path.insert(0, str(MODULE_ROOT))
        from reconstruction_adapter import reconstruct_sar_image, pixel_to_model  # type: ignore

    files = collect_label_files(label_root)
    if not files:
        raise FileNotFoundError(f"No label files found under {label_root}")

    sar_root = Path(str(data_cfg.get("sar_root", "")))
    if not sar_root.is_absolute():
        sar_root = (Path(cfg["project_root"]) / sar_root).resolve()

    height = int(data_cfg.get("image_height", 128)) if isinstance(data_cfg, dict) else 128
    width = int(data_cfg.get("image_width", 128)) if isinstance(data_cfg, dict) else 128

    save_dir = Path(args.save_dir)
    ensure_dir(str(save_dir))

    for path in files:
        try:
            base = path.stem
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[0] < 2:
                print(f"[WARN] {path}: unexpected label shape {arr.shape}")
                continue

            amp_map_np = arr[0].astype(np.float32)
            alpha_map_np = arr[1].astype(np.float32)

            amp_map_t = torch.from_numpy(amp_map_np)

            peaks = nms_topk_peaks(
                amp_map_t,
                k=args.topk,
                threshold=0.0,  # use adaptive threshold below
                window_size=args.nms_window,
                percentile=args.amp_pctl,
                scale=args.amp_pctl_scale,
                nz_only=args.nz_only,
                min_distance=args.min_dist,
            )

            scatterers: List[Dict[str, float]] = []

            # Sub-pixel refinement using 3x3 quadratic peak interpolation (fallback to centroid)
            def subpixel_refine(a: np.ndarray, r: int, c: int) -> tuple[float, float]:
                h, w = a.shape
                r0 = max(1, min(h - 2, r))
                c0 = max(1, min(w - 2, c))
                # Local 3x3 patch
                patch = a[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2]
                # Weighted centroid in 3x3 window
                ys, xs = np.mgrid[-1:2, -1:2]
                wsum = patch.sum()
                if wsum <= 0:
                    return float(r0), float(c0)
                dy = float((ys * patch).sum() / wsum)
                dx = float((xs * patch).sum() / wsum)
                # Clamp offsets to [-1,1]
                dy = max(-1.0, min(1.0, dy))
                dx = max(-1.0, min(1.0, dx))
                return float(r0 + dy), float(c0 + dx)

            for row, col, _v in peaks:
                r_sub, c_sub = subpixel_refine(amp_map_np, int(row), int(col))
                x, y = pixel_to_model(r_sub, c_sub)
                scatterers.append(
                    {
                        "A": float(amp_map_np[int(round(r_sub)), int(round(c_sub))]),
                        "alpha": float(alpha_map_np[int(round(r_sub)), int(round(c_sub))]),
                        "x": float(x),
                        "y": float(y),
                        "gamma": 0.0,
                        "L": 0.0,
                        "phi_prime": 0.0,
                        "pixel_row": int(round(r_sub)),
                        "pixel_col": int(round(c_sub)),
                    }
                )

            recon_complex = reconstruct_sar_image(scatterers)
            recon_mag = np.abs(recon_complex)

            raw_mag = None
            if args.compare_raw:
                # Match .raw path by relative subdir and stem
                rel_dir = path.parent.relative_to(label_root)
                raw_path = sar_root / rel_dir / (path.stem + ".raw")
                if raw_path.exists():
                    raw_t = read_sar_complex_tensor(str(raw_path), height, width)
                    if raw_t is not None:
                        raw_mag = np.abs(raw_t.numpy())

            visualize_and_save(save_dir, base, amp_map_np, peaks, recon_mag, raw_mag)
            print(f"OK {base}: peaks={len(peaks)} saved → {save_dir / ('recon_from_label_' + base + '.png')}")
        except Exception as exc:
            print(f"[ERR] {path}: {exc}")


if __name__ == "__main__":
    main()
