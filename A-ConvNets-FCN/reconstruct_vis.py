"""
A-ConvNets-FCN Reconstruction Visualization

Purpose:
- Load the trained A-ConvNets-FCN model.
- Predict 2-channel maps (Amplitude A, Alpha) from SAR .raw files.
- Detect peak scatterers from A map, convert (row,col) → (x,y) with project reconstruction utils.
- Reconstruct SAR image from detected scatterers and save a side-by-side visualization.

Usage:
  python A-ConvNets-FCN/reconstruct_vis.py \
    --config A-ConvNets-FCN/config.yaml \
    --checkpoint A-ConvNets-FCN/outputs/checkpoints/aconv_fcn_best.pt \
    [--raw <file-or-dir>] [--device auto] [--topk 5] [--amp-thr 0.0] \
    [--save-dir A-ConvNets-FCN/outputs/recon_vis]
    
python A-ConvNets-FCN/reconstruct_vis.py --config config.yaml --checkpoint A-ConvNets-FCN/outputs/checkpoints/aconv_fcn_best.pt --raw datasets/SAR_ASC_Project/tmp_Data_Processed_raw --topk 5 --amp-thr 0.0 --save-dir A-ConvNets-FCN/outputs/recon_vis

python A-ConvNets-FCN/reconstruct_vis.py --config config.yaml --checkpoint A-ConvNets-FCN/outputs/checkpoints/aconv_fcn_best.pt --raw datasets/SAR_ASC_Project/tmp_Data_Processed_raw --topk 80 --nms-window 5 --amp-pctl 99.5 --amp-pctl-scale 1.0 --save-dir A-ConvNets-FCN/outputs/recon_vis
"""

import os

# Avoid OpenMP duplicate runtime issues on Windows (MKL/oneMKL vs PyTorch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure we can import A-ConvNets-FCN local modules
MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from dataset import read_sar_complex_tensor  # noqa: E402
from model import AConvFCN  # noqa: E402
from utils.io import ensure_dir, load_config  # noqa: E402
from utils.peaks import nms_topk_peaks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-ConvNets-FCN reconstruction visualization")
    parser.add_argument("--config", default="A-ConvNets-FCN/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--checkpoint",
        default=str(MODULE_ROOT / "outputs/checkpoints/aconv_fcn_best.pt"),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--raw",
        default=None,
        help=".raw file or directory; if omitted, uses data.sar_root from config",
    )
    parser.add_argument("--device", default="auto", help="cuda|cpu|auto")
    parser.add_argument("--topk", type=int, default=5, help="Top-K peaks from amplitude map for reconstruction")
    parser.add_argument("--amp-thr", type=float, default=0.0, help="Amplitude threshold for peak selection")
    parser.add_argument("--nms-window", type=int, default=5, help="2D NMS window size (odd integer, >=3)")
    parser.add_argument(
        "--amp-pctl",
        type=float,
        default=99.5,
        help="Adaptive threshold percentile for A map when --amp-thr<=0",
    )
    parser.add_argument(
        "--amp-pctl-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to percentile threshold",
    )
    parser.add_argument(
        "--save-dir",
        default=str(MODULE_ROOT / "outputs/recon_vis"),
        help="Output directory for visualizations",
    )
    return parser.parse_args()


def resolve_device(arg_device: str) -> torch.device:
    if arg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg_device)


def load_model(ckpt_path: Path, cfg: Dict[str, object], device: torch.device) -> AConvFCN:
    model = AConvFCN(
        in_channels=cfg.get("model", {}).get("in_channels", 1),
        out_channels=cfg.get("model", {}).get("out_channels", 2),
        kernel_size=cfg.get("model", {}).get("conv_kernel", 5),
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = (
        checkpoint.get("model_state")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint.get("state_dict") if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resolve_raw_path(raw_arg: Optional[str], cfg: Dict[str, object]) -> Path:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    default_root = Path(str(data_cfg.get("sar_root", ""))) if data_cfg else Path()

    if raw_arg is None:
        if default_root and default_root.exists():
            return default_root.resolve()
        raise FileNotFoundError("No --raw provided and data.sar_root not found in config")

    raw_path = Path(raw_arg)
    candidates: List[Path] = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        project_root = Path(str(cfg.get("project_root", ""))) if isinstance(cfg, dict) else Path()
        if project_root:
            candidates.append(project_root / raw_path)
        if default_root:
            candidates.append(default_root / raw_path)
            candidates.append(default_root.parent / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate input path '{raw_arg}'. Checked: " + ", ".join(str(p) for p in candidates)
    )


def collect_raw_files(raw_path: Path) -> List[Path]:
    if raw_path.is_file():
        return [raw_path]
    if raw_path.is_dir():
        return sorted(p for p in raw_path.rglob("*.raw"))
    raise FileNotFoundError(f"Input path does not exist: {raw_path}")


def robust_vmax(values: np.ndarray, percentile: float = 99.5) -> float:
    if values.size == 0:
        return 1.0
    vmax = np.percentile(values, percentile)
    if vmax <= 0 or not np.isfinite(vmax):
        return 1.0
    return float(vmax)


def import_reconstruction_module(project_root: Path):
    """Import project-level utils.reconstruction ensuring correct utils package is used."""
    import importlib

    # Prepend project_root so that 'import utils' resolves to project utils (not A-ConvNets-FCN/utils)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Now import the standard package module
    return importlib.import_module("utils.reconstruction")


def detect_scatterers(
    amp_map: torch.Tensor,
    alpha_map: torch.Tensor,
    topk: int,
    amp_thr: float,
    pixel_to_model_fn,
    nms_window: int,
    amp_percentile: float,
    amp_scale: float,
) -> Tuple[List[Dict[str, float]], List[Tuple[int, int, float]]]:
    """Select peaks from amplitude map and produce scatterers for reconstruction.

    Returns (scatterers, peaks_for_plot)
    """
    peaks: List[Tuple[int, int, float]] = nms_topk_peaks(
        amp_map,
        k=topk,
        threshold=amp_thr,
        window_size=nms_window,
        percentile=amp_percentile,
        scale=amp_scale,
    )
    scatterers: List[Dict[str, float]] = []
    peaks_plot: List[Tuple[int, int, float]] = []

    # Sub-pixel refinement using 3x3 weighted centroid
    amp_np = amp_map.detach().cpu().numpy()
    alpha_np = alpha_map.detach().cpu().numpy()

    for row, col, _val in peaks:
        row_i = int(row)
        col_i = int(col)
        # refine on 3x3 patch around (row_i, col_i)
        h, w = amp_np.shape
        r0 = max(1, min(h - 2, row_i))
        c0 = max(1, min(w - 2, col_i))
        patch = amp_np[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2]
        ys, xs = np.mgrid[-1:2, -1:2]
        wsum = patch.sum()
        if wsum > 0:
            dy = float((ys * patch).sum() / wsum)
            dx = float((xs * patch).sum() / wsum)
            dy = max(-1.0, min(1.0, dy))
            dx = max(-1.0, min(1.0, dx))
            r_sub = float(r0 + dy)
            c_sub = float(c0 + dx)
        else:
            r_sub = float(r0)
            c_sub = float(c0)

        # physical coordinates
        x, y = pixel_to_model_fn(r_sub, c_sub)
        rr = int(round(r_sub))
        cc = int(round(c_sub))
        A_val = float(amp_np[rr, cc])
        alpha_val = float(alpha_np[rr, cc])
        scatterers.append(
            {
                "A": A_val,
                "alpha": alpha_val,
                "x": float(x),
                "y": float(y),
                "gamma": 0.0,
                "L": 0.0,
                "phi_prime": 0.0,
                "pixel_row": rr,
                "pixel_col": cc,
            }
        )
        peaks_plot.append((rr, cc, A_val))

    return scatterers, peaks_plot


def visualize_and_save(
    save_dir: Path,
    base_name: str,
    original_mag: np.ndarray,
    amp_map: np.ndarray,
    peaks: List[Tuple[int, int, float]],
    recon_mag: np.ndarray,
) -> None:
    save_dir = Path(ensure_dir(str(save_dir)))
    fig = plt.figure(figsize=(15, 5))

    # Original magnitude
    ax1 = plt.subplot(1, 3, 1)
    vmax1 = robust_vmax(original_mag)
    ax1.imshow(original_mag, cmap="gray", vmin=0, vmax=vmax1)
    ax1.set_title("Original | |Z|", fontsize=10)
    ax1.axis("off")

    # Amplitude map with peaks
    ax2 = plt.subplot(1, 3, 2)
    vmax2 = robust_vmax(amp_map)
    ax2.imshow(amp_map, cmap="viridis", vmin=0, vmax=vmax2)
    for row, col, _v in peaks:
        ax2.plot(int(col), int(row), "r+", markersize=8, markeredgewidth=2)
    ax2.set_title("Pred A with peaks", fontsize=10)
    ax2.axis("off")

    # Reconstructed magnitude
    ax3 = plt.subplot(1, 3, 3)
    vmax3 = robust_vmax(recon_mag)
    ax3.imshow(recon_mag, cmap="gray", vmin=0, vmax=vmax3)
    ax3.set_title("Reconstruction |Z|", fontsize=10)
    ax3.axis("off")

    plt.tight_layout()
    out_path = save_dir / f"recon_vis_{base_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)

    raw_root = resolve_raw_path(args.raw, cfg)
    raw_files = collect_raw_files(raw_root)
    if not raw_files:
        raise FileNotFoundError(f"No .raw files found under {raw_root}")

    model = load_model(Path(args.checkpoint), cfg, device)

    # Import project reconstruction utilities safely
    try:
        rec_mod = import_reconstruction_module(Path(cfg["project_root"]))
        reconstruct_sar_image = rec_mod.reconstruct_sar_image
        pixel_to_model = rec_mod.pixel_to_model
    except Exception:
        # Fallback to local lightweight adapter if project-level utils cannot be imported
        if str(MODULE_ROOT) not in sys.path:
            sys.path.insert(0, str(MODULE_ROOT))
        from reconstruction_adapter import reconstruct_sar_image, pixel_to_model  # type: ignore

    save_dir = Path(args.save_dir)
    ensure_dir(str(save_dir))

    height = int(cfg.get("data", {}).get("image_height", 128))
    width = int(cfg.get("data", {}).get("image_width", 128))

    for raw_file in raw_files:
        base = raw_file.stem
        try:
            # Read raw → complex → magnitude for reference
            raw_tensor = read_sar_complex_tensor(str(raw_file), height, width)
            if raw_tensor is None:
                print(f"[WARN] {raw_file}: failed to read or shape mismatch")
                continue

            original_mag = np.abs(raw_tensor.numpy())

            # Model inference
            image = raw_tensor.abs().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(image)[0].cpu()

            amp_map_t = pred[0]
            alpha_map_t = pred[1]

            # Detect peaks and build scatterers
            scatterers, peaks = detect_scatterers(
                amp_map_t,
                alpha_map_t,
                args.topk,
                args.amp_thr,
                pixel_to_model,
                args.nms_window,
                args.amp_pctl,
                args.amp_pctl_scale,
            )

            # Reconstruct
            recon_complex = reconstruct_sar_image(scatterers)
            recon_mag = np.abs(recon_complex)

            # Visualization uses refined peak positions
            visualize_and_save(
                save_dir,
                base,
                original_mag,
                amp_map_t.numpy(),
                peaks,
                recon_mag,
            )

            print(f"OK {base}: peaks={len(scatterers)} saved → {save_dir / ('recon_vis_' + base + '.png')}")
        except Exception as exc:
            print(f"[ERR] {raw_file}: {exc}")


if __name__ == "__main__":
    main()
