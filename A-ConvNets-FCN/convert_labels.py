import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from utils.io import ensure_dir, load_config  # noqa: E402

"""
功能
----
从原始 `*_yang.mat`（变量 `scatter_all`）直接生成符合 A-ConvNets-FCN 模型训练需求的 2 通道标签：
- 通道0: 线性幅度 A（仅在离散散射像素处为正，其余为 0）
- 通道1: alpha（仅在离散散射像素处为值，其余为 0）

与旧流程（5ch→2ch）不同，本脚本直接从原始参数生成，避免中间换算误差和坐标偏差。

输入/输出
--------
- MAT 根目录：默认取项目 `utils.config.ASC_MAT_ROOT`，也可通过 `--mat-root` 指定
- 输出根目录：`config.yaml:data.label_2ch_root`

坐标转换
--------
- 使用与训练/可视化一致的中心定义：y 轴以 H/2 为中心、x 轴以 W/2 为中心
- row = (H-1)/2 - y*C1, col = x*C1 + (W-1)/2, 其中 C1 = H / (0.3 * P_GRID_SIZE)
 python A-ConvNets-FCN/convert_labels.py --config config.yaml
"""


def model_to_pixel(x_model: float, y_model: float, height: int, width: int, p_grid_size: int) -> tuple[int, int]:
    C1 = height / (0.3 * p_grid_size)
    c_row = (height - 1) / 2.0
    c_col = (width - 1) / 2.0
    row_f = c_row - (y_model * C1)
    col_f = c_col + (x_model * C1)
    return int(round(row_f)), int(round(col_f))


def read_scatterers_from_mat(mat_path: Path) -> np.ndarray:
    """Load scatterers from *_yang.mat, return Nx7 array [x,y,alpha,gamma,phi_prime,L,A]."""
    data = sio.loadmat(mat_path)
    cell = data.get("scatter_all")
    if cell is None or cell.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    try:
        arr = np.vstack([c[0] for c in cell])
    except Exception:
        # Fallback: sometimes nested differently
        items = []
        for c in cell:
            v = c[0] if isinstance(c, np.ndarray) else c
            v = np.array(v).reshape(-1)
            if v.size >= 7:
                items.append(v[:7])
        arr = np.vstack(items) if items else np.empty((0, 7), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _import_project_config_by_path(project_root: Path):
    """Import project-level utils/config.py by absolute path to avoid name collisions.

    Returns a module object exposing IMG_HEIGHT, IMG_WIDTH, P_GRID_SIZE, ASC_MAT_ROOT.
    """
    import importlib.util

    cfg_path = project_root / "utils" / "config.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"project utils/config.py not found at {cfg_path}")
    spec = importlib.util.spec_from_file_location("project_utils_config", str(cfg_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for project utils.config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_from_mat_to_2ch(config_path: str, mat_root_override: str | None = None) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    label_2ch_root = Path(str(data_cfg.get("label_2ch_root", "")))
    ensure_dir(str(label_2ch_root))

    # Import project-level numeric constants by absolute path to avoid utils package collision
    project_root = Path(cfg["project_root"]).resolve()
    try:
        prj_config = _import_project_config_by_path(project_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to import project utils.config by path: {exc}")

    height = int(getattr(prj_config, "IMG_HEIGHT", int(data_cfg.get("image_height", 128))))
    width = int(getattr(prj_config, "IMG_WIDTH", int(data_cfg.get("image_width", 128))))
    p_grid_size = int(getattr(prj_config, "P_GRID_SIZE", 84))

    # Resolve MAT root
    default_mat_root = Path(getattr(prj_config, "ASC_MAT_ROOT", ""))
    mat_root = Path(mat_root_override) if mat_root_override else default_mat_root
    if not mat_root.is_absolute():
        mat_root = (project_root / mat_root).resolve()
    if not mat_root.exists():
        raise FileNotFoundError(f"MAT root does not exist: {mat_root}")

    print(f"MAT root resolved to: {mat_root}")

    converted = 0
    skipped = 0
    scanned = 0

    for dirpath, _, filenames in os.walk(mat_root):
        for filename in filenames:
            if not filename.lower().endswith(".mat"):
                continue
            mat_path = Path(dirpath) / filename
            try:
                scat = read_scatterers_from_mat(mat_path)
                scanned += 1
                if scat.shape[0] == 0:
                    skipped += 1
                    # Verbose hint for troubleshooting
                    print(f"[INFO] No scatterers found in {mat_path.name} (missing 'scatter_all' or empty).")
                    continue

                label = np.zeros((2, height, width), dtype=np.float32)

                for row in scat:
                    x_m, y_m, alpha, A = float(row[0]), float(row[1]), float(row[2]), float(row[6])
                    r, c = model_to_pixel(x_m, y_m, height, width, p_grid_size)
                    if 0 <= r < height and 0 <= c < width:
                        # Keep the stronger scatterer if collisions happen at the same pixel
                        if A > label[0, r, c]:
                            label[0, r, c] = A
                            label[1, r, c] = alpha

                rel_dir = Path(dirpath).relative_to(mat_root)
                out_dir = Path(ensure_dir(str(label_2ch_root / rel_dir)))
                out_path = out_dir / (filename.replace("_yang.mat", ".npy"))
                np.save(out_path, label)
                converted += 1
            except Exception as exc:
                print(f"[WARN] Failed to convert {mat_path}: {exc}")
                skipped += 1

    print(
        f"Scanned {scanned} .mat files under {mat_root}. Converted {converted}, Skipped {skipped}.\nOutput root: {label_2ch_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2-ch labels (A, alpha) directly from *_yang.mat")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mat-root", default=None, help="Override MAT root directory (optional)")
    args = parser.parse_args()
    convert_from_mat_to_2ch(args.config, args.mat_root)


if __name__ == "__main__":
    main()
