"""
Lightweight reconstruction utilities for A-ConvNets-FCN

Provides pixel_to_model (with configurable geometry) and reconstruct_sar_image
so that reconstruction can run without importing the project-level utils package.
"""

from typing import List, Dict

import numpy as np
from scipy.signal.windows import taylor


def pixel_to_model(row: int, col: int, img_height: int = 128, img_width: int = 128, p_grid_size: int = 84):
    """
    Convert integer pixel (row, col) to physical model coordinates (x, y).

    This mirrors utils.reconstruction.pixel_to_model with configurable constants.
    """
    C1 = img_height / (0.3 * p_grid_size)
    C2_y = (img_height - 1) / 2.0
    C2_x = (img_width - 1) / 2.0
    y_model = (C2_y - float(row)) / C1
    x_model = (float(col) - C2_x) / C1
    return x_model, y_model


def _model_rightangle_py(
    om_rad: float,
    fx: float,
    fy: float,
    fc: float,
    x: float,
    y: float,
    alpha: float,
    gamma: float,
    phi_prime_rad: float,
    L: float,
    A: float,
):
    C = 3e8
    f = np.sqrt(fx**2 + fy**2)
    o = np.arctan2(fy, fx)

    E1 = A * (1j * f / fc) ** alpha
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
    E3 = np.sinc(sinc_arg / np.pi) if L > 0 else 1.0

    E4 = np.exp(((-fy / fc) / (2 * np.sin(om_rad / 2))) * gamma)

    return E1 * E2 * E3 * E4


def reconstruct_sar_image(scatterers: List[Dict[str, float]]):
    """Reconstruct a SAR image from ASC parameter list.

    scatterers entries: {"A","alpha","x","y","gamma","L","phi_prime"}
    Returns complex image (numpy.ndarray, shape [q,q]) whose magnitude can be visualized.
    """
    fc = 1e10
    B = 5e8
    om_deg = 2.86
    p = 84
    q = 128

    om_rad = om_deg * np.pi / 180.0
    bw_ratio = B / fc

    fx_range = np.linspace(fc * (1 - bw_ratio / 2), fc * (1 + bw_ratio / 2), p)
    fy_range = np.linspace(-fc * np.sin(om_rad / 2), fc * np.sin(om_rad / 2), p)

    K_rect = np.zeros((p, p), dtype=np.complex128)

    for i, fy in enumerate(fy_range):
        for j, fx in enumerate(fx_range):
            e_total = 0j
            for s in scatterers:
                e_total += _model_rightangle_py(
                    om_rad,
                    fx,
                    fy,
                    fc,
                    s["x"],
                    s["y"],
                    s["alpha"],
                    s.get("gamma", 0.0),
                    np.deg2rad(s.get("phi_prime", 0.0)),
                    s.get("L", 0.0),
                    s["A"],
                )
            K_rect[i, j] = e_total

    K_rect = np.flipud(K_rect)

    win = taylor(p, nbar=4, sll=35)
    window_2d = np.outer(win, win)
    K_windowed = K_rect * window_2d

    Z = np.zeros((q, q), dtype=np.complex128)
    start = (q - p) // 2
    Z[start : start + p, start : start + p] = K_windowed

    img_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Z)))
    return img_complex
