import numpy as np
from scipy.interpolate import griddata
from scipy.signal.windows import taylor
from skimage.feature import peak_local_max
from utils import config


def extract_scatterers_from_maps(A_map, alpha_map, is_gt=False, min_distance=2, threshold_abs=0.1):
    """
    Extracts discrete scatterer parameters from dense maps.

    Args:
        A_map (np.ndarray): The amplitude map.
        alpha_map (np.ndarray): The alpha map.
        is_gt (bool): If True, treats the map as ground truth (all non-zero points are scatterers).
        min_distance (int): Minimum distance between peaks for predicted maps.
        threshold_abs (float): Minimum amplitude to be considered a peak for predicted maps.

    Returns:
        list: A list of dictionaries, each representing a scatterer.
    """
    scatterers = []
    center_y, center_x = [d // 2 for d in A_map.shape]

    if is_gt:
        # For ground truth, which is sparse, every non-zero point is a scatterer.
        coordinates = np.argwhere(A_map > 0)
    else:
        # For model predictions, which are dense, we find distinct peaks.
        coordinates = peak_local_max(A_map, min_distance=min_distance, threshold_abs=threshold_abs)

    for r, c in coordinates:
        y_m = (r - center_y) * config.PIXEL_SPACING
        x_m = (c - center_x) * config.PIXEL_SPACING

        scatterers.append(
            {"A": A_map[r, c], "alpha": alpha_map[r, c], "x": x_m, "y": y_m, "gamma": 0, "L": 0, "phi_prime": 0}
        )
    return scatterers


def model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A):
    """Python implementation of the core ASC model from model_rightangle.m"""
    C = 3e8
    f = np.sqrt(fx**2 + fy**2)
    o = np.arctan2(fy, fx)

    E1 = (1j * f / fc) ** alpha
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
    E3 = np.sinc(sinc_arg / np.pi) if L > 0 else 1.0

    E4 = np.exp(((-fy / fc) / (2 * np.sin(om_rad / 2))) * gamma)

    return A * E1 * E2 * E3 * E4


def reconstruct_sar_image(scatterers, img_params):
    """Reconstructs a SAR image from ASCs, closely matching spotlight.m"""
    fc = 1e10
    B = 5e8
    om_deg = 2.86
    p = 84
    q = 128

    om_rad = om_deg * np.pi / 180.0
    bw_ratio = B / fc

    fx1, fx2 = (1 - bw_ratio / 2) * fc, (1 + bw_ratio / 2) * fc
    fy1, fy2 = -fc * np.sin(om_rad / 2), fc * np.sin(om_rad / 2)

    fx_range = np.linspace(fx1, fx2, p)
    fy_range = np.linspace(fy1, fy2, p)

    K_rect = np.zeros((p, p), dtype=np.complex128)

    for i, fy in enumerate(fy_range):
        for j, fx in enumerate(fx_range):
            e_total = 0
            for s in scatterers:
                e_total += model_rightangle_py(
                    om_rad, fx, fy, fc, s["x"], s["y"], s["alpha"], s["gamma"], s["phi_prime"], s["L"], s["A"]
                )
            K_rect[i, j] = e_total

    K_rect = np.flipud(K_rect)

    win = taylor(p, nbar=4, sll=35)
    window_2d = np.outer(win, win)
    K_windowed = K_rect * window_2d

    Z = np.zeros((q, q), dtype=np.complex128)
    pad_pre = (q - p) // 2
    pad_post = q - p - pad_pre
    Z = np.pad(K_windowed, ((pad_pre, pad_post), (pad_pre, pad_post)), "constant")

    img_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Z)))

    return img_complex
