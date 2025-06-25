import numpy as np
import scipy.io as sio
from scipy.signal.windows import taylor
from scipy.ndimage import maximum_filter
from utils import config


def pixel_to_model(row, col):
    """
    The EXACT inverse of the model_to_pixel function.
    Converts integer pixel coordinates back to continuous physical model coordinates.
    """
    C1 = config.IMG_HEIGHT / (0.3 * config.P_GRID_SIZE)
    C2 = config.IMG_WIDTH / 2  # 64.0, This is correct for a 0-indexed system

    y_model = (C2 - row) / C1
    x_model = (col - C2) / C1
    return x_model, y_model


def extract_scatterers_from_mat(mat_path):
    """
    修正版：正确地从嵌套的MATLAB元胞数组中提取散射中心参数。
    """
    scatterers = []
    try:
        mat_data = sio.loadmat(mat_path)
        # 1. 直接获取最外层的 object 数组
        scatter_all_obj_array = mat_data["scatter_all"]

        # 2. 遍历这个 object 数组
        for cell in scatter_all_obj_array:
            # 3. 提取每个元胞中的参数数组 (通常在 cell[0][0])
            params = cell[0][0]

            # 4. 提取参数并添加到列表
            scatterers.append(
                {
                    "x": params[0],
                    "y": params[1],
                    "alpha": params[2],
                    "gamma": params[3],
                    "phi_prime": params[4],
                    "L": params[5],
                    "A": params[6],
                }
            )
    except (KeyError, IndexError, FileNotFoundError) as e:
        print(f"警告: 读取或解析 .mat 文件时出错 {mat_path}: {e}")
    return scatterers


def extract_scatterers_from_prediction_5ch(prediction_maps):
    """
    Extracts high-precision scatterers from the model's 5-channel output.
    This is the core of the new inference pipeline.
    """
    heatmap, pred_A_log1p, pred_alpha, pred_dx, pred_dy = [prediction_maps[i] for i in range(5)]

    # Apply np.expm1 to get back to the original amplitude scale
    pred_A = np.expm1(pred_A_log1p)

    # --- Peak Detection using Non-Maximum Suppression on the Heatmap ---
    # Find local maxima using a maximum filter
    neighborhood_size = 5  # A 5x5 window for NMS
    local_max = maximum_filter(heatmap, size=neighborhood_size) == heatmap

    # Apply a threshold to the heatmap to get candidate peaks
    candidates = (heatmap > 0.5) & local_max
    peak_coords = np.argwhere(candidates)

    scatterers = []
    for r, c in peak_coords:
        # 1. Get the integer pixel location
        x_base, y_base = pixel_to_model(r, c)

        # 2. Get the predicted sub-pixel offsets (dx, dy)
        # Note: The offset is in pixel units, so multiply by pixel spacing
        dx = pred_dx[r, c] * config.PIXEL_SPACING
        dy = pred_dy[r, c] * config.PIXEL_SPACING

        # 3. Calculate the final high-precision physical coordinates
        x_final = x_base + dx
        y_final = y_base - dy  # In our coordinate system, row offset is negative y

        # 4. Get the predicted parameters
        A_final = pred_A[r, c]
        alpha_final = pred_alpha[r, c]

        # The model only predicts localized scatterers
        scatterers.append(
            {"A": A_final, "alpha": alpha_final, "x": x_final, "y": y_final, "gamma": 0, "L": 0, "phi_prime": 0}
        )

    return scatterers


def model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A):
    """Python implementation of the core ASC model from model_rightangle.m"""
    C = 3e8
    f = np.sqrt(fx**2 + fy**2)
    o = np.arctan2(fy, fx)

    E1 = A * (1j * f / fc) ** alpha
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
    E3 = np.sinc(sinc_arg / np.pi) if L > 0 else 1.0

    E4 = np.exp(((-fy / fc) / (2 * np.sin(om_rad / 2))) * gamma)

    return E1 * E2 * E3 * E4


def reconstruct_sar_image(scatterers):
    """Reconstructs a SAR image from a list of ASCs."""
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
            e_total = sum(
                model_rightangle_py(
                    om_rad,
                    fx,
                    fy,
                    fc,
                    s["x"],
                    s["y"],
                    s["alpha"],
                    s["gamma"],
                    s["phi_prime"] * np.pi / 180.0,
                    s["L"],
                    s["A"],
                )
                for s in scatterers
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


def get_enhanced_reconstruction_configs():
    """
    获取不同的增强配置，避免导入时执行
    """
    try:
        from utils.enhanced_reconstruction import optimized_reconstruct_sar_image, ENHANCEMENT_CONFIGS

        return optimized_reconstruct_sar_image, ENHANCEMENT_CONFIGS
    except ImportError:
        print("警告：无法导入增强重建模块，将使用标准重建方法")
        return None, None


def enhanced_reconstruct_sar_image(scatterers, config_name="moderate"):
    """
    使用增强配置重建SAR图像的便捷函数
    """
    optimized_func, configs = get_enhanced_reconstruction_configs()
    if optimized_func is not None and configs is not None:
        return optimized_func(scatterers, configs.get(config_name, configs["moderate"]))
    else:
        # 回退到标准重建方法
        return reconstruct_sar_image(scatterers)
