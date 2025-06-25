# utils/enhanced_reconstruction.py - 改进的重建方法

import numpy as np
from scipy.signal.windows import taylor
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt


def optimized_reconstruct_sar_image(scatterers, enhancement_config=None):
    """
    优化的SAR图像重建函数，支持多种增强策略

    Args:
        scatterers: 散射中心列表
        enhancement_config: 增强配置字典
    """
    if enhancement_config is None:
        enhancement_config = {
            "amplitude_boost": 2.0,  # 幅度增强倍数
            "weak_scatterer_boost": 3.0,  # 弱散射中心额外增强
            "gaussian_smoothing": 0.5,  # 高斯平滑参数
            "taylor_window_sll": 30,  # Taylor窗副瓣电平
            "frequency_weighting": True,  # 是否使用频率加权
            "adaptive_scaling": True,  # 自适应幅度缩放
        }

    # SAR成像参数
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

    # 重建频域响应
    for i, fy in enumerate(fy_range):
        for j, fx in enumerate(fx_range):
            e_total = 0
            for s in scatterers:
                # 根据散射中心强度动态调整幅度
                amplitude = s["A"] * enhancement_config["amplitude_boost"]

                # 对弱散射中心进行额外增强
                if hasattr(s, "strength") and s.get("strength") == "low":
                    amplitude *= enhancement_config["weak_scatterer_boost"]

                # 自适应幅度缩放
                if enhancement_config["adaptive_scaling"]:
                    # 根据频率位置调整幅度
                    freq_weight = 1.0
                    if enhancement_config["frequency_weighting"]:
                        f_norm = np.sqrt(fx**2 + fy**2) / fc
                        freq_weight = 1.0 + 0.5 * f_norm  # 高频增强
                    amplitude *= freq_weight

                e_single = enhanced_model_rightangle_py(
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
                    amplitude,
                )
                e_total += e_single

            K_rect[i, j] = e_total

    K_rect = np.flipud(K_rect)

    # 改进的窗函数
    win = taylor(p, nbar=4, sll=enhancement_config["taylor_window_sll"])
    window_2d = np.outer(win, win)
    K_windowed = K_rect * window_2d

    # 零填充
    Z = np.zeros((q, q), dtype=np.complex128)
    start = (q - p) // 2
    Z[start : start + p, start : start + p] = K_windowed

    # IFFT重建
    img_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Z)))

    # 后处理增强
    if enhancement_config["gaussian_smoothing"] > 0:
        img_magnitude = np.abs(img_complex)
        img_phase = np.angle(img_complex)

        # 对幅度进行轻微平滑，保持相位
        smoothed_magnitude = gaussian_filter(img_magnitude, sigma=enhancement_config["gaussian_smoothing"])
        img_complex = smoothed_magnitude * np.exp(1j * img_phase)

    return img_complex


def enhanced_model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A):
    """
    增强版ASC模型，支持更精确的计算
    """
    C = 3e8
    f = np.sqrt(fx**2 + fy**2)
    o = np.arctan2(fy, fx)

    # 频率依赖项（增强版）
    E1 = A * (1j * f / fc) ** alpha

    # 空间相位项
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    # 分布散射中心的sinc函数
    if L > 0:
        sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
        E3 = np.sinc(sinc_arg / np.pi) if sinc_arg != 0 else 1.0
    else:
        E3 = 1.0

    # 距离依赖项
    E4 = np.exp(((-fy / fc) / (2 * np.sin(om_rad / 2))) * gamma)

    return E1 * E2 * E3 * E4


def apply_advanced_enhancement(img_complex, method="multi_scale"):
    """
    应用高级图像增强技术
    """
    img_magnitude = np.abs(img_complex)

    if method == "multi_scale":
        # 多尺度增强
        enhanced = np.zeros_like(img_magnitude)
        scales = [0.5, 1.0, 2.0]
        weights = [0.3, 0.4, 0.3]

        for scale, weight in zip(scales, weights):
            if scale != 1.0:
                # 缩放图像
                h, w = img_magnitude.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(img_magnitude, (new_w, new_h))
                scaled = cv2.resize(scaled, (w, h))  # 恢复原尺寸
            else:
                scaled = img_magnitude

            enhanced += weight * scaled

        return enhanced

    elif method == "clahe_advanced":
        # 高级CLAHE
        img_uint8 = ((img_magnitude / img_magnitude.max()) * 255).astype(np.uint8)

        # 多次应用不同参数的CLAHE
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        enhanced1 = clahe1.apply(img_uint8)
        enhanced2 = clahe2.apply(img_uint8)

        # 加权组合
        combined = 0.6 * enhanced1 + 0.4 * enhanced2
        return combined.astype(np.float32) / 255.0 * img_magnitude.max()

    elif method == "unsharp_mask":
        # 非锐化掩模
        gaussian_blur = gaussian_filter(img_magnitude, sigma=1.0)
        unsharp_mask = img_magnitude + 1.5 * (img_magnitude - gaussian_blur)
        return np.clip(unsharp_mask, 0, np.inf)

    return img_magnitude


def create_enhanced_visualization_comparison(original_img, gt_img, pred_imgs, titles, save_path):
    """
    创建增强的对比可视化
    """
    n_pred = len(pred_imgs)
    fig, axes = plt.subplots(2, max(3, n_pred), figsize=(6 * max(3, n_pred), 12))

    if n_pred < 3:
        # 如果预测图像少于3个，补齐到3个
        while len(pred_imgs) < 3:
            pred_imgs.append(np.zeros_like(pred_imgs[0]))
            titles.append("Empty")

    # 第一行：基础对比
    axes[0, 0].imshow(original_img, cmap="gray")
    axes[0, 0].set_title("Original SAR", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.abs(gt_img), cmap="gray")
    axes[0, 1].set_title("GT Reconstruction", fontsize=14)
    axes[0, 1].axis("off")

    for i, (pred_img, title) in enumerate(zip(pred_imgs[: min(n_pred, len(axes[0]) - 2)], titles)):
        if i + 2 < len(axes[0]):
            axes[0, i + 2].imshow(np.abs(pred_img), cmap="gray")
            axes[0, i + 2].set_title(title, fontsize=14)
            axes[0, i + 2].axis("off")

    # 第二行：增强效果
    for i in range(min(n_pred, len(axes[1]))):
        if i < len(pred_imgs):
            enhanced = apply_advanced_enhancement(pred_imgs[i], "clahe_advanced")
            axes[1, i].imshow(enhanced, cmap="hot")
            axes[1, i].set_title(f"Enhanced {titles[i]}", fontsize=14)
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# 预设的增强配置
ENHANCEMENT_CONFIGS = {
    "conservative": {
        "amplitude_boost": 1.5,
        "weak_scatterer_boost": 2.0,
        "gaussian_smoothing": 0.3,
        "taylor_window_sll": 35,
        "frequency_weighting": False,
        "adaptive_scaling": False,
    },
    "moderate": {
        "amplitude_boost": 2.0,
        "weak_scatterer_boost": 3.0,
        "gaussian_smoothing": 0.5,
        "taylor_window_sll": 30,
        "frequency_weighting": True,
        "adaptive_scaling": True,
    },
    "aggressive": {
        "amplitude_boost": 3.0,
        "weak_scatterer_boost": 5.0,
        "gaussian_smoothing": 0.7,
        "taylor_window_sll": 25,
        "frequency_weighting": True,
        "adaptive_scaling": True,
    },
}
