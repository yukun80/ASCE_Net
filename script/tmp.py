import os
import numpy as np
import scipy.io as sio
from scipy.signal.windows import taylor
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# =============================================================================
# 1. 全局雷达参数 (来自你的 spotlight.m)
# =============================================================================
FC = 1e10  # 中心频率 (Hz)
B = 5e8  # 带宽 (Hz)
OM = 2.86  # 观测角 (degrees)
P = 84  # K-space处理网格大小
Q = 128  # 最终图像大小 (补零后)
C = 3e8  # 光速 (m/s)


def model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A):
    """
    Python实现 model_rightangle.m 的功能，此函数无需修改。
    """
    f = np.sqrt(fx**2 + fy**2)
    o = np.arctan2(fy, fx)

    E1 = (1j * f / fc) ** alpha
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
    E3 = np.sinc(sinc_arg / np.pi) if sinc_arg != 0 else 1.0

    E4 = np.exp(((-fy / fc) / (2 * np.sin(om_rad / 2))) * gamma)

    return A * E1 * E2 * E3 * E4


def spotlight_py(asc_params, fc, b, om, p, q):
    """
    Python实现 spotlight.m 的功能，修改了参数解析部分。
    """
    # ==========================================================
    # !!! 核心修改：根据新的理解解析8个参数 !!!
    # ==========================================================
    # 假设参数顺序为: [幅度模, 幅度相位(rad), x, y, α, L, φ'(deg), γ]
    A_mag = asc_params[0]
    A_phase_rad = asc_params[1]
    x = asc_params[2]
    y = asc_params[3]
    alpha = asc_params[4]
    L = asc_params[5]
    phi_prime_deg = asc_params[6]
    gamma = asc_params[7]

    # 1. 重建复数幅度 A
    A = A_mag * np.exp(1j * A_phase_rad)

    # ==========================================================
    # !!! 镜面反转调试区 !!!
    # ==========================================================
    # 之前的分析表明，镜面反转很可能是y坐标的符号问题。
    # 请取消下面这行代码的注释来修正它。
    y = -y
    # ==========================================================

    om_rad = om * np.pi / 180.0
    phi_prime_rad = phi_prime_deg * np.pi / 180.0  # 将角度制的φ'转为弧度

    bw_ratio = b / fc
    fx1 = (1 - bw_ratio / 2) * fc
    fx2 = (1 + bw_ratio / 2) * fc
    fy1 = -fc * np.sin(om_rad / 2)
    fy2 = fc * np.sin(om_rad / 2)

    fx_range = np.linspace(fx1, fx2, p)
    fy_range = np.linspace(fy1, fy2, p)

    K = np.zeros((p, p), dtype=np.complex128)

    for i, fy in enumerate(fy_range):
        for j, fx in enumerate(fx_range):
            K[i, j] = model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A)

    K = np.flipud(K)

    win = taylor(p, nbar=3, sll=35)
    win_2d = np.outer(win, win)
    K = K * win_2d

    Z = np.zeros((q, q), dtype=np.complex128)
    Z[:p, :p] = K

    return Z


def reconstruct_image_from_asc(asc_feature_matrix):
    """
    主重建函数，此函数无需修改。
    """
    num_scatterers = asc_feature_matrix.shape[1]

    K_total = np.zeros((Q, Q), dtype=np.complex128)

    for i in range(num_scatterers):
        asc_params = asc_feature_matrix[:, i]
        K_temp = spotlight_py(asc_params, FC, B, OM, P, Q)
        K_total += K_temp

    image_complex = fftshift(ifft2(K_total))
    image_reconstructed = np.abs(image_complex)

    return image_reconstructed


# =============================================================================
# 3. 主程序入口
# =============================================================================
if __name__ == "__main__":
    # --- 用户配置区 ---
    asc_dataset_root = (
        r"E:\Document\paper_library\3rd_paper_250512\dataset\MSTAR-ASC\10 categories\test\T72(SN_132)\ASCM_FEATURE"
    )
    output_root = r"E:\Document\paper_library\3rd_paper_250512\dataset\MSTAR-ASC\10 categories\test\T72(SN_132)\My_Reconstructions_v2"  # 建议用新文件夹名以区分
    # --------------------------

    os.makedirs(output_root, exist_ok=True)
    print(f"输出目录已确认: {output_root}")

    if not os.path.isdir(asc_dataset_root):
        print(f"错误: 输入目录 '{asc_dataset_root}' 不存在或不是一个有效的目录。")
    else:
        mat_files = [f for f in os.listdir(asc_dataset_root) if f.lower().endswith(".mat")]

        if not mat_files:
            print(f"警告: 在目录 '{asc_dataset_root}' 中没有找到任何.mat文件。")
        else:
            print(f"开始批量重建 {len(mat_files)} 个文件...")

            for filename in mat_files:
                input_path = os.path.join(asc_dataset_root, filename)
                output_filename = os.path.splitext(filename)[0] + ".jpg"
                output_path = os.path.join(output_root, output_filename)

                try:
                    asc_mat = sio.loadmat(input_path)
                    ascm_feature = asc_mat["ASCM_feature"]

                    reconstructed_image = reconstruct_image_from_asc(ascm_feature)

                    # 归一化图像亮度以便于保存和查看
                    img_max = reconstructed_image.max()
                    if img_max > 0:
                        reconstructed_image_normalized = (reconstructed_image / img_max * 255).astype(np.uint8)
                    else:
                        reconstructed_image_normalized = reconstructed_image.astype(np.uint8)

                    plt.imsave(output_path, reconstructed_image_normalized, cmap="gray", format="jpg")

                    print(f"  [√] 已重建并保存: {output_filename}")

                except Exception as e:
                    print(f"  [X] 处理 {filename} 时发生错误: {e}")

            print("批量重建完成！")
