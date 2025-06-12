import os
import numpy as np
import scipy.io as sio
from scipy.signal.windows import taylor
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# =============================================================================
# 1. 全局雷达参数 (来自你的 spotlight.m)
# =============================================================================
# MSTAR数据集的标准参数
FC = 1e10  # 中心频率 (Hz)
B = 5e8  # 带宽 (Hz)
OM = 2.86  # 观测角 (degrees)
P = 84  # K-space处理网格大小
Q = 128  # 最终图像大小 (补零后)
C = 3e8  # 光速 (m/s)


def model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A):
    """
    Python实现 model_rightangle.m 的功能
    这是ASC模型的核心物理方程。
    """
    f = np.sqrt(fx**2 + fy**2)
    # 避免除以零
    o = np.arctan2(fy, fx)

    # E1: 频率依赖项
    E1 = (1j * f / fc) ** alpha

    # E2: 位置项
    E2 = np.exp(-1j * 4 * np.pi * f / C * (x * np.cos(o) + y * np.sin(o)))

    # E3: 分布式散射体项 (长度和方向)
    # np.sinc(x) 计算的是 sin(pi*x)/(pi*x)，而MATLAB的sinc(x)是sin(x)/x
    # 我们需要自己实现 sin(x)/x
    sinc_arg = (2 * f * L / C) * np.sin(o - phi_prime_rad)
    # 避免除以零
    E3 = np.sin(sinc_arg) / sinc_arg if sinc_arg != 0 else 1.0

    # E4: 局部散射体项 (角度依赖)
    # 严格按照你的MATLAB代码实现
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

    # 计算频率网格范围
    bw_ratio = b / fc
    fx1 = (1 - bw_ratio / 2) * fc
    fx2 = (1 + bw_ratio / 2) * fc
    fy1 = -fc * np.sin(om_rad / 2)
    fy2 = fc * np.sin(om_rad / 2)

    # 创建频率网格
    fx_range = np.linspace(fx1, fx2, p)
    fy_range = np.linspace(fy1, fy2, p)

    # 初始化K-space矩阵
    K = np.zeros((p, p), dtype=np.complex128)

    # 填充K-space矩阵
    for i, fy in enumerate(fy_range):
        for j, fx in enumerate(fx_range):
            K[i, j] = model_rightangle_py(om_rad, fx, fy, fc, x, y, alpha, gamma, phi_prime_rad, L, A)

    # 你的代码中是flipud，对应numpy的np.flipud
    K = np.flipud(K)

    # 加窗 (来自你的 taylorwin(p, 3, -35))
    # nbar=3, sll=35. 注意scipy的sll是正数
    win = taylor(p, nbar=3, sll=35)
    win_2d = np.outer(win, win)
    K = K * win_2d

    # 补零到最终图像大小
    Z = np.zeros((q, q), dtype=np.complex128)
    Z[:p, :p] = K

    return Z


def reconstruct_image_from_asc(asc_feature_matrix):
    """
    主重建函数，等同于 simulation.m 的逻辑
    """
    num_scatterers = asc_feature_matrix.shape[1]
    print(f"找到 {num_scatterers} 个散射中心, 开始重建...")

    # 初始化总的K-space
    K_total = np.zeros((Q, Q), dtype=np.complex128)

    # 循环累加每个散射中心的贡献
    for i in range(num_scatterers):
        asc_params = asc_feature_matrix[:, i]
        K_temp = spotlight_py(asc_params, FC, B, OM, P, Q)
        K_total += K_temp

    # 从频域到图像域
    # 注意：IFFT之后需要进行fftshift才能将零频分量移到中心
    image_complex = fftshift(ifft2(K_total))

    # 取幅度得到最终图像
    image_reconstructed = np.abs(image_complex)

    return image_reconstructed


# =============================================================================
# 3. 主程序入口
# =============================================================================
if __name__ == "__main__":
    # --- 用户配置区 ---
    # 1. MSTAR-ASC数据集的根目录
    asc_dataset_root = (
        r"E:\Document\paper_library\3rd_paper_250512\dataset\MSTAR-ASC\10 categories\test\T72(SN_132)\ASCM_FEATURE"
    )

    # 2. 重建图像的输出根目录 (程序会自动创建)
    output_root = r"E:\Document\paper_library\3rd_paper_250512\dataset\MSTAR-ASC\10 categories\test\T72(SN_132)\My_Reconstructions_v2"
    # --------------------------

    # 自动创建输出目录
    os.makedirs(output_root, exist_ok=True)
    print(f"输出目录已确认: {output_root}")

    # 检查输入目录是否存在
    if not os.path.isdir(asc_dataset_root):
        print(f"错误: 输入目录 '{asc_dataset_root}' 不存在或不是一个有效的目录。")
    else:
        # 获取目录下所有.mat文件
        mat_files = [f for f in os.listdir(asc_dataset_root) if f.lower().endswith(".mat")]

        if not mat_files:
            print(f"警告: 在目录 '{asc_dataset_root}' 中没有找到任何.mat文件。")
        else:
            print(f"开始批量重建 {len(mat_files)} 个文件...")

            # 遍历所有.mat文件进行处理
            for filename in mat_files:
                input_path = os.path.join(asc_dataset_root, filename)

                # 构建输出文件名 (e.g., HB03333_015.mat -> HB03333_015_reconstruct.jpg)
                output_filename = os.path.splitext(filename)[0] + "_reconstruct.jpg"
                output_path = os.path.join(output_root, output_filename)

                try:
                    # 加载ASC参数矩阵
                    asc_mat = sio.loadmat(input_path)
                    # 假设.mat文件中的变量名为 'ASCM_feature'
                    ascm_feature = asc_mat["ASCM_feature"]

                    # 执行重建
                    reconstructed_image = reconstruct_image_from_asc(ascm_feature)

                    # 归一化图像亮度以便于保存和查看
                    img_max = reconstructed_image.max()
                    if img_max > 0:
                        reconstructed_image_normalized = (reconstructed_image / img_max * 255).astype(np.uint8)
                    else:
                        reconstructed_image_normalized = reconstructed_image.astype(np.uint8)

                    plt.imsave(output_path, reconstructed_image_normalized, cmap="gray", format="jpg")

                    print(f"  [√] 已重建并保存: {output_filename}")

                except FileNotFoundError:
                    print(f"  [X] 错误: 找不到文件 {input_path}")
                except KeyError:
                    print(f"  [X] 错误: 在 {filename} 中找不到名为 'ASCM_feature' 的变量。")
                except Exception as e:
                    print(f"  [X] 处理 {filename} 时发生未知错误: {e}")

            print("批量重建完成！")
