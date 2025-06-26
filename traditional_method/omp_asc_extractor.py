"""
OMP-based ASC Extractor for MSTAR Dataset
==========================================

实现基于正交匹配追踪(OMP)算法的散射中心提取器，用于MSTAR数据集。
该实现使用scikit-learn的官方OMP实现，结合SAR成像物理模型。

作者: SAR算法工程师
日期: 2025年1月
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy.signal.windows import taylor
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.preprocessing import normalize
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import config


class OMPASCExtractor:
    """基于OMP算法的散射中心提取器"""

    def __init__(
        self,
        img_size=(128, 128),
        search_region=(-6.4, 6.4, -6.4, 6.4),  # (x_min, x_max, y_min, y_max) in meters
        grid_resolution=0.2,  # 网格分辨率 (米)
        amplitude_range=(0.1, 10.0),  # 幅度搜索范围
        alpha_range=(-2.0, 2.0),  # Alpha参数搜索范围
        n_nonzero_coefs=20,  # OMP稀疏度控制
        cross_validation=True,
    ):
        """
        初始化OMP ASC提取器

        Parameters:
        -----------
        img_size : tuple
            图像尺寸 (height, width)
        search_region : tuple
            搜索区域 (x_min, x_max, y_min, y_max) 单位：米
        grid_resolution : float
            空间网格分辨率，单位：米
        amplitude_range : tuple
            幅度搜索范围
        alpha_range : tuple
            Alpha参数搜索范围
        n_nonzero_coefs : int
            OMP算法的稀疏度参数
        cross_validation : bool
            是否使用交叉验证自动选择稀疏度
        """
        self.img_size = img_size
        self.search_region = search_region
        self.grid_resolution = grid_resolution
        self.amplitude_range = amplitude_range
        self.alpha_range = alpha_range
        self.n_nonzero_coefs = n_nonzero_coefs
        self.cross_validation = cross_validation

        # SAR成像参数
        self.fc = 1e10  # 中心频率 10 GHz
        self.B = 5e8  # 带宽 500 MHz
        self.om_deg = 2.86  # 俯仰角度
        self.p = 84  # 频域采样点数
        self.q = 128  # 图像尺寸

        self.om_rad = self.om_deg * np.pi / 180.0
        self.bw_ratio = self.B / self.fc

        # 构建频率网格
        self.fx_range = np.linspace(self.fc * (1 - self.bw_ratio / 2), self.fc * (1 + self.bw_ratio / 2), self.p)
        self.fy_range = np.linspace(-self.fc * np.sin(self.om_rad / 2), self.fc * np.sin(self.om_rad / 2), self.p)

        # 构建字典
        self.dictionary = None
        self.dictionary_params = None
        print(f"🎯 OMP ASC提取器初始化完成")
        print(f"   图像尺寸: {img_size}")
        print(f"   搜索区域: {search_region} 米")
        print(f"   网格分辨率: {grid_resolution} 米")
        print(f"   OMP稀疏度: {n_nonzero_coefs}")
        print(f"   交叉验证: {cross_validation}")

    def _sar_model(self, fx, fy, x, y, alpha, A=1.0):
        """
        SAR成像的单散射中心模型

        Parameters:
        -----------
        fx, fy : float
            频率坐标
        x, y : float
            散射中心位置 (米)
        alpha : float
            频率依赖相位参数
        A : float
            散射幅度

        Returns:
        --------
        complex
            该散射中心在(fx,fy)处的复数响应
        """
        f = np.sqrt(fx**2 + fy**2)
        theta = np.arctan2(fy, fx)

        # ASC模型简化版（只考虑位置和频率依赖性）
        E1 = A * (1j * f / self.fc) ** alpha
        E2 = np.exp(-1j * 4 * np.pi * f / 3e8 * (x * np.cos(theta) + y * np.sin(theta)))

        return E1 * E2

    def _build_dictionary(self):
        """构建OMP字典矩阵"""
        print("🔧 构建OMP字典矩阵...")

        # 生成位置网格
        x_min, x_max, y_min, y_max = self.search_region
        x_coords = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_coords = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)

        # 生成参数网格
        alpha_values = np.linspace(self.alpha_range[0], self.alpha_range[1], 11)

        # 计算字典大小
        n_positions = len(x_coords) * len(y_coords)
        n_alphas = len(alpha_values)
        n_atoms = n_positions * n_alphas
        n_measurements = self.p * self.p  # K空间测量数

        print(f"   位置网格: {len(x_coords)} x {len(y_coords)} = {n_positions}")
        print(f"   Alpha参数: {n_alphas}")
        print(f"   字典原子总数: {n_atoms}")
        print(f"   测量维度: {n_measurements}")

        # 初始化字典和参数
        dictionary = np.zeros((n_measurements, n_atoms), dtype=complex)
        dictionary_params = []

        atom_idx = 0
        for x in tqdm(x_coords, desc="构建字典", leave=False):
            for y in y_coords:
                for alpha in alpha_values:
                    # 为当前参数组合生成字典原子
                    atom = np.zeros((self.p, self.p), dtype=complex)

                    for i, fy in enumerate(self.fy_range):
                        for j, fx in enumerate(self.fx_range):
                            atom[i, j] = self._sar_model(fx, fy, x, y, alpha)

                    # 展平并归一化
                    atom_flat = atom.flatten()
                    atom_flat = atom_flat / (np.linalg.norm(atom_flat) + 1e-12)

                    dictionary[:, atom_idx] = atom_flat
                    dictionary_params.append({"x": x, "y": y, "alpha": alpha})
                    atom_idx += 1

                # 转换为实数字典（垂直堆叠实部和虚部）
        dict_real = np.vstack([dictionary.real, dictionary.imag])

        self.dictionary = dict_real
        self.dictionary_params = dictionary_params

        print(f"✅ 字典构建完成，尺寸: {self.dictionary.shape}")
        return self.dictionary, self.dictionary_params

    def _k_space_to_image(self, k_space_data):
        """将K空间数据转换为图像域"""
        # 应用Taylor窗
        win = taylor(self.p, nbar=4, sll=35)
        window_2d = np.outer(win, win)
        k_windowed = k_space_data * window_2d

        # 零填充到图像尺寸
        Z = np.zeros((self.q, self.q), dtype=complex)
        start = (self.q - self.p) // 2
        Z[start : start + self.p, start : start + self.p] = k_windowed

        # IFFT到图像域
        img_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Z)))
        return img_complex

    def _image_to_k_space(self, img_complex):
        """将图像域数据转换为K空间（MSTAR数据处理）"""
        # FFT到频域
        Z_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_complex)))

        # 提取有效K空间区域
        start = (self.q - self.p) // 2
        k_space = Z_full[start : start + self.p, start : start + self.p]

        # 去除Taylor窗效应
        win = taylor(self.p, nbar=4, sll=35)
        window_2d = np.outer(win, win)
        k_space = k_space / (window_2d + 1e-12)

        return k_space

    def extract_asc_from_raw(self, raw_file_path):
        """从raw文件提取ASC参数"""
        try:
            # 读取RAW文件
            img_complex = self._read_raw_file(raw_file_path)
            return self.extract_asc_from_image(img_complex)
        except Exception as e:
            print(f"❌ 处理RAW文件失败: {e}")
            return []

    def extract_asc_from_mat(self, mat_file_path):
        """从mat文件提取ASC参数"""
        try:
            # 读取MAT文件
            mat_data = sio.loadmat(mat_file_path)
            Img = mat_data["Img"]
            phase = mat_data["phase"]

            # 组合为复数图像
            img_complex = Img * np.exp(1j * phase)
            return self.extract_asc_from_image(img_complex)
        except Exception as e:
            print(f"❌ 处理MAT文件失败: {e}")
            return []

    def extract_asc_from_image(self, img_complex):
        """从复数图像提取ASC参数"""
        # 确保字典已构建
        if self.dictionary is None:
            self._build_dictionary()

        print("🔍 使用OMP算法提取ASC参数...")

        # 转换到K空间
        k_space = self._image_to_k_space(img_complex)

        # 准备OMP输入（实数向量）
        y_complex = k_space.flatten()
        y_real = np.hstack([y_complex.real, y_complex.imag])

        # 归一化
        y_norm = np.linalg.norm(y_real)
        if y_norm > 0:
            y_real = y_real / y_norm

        # 执行OMP算法
        if self.cross_validation:
            print("   使用交叉验证选择最优稀疏度...")
            omp = OrthogonalMatchingPursuitCV(cv=3, max_iter=min(50, self.n_nonzero_coefs * 2))
        else:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)

        # 拟合OMP模型
        omp.fit(self.dictionary, y_real)

        # 提取稀疏系数
        sparse_coefs = omp.coef_
        active_indices = np.where(np.abs(sparse_coefs) > 1e-6)[0]

        print(f"   检测到 {len(active_indices)} 个活跃散射中心")

        # 转换为ASC参数
        asc_list = []
        for idx in active_indices:
            params = self.dictionary_params[idx].copy()
            amplitude = np.abs(sparse_coefs[idx]) * y_norm  # 恢复原始幅度尺度
            params["A"] = amplitude

            # 添加其他默认参数
            params["gamma"] = 0.0
            params["phi_prime"] = 0.0
            params["L"] = 0.0

            asc_list.append(params)

        # 按幅度排序
        asc_list.sort(key=lambda x: x["A"], reverse=True)

        return asc_list

    def _read_raw_file(self, raw_file_path):
        """读取RAW格式的复数SAR图像"""
        try:
            # 读取二进制数据
            with open(raw_file_path, "rb") as f:
                data = np.fromfile(f, dtype=np.float32)

            # 假设数据格式：幅度+相位
            n_pixels = self.img_size[0] * self.img_size[1]
            if len(data) >= 2 * n_pixels:
                magnitude = data[:n_pixels].reshape(self.img_size)
                phase = data[n_pixels : 2 * n_pixels].reshape(self.img_size)

                # 组合为复数图像
                img_complex = magnitude * np.exp(1j * phase)
                return img_complex
            else:
                raise ValueError(f"数据大小不匹配: 期望{2*n_pixels}, 实际{len(data)}")

        except Exception as e:
            raise RuntimeError(f"读取RAW文件失败: {e}")

    def reconstruct_image_from_asc(self, asc_list):
        """从ASC参数列表重建SAR图像"""
        print("🔧 从ASC参数重建SAR图像...")

        # 在K空间中合成信号
        k_space_recon = np.zeros((self.p, self.p), dtype=complex)

        for asc in asc_list:
            for i, fy in enumerate(self.fy_range):
                for j, fx in enumerate(self.fx_range):
                    k_space_recon[i, j] += self._sar_model(fx, fy, asc["x"], asc["y"], asc["alpha"], asc["A"])

        # 转换到图像域
        img_recon = self._k_space_to_image(k_space_recon)
        return img_recon

    def evaluate_reconstruction(self, original_img, reconstructed_img):
        """评估重建质量"""
        # 计算相对RMSE
        original_norm = np.linalg.norm(original_img)
        if original_norm > 0:
            diff_norm = np.linalg.norm(reconstructed_img - original_img)
            relative_rmse = diff_norm / original_norm
        else:
            relative_rmse = float("inf")

        # 计算相关系数
        orig_flat = original_img.flatten()
        recon_flat = reconstructed_img.flatten()
        correlation = np.corrcoef(np.abs(orig_flat), np.abs(recon_flat))[0, 1]

        return {"relative_rmse": relative_rmse, "correlation": correlation if not np.isnan(correlation) else 0.0}

    def save_results(self, asc_list, output_path):
        """保存ASC提取结果为MAT格式"""
        # 转换为MATLAB兼容格式
        scatter_all = []
        for asc in asc_list:
            params = [asc["x"], asc["y"], asc["alpha"], asc["gamma"], asc["phi_prime"], asc["L"], asc["A"]]
            scatter_all.append(params)

        # 保存为MAT文件
        sio.savemat(output_path, {"scatter_all": np.array(scatter_all)})
        print(f"✅ ASC结果已保存到: {output_path}")

    def visualize_results(self, original_img, asc_list, save_path=None):
        """可视化ASC提取结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始图像
        axes[0, 0].imshow(np.abs(original_img), cmap="gray")
        axes[0, 0].set_title("原始SAR图像")
        axes[0, 0].axis("off")

        # ASC位置叠加
        axes[0, 1].imshow(np.abs(original_img), cmap="gray", alpha=0.7)

        # 转换ASC位置到像素坐标
        for asc in asc_list:
            # 简化的坐标转换
            pixel_x = int((asc["x"] - self.search_region[0]) / self.grid_resolution)
            pixel_y = int((asc["y"] - self.search_region[2]) / self.grid_resolution)

            # 确保在图像范围内
            if 0 <= pixel_x < self.img_size[1] and 0 <= pixel_y < self.img_size[0]:
                axes[0, 1].plot(pixel_x, pixel_y, "r+", markersize=10, markeredgewidth=2)

        axes[0, 1].set_title(f"检测到的ASC位置 ({len(asc_list)}个)")
        axes[0, 1].axis("off")

        # 重建图像
        img_recon = self.reconstruct_image_from_asc(asc_list)
        axes[1, 0].imshow(np.abs(img_recon), cmap="gray")
        axes[1, 0].set_title("OMP重建图像")
        axes[1, 0].axis("off")

        # 差值图像
        diff_img = np.abs(original_img - img_recon)
        axes[1, 1].imshow(diff_img, cmap="hot")
        axes[1, 1].set_title("重建误差")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 可视化结果已保存到: {save_path}")

        plt.show()


def main():
    """主函数：演示OMP ASC提取器的使用"""
    print("🚀 OMP-based ASC提取器演示")
    print("=" * 50)

    # 初始化提取器
    extractor = OMPASCExtractor(
        img_size=(128, 128),
        search_region=(-6.4, 6.4, -6.4, 6.4),
        grid_resolution=0.4,  # 较粗的网格以减少计算量
        n_nonzero_coefs=15,
        cross_validation=False,  # 演示时关闭交叉验证
    )

    # 数据路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mat_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "01_Data_Processed_mat_part-tmp")
    raw_root = os.path.join(project_root, "datasets", "SAR_ASC_Project", "02_Data_Processed_raw")
    output_root = os.path.join(project_root, "traditional_method", "omp_results")

    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)

    # 查找测试文件
    test_files = []
    if os.path.exists(mat_root):
        for root, dirs, files in os.walk(mat_root):
            for file in files[:3]:  # 只处理前3个文件作为演示
                if file.endswith(".mat"):
                    test_files.append(os.path.join(root, file))

    if not test_files:
        print("❌ 未找到测试文件，请先运行MATLAB预处理脚本")
        return

    print(f"📁 找到 {len(test_files)} 个测试文件")

    # 处理测试文件
    for i, test_file in enumerate(test_files):
        print(f"\n📊 处理文件 {i+1}/{len(test_files)}: {os.path.basename(test_file)}")

        try:
            # 提取ASC
            asc_list = extractor.extract_asc_from_mat(test_file)

            if asc_list:
                print(f"✅ 提取到 {len(asc_list)} 个散射中心")

                # 显示前5个最强散射中心
                print("   前5个最强散射中心:")
                for j, asc in enumerate(asc_list[:5]):
                    print(
                        f"     {j+1}. 位置:({asc['x']:.2f}, {asc['y']:.2f}), "
                        f"幅度:{asc['A']:.3f}, Alpha:{asc['alpha']:.3f}"
                    )

                # 保存结果
                base_name = os.path.splitext(os.path.basename(test_file))[0]
                output_file = os.path.join(output_root, f"{base_name}_omp_asc.mat")
                extractor.save_results(asc_list, output_file)

                # 重建评估
                mat_data = sio.loadmat(test_file)
                original_img = mat_data["Img"] * np.exp(1j * mat_data["phase"])
                recon_img = extractor.reconstruct_image_from_asc(asc_list)

                metrics = extractor.evaluate_reconstruction(original_img, recon_img)
                print(f"   重建质量 - RMSE: {metrics['relative_rmse']:.4f}, " f"相关系数: {metrics['correlation']:.4f}")

                # 可视化（仅第一个文件）
                if i == 0:
                    viz_path = os.path.join(output_root, f"{base_name}_omp_visualization.png")
                    extractor.visualize_results(original_img, asc_list, viz_path)
            else:
                print("⚠️ 未检测到散射中心")

        except Exception as e:
            print(f"❌ 处理失败: {e}")

    print("\n🎉 OMP ASC提取演示完成！")


if __name__ == "__main__":
    main()
