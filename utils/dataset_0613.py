# utils/dataset.py (升级版)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config


def read_sar_complex_tensor(file_path, height, width):
    """从.raw文件中读取SAR复数数据，进行鲁棒的标准化，并返回一个复数张量"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if data.size != height * width * 2:
            # print(f"Warning: Unexpected data size in {file_path}. Skipping.")
            return None

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        magnitude = data[: height * width].reshape(height, width)
        phase = data[height * width :].reshape(height, width)

        real_part = np.nan_to_num(magnitude * np.cos(phase))
        imag_part = np.nan_to_num(magnitude * np.sin(phase))

        # --- 鲁棒标准化：百分位截断 + 范围缩放 ---
        def robust_scaling(matrix):
            # 1. 百分位截断，去除极端异常值
            p1, p99 = np.percentile(matrix, [1, 99])
            clipped_matrix = np.clip(matrix, p1, p99)

            # 2. 范围缩放（Min-Max Scaling）到 [-1, 1]
            min_val = clipped_matrix.min()
            max_val = clipped_matrix.max()
            if max_val - min_val > 1e-8:
                scaled_matrix = -1.0 + 2.0 * (clipped_matrix - min_val) / (max_val - min_val)
            else:
                # 如果矩阵值都一样，则返回全零矩阵
                scaled_matrix = np.zeros_like(matrix)
            return scaled_matrix

        real_part_scaled = robust_scaling(real_part)
        imag_part_scaled = robust_scaling(imag_part)

        complex_tensor = torch.complex(
            torch.from_numpy(real_part_scaled).float(), torch.from_numpy(imag_part_scaled).float()
        )
        return complex_tensor.unsqueeze(0)

    except Exception as e:
        print(f"Error reading or processing raw file {file_path}: {e}")
        return None


class MSTAR_ASC_Dataset(Dataset):
    def __init__(self, sar_root, label_root):
        """
        一个可以同时处理嵌套目录和扁平目录的数据集类。
        sar_root: 存放 .raw SAR图像的根目录。
        label_root: 存放 .npy 参数图像标签的根目录。
        """
        self.sar_root = sar_root
        self.label_root = label_root
        self.samples = []

        if not os.path.exists(label_root):
            print(f"Error: Label directory not found at {label_root}")
            return

        # os.walk() 可以优雅地处理嵌套和扁平两种目录结构
        for dirpath, _, filenames in os.walk(self.label_root):
            for filename in filenames:
                if filename.endswith("_A.npy"):
                    # 1. 构建标签文件路径
                    label_A_path = os.path.join(dirpath, filename)
                    base_name = filename.replace("_A.npy", "")
                    label_alpha_path = os.path.join(dirpath, base_name + "_alpha.npy")

                    # 2. 基于标签路径，反向构建原始SAR图像的路径
                    # 计算标签文件相对于标签根目录的相对路径
                    relative_path = os.path.relpath(dirpath, self.label_root)

                    # 从标签文件名中恢复原始SAR文件名
                    raw_base_name = base_name.replace("_yang", "")
                    sar_raw_filename = raw_base_name + ".128x128.raw"

                    # 使用相对路径在SAR图像根目录中找到对应的文件
                    sar_raw_path = os.path.join(self.sar_root, relative_path, sar_raw_filename)

                    # 3. 确保所有需要的文件都真实存在
                    if os.path.exists(label_alpha_path) and os.path.exists(sar_raw_path):
                        self.samples.append(
                            {
                                "sar_path": sar_raw_path,
                                "label_A_path": label_A_path,
                                "label_alpha_path": label_alpha_path,
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        sar_complex_tensor = read_sar_complex_tensor(sample["sar_path"], config.IMG_HEIGHT, config.IMG_WIDTH)

        if sar_complex_tensor is None:
            return self.__getitem__((idx + 1) % len(self))

        label_A = np.load(sample["label_A_path"])
        label_alpha = np.load(sample["label_alpha_path"])

        # <<< ADDED: Label Normalization for Amplitude >>>
        # Apply log1p transform only to non-zero values to compress the range
        # This is a robust way to handle the wide dynamic range of amplitude.
        label_A[label_A > 0] = np.log1p(label_A[label_A > 0])
        # Alpha is already in a nice [-1, 1] range, so we leave it as is.
        # We will handle its range constraint on the model output side.

        label_tensor = torch.from_numpy(np.stack([label_A, label_alpha], axis=0)).float()

        return sar_complex_tensor, label_tensor
