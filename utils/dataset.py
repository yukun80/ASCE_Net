# utils/dataset.py (最终修正版)
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config
from collections import Counter


def read_sar_complex_tensor(file_path, height, width):
    """从.raw文件中读取SAR复数数据。"""
    try:
        data = np.fromfile(file_path, dtype=">f4")
        if data.size != height * width * 2:
            print(f"警告: 文件大小不匹配 {file_path}, 期望 {height * width * 2}, 得到 {data.size}")
            return None

        magnitude = data[: height * width].reshape(height, width)
        phase = data[height * width :].reshape(height, width)

        real_part = np.nan_to_num(magnitude * np.cos(phase))
        imag_part = np.nan_to_num(magnitude * np.sin(phase))

        complex_tensor = torch.complex(torch.from_numpy(real_part).float(), torch.from_numpy(imag_part).float())

        return complex_tensor.unsqueeze(0)
    except Exception as e:
        print(f"错误: 读取文件时发生异常 {file_path}: {e}")
        return None


class MSTAR_ASC_5CH_Dataset(Dataset):
    """为加载SAR图像及其对应的5通道标签而设计的数据集。"""

    def __init__(self, split="train", vehicle_types=None):
        """
        初始化数据集。
        :param split: 'train' 或 'test'，用于选择数据集部分。
        :param vehicle_types: (可选) 一个列表，用于筛选特定的车辆类型。
        """
        self.split = split
        self.sar_root = config.SAR_RAW_ROOT
        self.label_root = config.LABEL_SAVE_ROOT_5CH
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(config.VEHICLE_TYPES)}

        print(f"Dataset mode: {self.split.capitalize()}")

        # --- 核心逻辑修改：根据 split 参数确定搜索目录 ---
        if self.split == "train":
            search_dir = "train_17_deg"
        elif self.split == "test":
            search_dir = "test_15_deg"
        else:
            raise ValueError(f"无效的数据集分割参数: '{self.split}'. 必须是 'train' 或 'test'。")

        base_search_path = os.path.join(self.label_root, search_dir)

        if not os.path.exists(base_search_path):
            raise FileNotFoundError(f"数据集目录未找到: {base_search_path}")

        for dirpath, _, filenames in os.walk(base_search_path):
            for filename in filenames:
                if filename.endswith("_5ch.npy"):
                    npy_basename = filename.replace("_5ch.npy", "")
                    rel_path = os.path.relpath(dirpath, base_search_path)

                    vehicle_type = rel_path.split(os.sep)[0]
                    # 如果指定了车辆类型，则进行筛选
                    if vehicle_types and vehicle_type not in vehicle_types:
                        continue

                    sar_path = os.path.join(self.sar_root, search_dir, rel_path, f"{npy_basename}.raw")
                    label_path = os.path.join(dirpath, filename)

                    if os.path.exists(sar_path) and vehicle_type in self.class_to_idx:
                        self.samples.append(
                            {
                                "sar_path": sar_path,
                                "label_path": label_path,
                                "class_idx": self.class_to_idx[vehicle_type],
                            }
                        )

        print(f"Dataset initialized with {len(self.samples)} samples for '{self.split}' split.")
        self.print_class_distribution()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        sar_path = sample_info["sar_path"]
        label_path = sample_info["label_path"]

        sar_tensor = read_sar_complex_tensor(sar_path, config.IMG_HEIGHT, config.IMG_WIDTH)

        if sar_tensor is None:
            print(f"警告: 加载样本失败 {sar_path}, 将尝试加载下一个样本。")
            return self.__getitem__((idx + 1) % len(self))

        label_tensor = torch.from_numpy(np.load(label_path)).float()

        return sar_tensor, label_tensor

    def print_class_distribution(self):
        """打印数据集中各类别的样本数量分布。"""
        if not self.samples:
            print("Class distribution: {}")
            return

        class_indices = [s["class_idx"] for s in self.samples]
        counts = Counter(class_indices)

        idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        distribution = {idx_to_class[i]: count for i, count in sorted(counts.items())}

        print(f"Class distribution for '{self.split}' split: {distribution}")
