import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from utils.io import load_config  # noqa: E402


# 说明：
# 读取以大端 float32 存储的 SAR 复数据（幅度/相位），并转换为 PyTorch 复数张量。
# - 参数：
#   file_path: 原始 .raw 文件路径；height/width: 图像尺寸（像素）。
# - 返回：
#   若读取成功并尺寸匹配，返回形状为 [H, W]、dtype=torch.complex64 的张量，否则返回 None。
# - 行为：
#   文件以顺序存放幅度(H*W)与相位(H*W)两段，先转为实部/虚部，再合成为复数。
def read_sar_complex_tensor(file_path: str, height: int, width: int) -> Optional[torch.Tensor]:
    """Read SAR complex data stored as magnitude/phase big-endian float32."""
    try:
        data = np.fromfile(file_path, dtype=">f4")
    except FileNotFoundError:
        return None
    expected = height * width * 2
    if data.size != expected:
        return None
    magnitude = data[: height * width].reshape(height, width)
    phase = data[height * width :].reshape(height, width)
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    complex_tensor = torch.complex(
        torch.from_numpy(np.nan_to_num(real_part)).float(),
        torch.from_numpy(np.nan_to_num(imag_part)).float(),
    )
    return complex_tensor


# 说明：
# 将 5 通道标签转换为 2 通道 (A, alpha)。
# - 输入：label_5ch，形状至少含有 5 个通道，约定：
#   [0]=mask/其他，[1]=对数幅度 amp_log，[2]=alpha，其余通道按需求扩展。
# - 处理：对数幅度经 expm1 转线性；利用正幅度作为有效掩码，将无效位置清零。
# - 输出：形状为 [2, H, W] 的 float32 数组，通道 0 为线性幅度 A，通道 1 为 alpha。
def five_to_two(label_5ch: np.ndarray) -> np.ndarray:
    """Convert 5-channel labels to 2-channel (A, alpha)."""
    if label_5ch.shape[0] < 5:
        raise ValueError("Expected 5-channel label input")
    amp_log = label_5ch[1]
    alpha = label_5ch[2]
    amp_linear = np.expm1(amp_log)
    label_2ch = np.zeros((2, amp_linear.shape[0], amp_linear.shape[1]), dtype=np.float32)
    label_2ch[0] = amp_linear.astype(np.float32)
    label_2ch[1] = alpha.astype(np.float32)
    mask = amp_linear > 0
    label_2ch[0][~mask] = 0.0
    label_2ch[1][~mask] = 0.0
    return label_2ch


# 说明：
# 面向 A-ConvNets-FCN 训练/推理的数据集封装，提供
# - 输入：
#   config_path: 配置文件路径（读取 data.sar_root、data.label_2ch_root、data.label_5ch_root、image_height/width）；
#   label_on_the_fly: 若 True，则可直接读取 5 通道标签并在取样时转换为 2 通道；
#   transform: 可选数据增强/预处理，签名为 (image, label) -> (image, label)。
# - 输出：
#   __getitem__ 返回 (magnitude, label_2ch)；magnitude 为 [1,H,W] 的幅度张量，label 为 [2,H,W]。
# - 行为：
#   递归遍历标签目录，匹配同名 .raw（幅相格式）作为 SAR 输入；无样本或尺寸异常会抛出错误/跳过。
class ASC2ChannelDataset(Dataset):
    """Dataset that serves SAR magnitude images with 2-channel labels."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        label_on_the_fly: bool = False,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        cfg = load_config(config_path)
        self.image_height = int(cfg["data"].get("image_height", 128))
        self.image_width = int(cfg["data"].get("image_width", 128))
        self.transform = transform
        self.label_on_the_fly = label_on_the_fly

        data_cfg = cfg["data"]
        self.sar_root = Path(data_cfg["sar_root"])
        self.label_2ch_root = Path(data_cfg["label_2ch_root"])
        self.label_5ch_root = Path(data_cfg["label_5ch_root"]) if label_on_the_fly else None

        self.samples: List[dict] = []

        label_root = self.label_2ch_root if self.label_2ch_root.exists() else self.label_5ch_root
        if label_root is None or not label_root.exists():
            raise FileNotFoundError(
                "No label directory found. Run convert_labels.py first or enable on-the-fly conversion with label_on_the_fly=True."
            )

        for dirpath, _, filenames in os.walk(label_root):
            for filename in filenames:
                if not filename.endswith(".npy"):
                    continue
                label_path = Path(dirpath) / filename
                rel_dir = Path(dirpath).relative_to(label_root)
                raw_path = (self.sar_root / rel_dir / filename.replace(".npy", ".raw")).resolve()
                if raw_path.exists():
                    self.samples.append({"sar": raw_path, "label": label_path})

        if not self.samples:
            raise RuntimeError("No samples found for ASC2ChannelDataset")

    # 说明：返回样本总数。
    def __len__(self) -> int:
        return len(self.samples)

    # 说明：按索引获取 1 个样本。
    # - 读入 SAR 复数数据，取幅度并在通道维扩展为 [1,H,W]；
    # - 标签若为 5 通道则即时转换为 2 通道，否则按 2 通道读取；
    # - 若提供 transform，则对 (magnitude, label) 同步变换；
    # - 若当前 .raw 读取失败，循环到下一个索引样本。
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sar_tensor = read_sar_complex_tensor(str(sample["sar"]), self.image_height, self.image_width)
        if sar_tensor is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        magnitude = sar_tensor.abs().unsqueeze(0)

        label_array = np.load(sample["label"])
        if label_array.shape[0] == 5:
            label_2ch = five_to_two(label_array)
        else:
            label_2ch = label_array.astype(np.float32)
        label_tensor = torch.from_numpy(label_2ch)

        if self.transform:
            magnitude, label_tensor = self.transform(magnitude, label_tensor)

        return magnitude, label_tensor
