import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config


def read_sar_complex_tensor(file_path, height, width):
    """Reads SAR complex data from a .raw file with the correct byte order."""
    try:
        data = np.fromfile(file_path, dtype=">f4")
        if data.size != height * width * 2:
            return None
        magnitude = data[: height * width].reshape(height, width)
        phase = data[height * width :].reshape(height, width)
        real_part = np.nan_to_num(magnitude * np.cos(phase))
        imag_part = np.nan_to_num(magnitude * np.sin(phase))
        complex_tensor = torch.complex(torch.from_numpy(real_part).float(), torch.from_numpy(imag_part).float())
        return complex_tensor.unsqueeze(0)
    except Exception:
        return None


class MSTAR_ASC_5CH_Dataset(Dataset):
    """Dataset for loading SAR images and their corresponding 5-channel labels."""

    def __init__(self):
        self.label_root = config.LABEL_SAVE_ROOT_5CH
        self.sar_root = config.SAR_RAW_ROOT
        self.samples = []

        if not os.path.exists(self.label_root):
            raise FileNotFoundError(f"Label directory not found: {self.label_root}")

        for dirpath, _, filenames in os.walk(self.label_root):
            for filename in filenames:
                if filename.endswith(".npy"):
                    label_path = os.path.join(dirpath, filename)

                    # Find corresponding raw SAR file
                    raw_base_name = filename.replace(".npy", "")
                    rel_dir = os.path.relpath(dirpath, self.label_root)
                    sar_path = os.path.join(self.sar_root, rel_dir, raw_base_name + ".raw")

                    if os.path.exists(sar_path):
                        self.samples.append({"sar": sar_path, "label": label_path})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        sar_tensor = read_sar_complex_tensor(sample_paths["sar"], config.IMG_HEIGHT, config.IMG_WIDTH)
        label_tensor = torch.from_numpy(np.load(sample_paths["label"])).float()

        if sar_tensor is None:
            # Handle corrupted data by loading the next sample
            return self.__getitem__((idx + 1) % len(self))

        return sar_tensor, label_tensor
