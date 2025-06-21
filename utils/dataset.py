# filename: utils/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config


def read_sar_complex_tensor(file_path, height, width):
    """Reads SAR complex data from a .raw file with the correct byte order."""
    try:
        # MSTAR data is stored in big-endian format
        data = np.fromfile(file_path, dtype=">f4")

        expected_size = height * width * 2
        if data.size != expected_size:
            print(
                f"Warning: Corrupted file or wrong dimensions for {os.path.basename(file_path)}. Expected {expected_size} values, found {data.size}."
            )
            return None

        magnitude = data[: height * width].reshape(height, width)
        phase = data[height * width :].reshape(height, width)

        real_part = np.nan_to_num(magnitude * np.cos(phase))
        imag_part = np.nan_to_num(magnitude * np.sin(phase))

        complex_tensor = torch.complex(torch.from_numpy(real_part).float(), torch.from_numpy(imag_part).float())
        return complex_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


class MSTAR_ASC_5CH_Dataset(Dataset):
    """
    Dataset for loading SAR images and their corresponding 5-channel labels.
    Supports different modes ('train', 'val') for different data splits.
    """

    def __init__(self, mode="train"):
        """
        Initializes the dataset loader.
        Args:
            mode (str): 'train' to load from 'train_17_deg', 'val' to load from 'test_15_deg'.
        """
        if mode not in ["train", "val"]:
            raise ValueError("Mode must be 'train' or 'val'.")

        self.mode = mode
        self.label_root = config.LABEL_SAVE_ROOT_5CH
        self.sar_root = config.SAR_RAW_ROOT
        self.samples = []

        data_subdir = "train_17_deg" if self.mode == "train" else "test_15_deg"

        self.data_path = os.path.join(self.label_root, data_subdir)
        sar_data_path = os.path.join(self.sar_root, data_subdir)

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Label directory for mode '{self.mode}' not found: {self.data_path}")
        if not os.path.exists(sar_data_path):
            raise FileNotFoundError(f"SAR raw data directory for mode '{self.mode}' not found: {sar_data_path}")

        print(f"Searching for samples in: {self.data_path}")
        for dirpath, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.endswith("_5ch.npy"):
                    label_path = os.path.join(dirpath, filename)

                    # --- THIS IS THE CORRECTED LOGIC ---
                    # Create the base name by removing the label-specific suffix.
                    # Example: "HB03787.000.128x128_5ch.npy" -> "HB03787.000.128x128"
                    raw_base_name = filename.replace("_5ch.npy", "")

                    # Construct the full name of the corresponding .raw file.
                    # Example: "HB03787.000.128x128.raw"
                    sar_filename = raw_base_name + ".raw"

                    # Get the relative path from the data_path (e.g., "BMP2/SN_9563")
                    rel_dir = os.path.relpath(dirpath, self.data_path)

                    # Construct the full path to the .raw file.
                    sar_path = os.path.join(sar_data_path, rel_dir, sar_filename)
                    # --- END OF CORRECTION ---

                    if os.path.exists(sar_path):
                        self.samples.append({"sar": sar_path, "label": label_path})

        print(f"Found {len(self.samples)} samples for mode '{self.mode}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        sar_tensor = read_sar_complex_tensor(sample_paths["sar"], config.IMG_HEIGHT, config.IMG_WIDTH)

        if sar_tensor is None:
            print(f"Warning: Skipping corrupted sample at index {idx}: {sample_paths['sar']}")
            return self.__getitem__((idx + 1) % len(self))

        label_tensor = torch.from_numpy(np.load(sample_paths["label"])).float()

        return sar_tensor, label_tensor
