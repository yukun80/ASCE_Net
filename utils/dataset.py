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

    def __init__(self, split=None, target_classes=None):
        """
        Args:
            split (str): 'train', 'test', or None (use all data)
            target_classes (list): List of target classes to include, or None for all
        """
        # --- NEW: Get current paths based on configuration ---
        paths = config.get_current_paths()
        self.label_root = paths["label_save_root"]
        self.sar_root = paths["sar_raw_root"]
        self.samples = []

        # --- NEW: Split and class filtering ---
        self.split = split
        self.target_classes = target_classes or config.TARGET_CLASSES

        if not os.path.exists(self.label_root):
            raise FileNotFoundError(f"Label directory not found: {self.label_root}")

        if not os.path.exists(self.sar_root):
            raise FileNotFoundError(f"SAR data directory not found: {self.sar_root}")

        # --- NEW: Determine which folders to search ---
        if config.USE_COMPLETE_DATASET:
            if split == "train":
                search_folders = [config.TRAIN_FOLDER]
            elif split == "test":
                search_folders = [config.TEST_FOLDER]
            elif split is None:
                search_folders = [config.TRAIN_FOLDER, config.TEST_FOLDER]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or None")
        else:
            # For testing dataset, search all folders
            search_folders = ["."]  # Search from root

        for folder in search_folders:
            search_path = os.path.join(self.label_root, folder) if config.USE_COMPLETE_DATASET else self.label_root

            for dirpath, _, filenames in os.walk(search_path):
                # --- NEW: Filter by target classes if using complete dataset ---
                if config.USE_COMPLETE_DATASET:
                    # Check if current directory corresponds to a target class
                    current_class = os.path.basename(dirpath)
                    if current_class not in self.target_classes:
                        continue

                for filename in filenames:
                    if filename.endswith("_5ch.npy"):
                        label_path = os.path.join(dirpath, filename)

                        # Find corresponding raw SAR file
                        raw_base_name = filename.replace("_5ch.npy", "")
                        rel_dir = os.path.relpath(dirpath, self.label_root)
                        sar_path = os.path.join(self.sar_root, rel_dir, raw_base_name + ".128x128.raw")

                        if os.path.exists(sar_path):
                            # --- NEW: Store additional metadata ---
                            sample_info = {
                                "sar": sar_path,
                                "label": label_path,
                                "class": current_class if config.USE_COMPLETE_DATASET else "unknown",
                                "split": folder if config.USE_COMPLETE_DATASET else "unknown",
                            }
                            self.samples.append(sample_info)

        print(f"Dataset initialized with {len(self.samples)} samples")
        if config.USE_COMPLETE_DATASET and split:
            class_counts = {}
            for sample in self.samples:
                class_name = sample["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            print(f"Class distribution in {split} set: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        sar_tensor = read_sar_complex_tensor(sample_info["sar"], config.IMG_HEIGHT, config.IMG_WIDTH)
        label_tensor = torch.from_numpy(np.load(sample_info["label"])).float()

        if sar_tensor is None:
            # Handle corrupted data by loading the next sample
            return self.__getitem__((idx + 1) % len(self))

        return sar_tensor, label_tensor

    def get_sample_info(self, idx):
        """Return metadata about a specific sample"""
        return self.samples[idx]
