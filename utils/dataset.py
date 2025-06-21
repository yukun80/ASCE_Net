import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config

# --- NEW: Import Counter for easy counting ---
from collections import Counter


def read_sar_complex_tensor(file_path, height, width):
    """Reads SAR complex data from a .raw file with the correct byte order."""
    try:
        data = np.fromfile(file_path, dtype=">f4")
        if data.size != height * width * 2:
            print(
                f"Warning: Corrupted file or wrong dimensions for {os.path.basename(file_path)}. "
                f"Expected {height*width*2} values, got {data.size}. Skipping."
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
    Now loads data based on predefined train/test directories and prints class distribution.
    """

    def __init__(self, mode="train"):
        self.label_root = config.LABEL_SAVE_ROOT_5CH
        self.sar_root = config.SAR_RAW_ROOT
        self.samples = []

        if mode not in ["train", "val"]:
            raise ValueError(f"Mode must be 'train' or 'val', but got {mode}")
        self.mode = mode

        if self.mode == "train":
            target_dir_name = config.TRAIN_FOLDER  # "train_17_deg"
        else:
            target_dir_name = config.TEST_FOLDER  # "test_15_deg"

        search_path = os.path.join(self.label_root, target_dir_name)

        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Dataset directory for mode '{self.mode}' not found at: {search_path}")

        for dirpath, _, filenames in os.walk(search_path):
            for filename in filenames:
                if filename.endswith("_5ch.npy"):
                    # Check if the path contains one of the target classes
                    if not any(target_class in dirpath for target_class in config.TARGET_CLASSES):
                        continue

                    label_path = os.path.join(dirpath, filename)
                    raw_base_name = filename.replace("_5ch.npy", "")
                    rel_dir = os.path.relpath(dirpath, self.label_root)
                    sar_path = os.path.join(self.sar_root, rel_dir, raw_base_name + ".128x128.raw")

                    if os.path.exists(sar_path):
                        self.samples.append({"sar": sar_path, "label": label_path})

        if not self.samples:
            raise RuntimeError(
                f"No samples were found for mode '{self.mode}'. Please check your dataset at {search_path}."
            )

        # --- MODIFICATION START: Print class distribution ---
        self._print_class_distribution()
        # --- MODIFICATION END ---

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        sar_tensor = read_sar_complex_tensor(sample_paths["sar"], config.IMG_HEIGHT, config.IMG_WIDTH)
        label_tensor = torch.from_numpy(np.load(sample_paths["label"])).float()

        if sar_tensor is None:
            print(f"Warning: Failed to load sample at index {idx}. Trying next sample.")
            return self.__getitem__((idx + 1) % len(self))

        return sar_tensor, label_tensor

    def _print_class_distribution(self):
        """
        Helper function to count samples per class and print a summary.
        """
        class_counts = Counter()
        for sample in self.samples:
            path_parts = sample["label"].split(os.sep)
            # Find which target class this path belongs to
            for part in path_parts:
                if part in config.TARGET_CLASSES:
                    class_counts[part] += 1
                    break

        # --- Format and print the summary ---
        print("\n" + "=" * 50)
        print(f"Dataset Initialized: {self.mode.upper()}")
        print(f"Total Samples: {len(self.samples)}")
        print("Class Distribution:")
        for target_class, count in sorted(class_counts.items()):
            print(f"  - {target_class}: {count} samples")
        print("=" * 50 + "\n")
