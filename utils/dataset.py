import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import config


def read_sar_complex_tensor(file_path, height, width):
    """
    Reads SAR complex data from a .raw file with the correct byte order
    and returns a complex tensor without altering its physical scale.
    """
    try:
        # --- FIX 1: Specify Big-Endian Byte Order ---
        # MSTAR .raw files are stored in big-endian format. '>f4' tells NumPy
        # to read 4-byte floats with big-endian byte order.
        data = np.fromfile(file_path, dtype=">f4")
        # --- END FIX 1 ---

        if data.size != height * width * 2:
            print(
                f"Warning: Unexpected data size in {file_path}. Expected {height*width*2}, got {data.size}. Skipping."
            )
            return None

        magnitude = data[: height * width].reshape(height, width)
        phase = data[height * width :].reshape(height, width)

        # Calculate real and imaginary parts from magnitude and phase
        real_part = np.nan_to_num(magnitude * np.cos(phase))
        imag_part = np.nan_to_num(magnitude * np.sin(phase))

        # --- FIX 2: Remove Data Normalization ---
        # The physics-based reconstruction requires the original, unscaled
        # physical values. The previous scaling to [-1, 1] was incorrect
        # for this task. We now directly use the raw complex values.
        complex_tensor = torch.complex(torch.from_numpy(real_part).float(), torch.from_numpy(imag_part).float())
        # --- END FIX 2 ---

        return complex_tensor.unsqueeze(0)

    except Exception as e:
        print(f"Error reading or processing raw file {file_path}: {e}")
        return None


class MSTAR_ASC_Dataset(Dataset):
    def __init__(self, sar_root, label_root):
        self.sar_root = sar_root
        self.label_root = label_root
        self.samples = []

        if not os.path.exists(label_root):
            print(f"Error: Label directory not found at {label_root}")
            return

        for dirpath, _, filenames in os.walk(self.label_root):
            for filename in filenames:
                if filename.endswith("_A.npy"):
                    base_name = filename.replace("_A.npy", "")
                    raw_base_name = base_name.replace("_yang", "")

                    # Construct full paths for all necessary files
                    paths = {
                        "label_A_path": os.path.join(dirpath, filename),
                        "label_alpha_path": os.path.join(dirpath, base_name + "_alpha.npy"),
                        "sar_path": os.path.join(
                            self.sar_root,
                            os.path.relpath(dirpath, self.label_root),
                            raw_base_name + ".128x128.raw",
                        ),
                    }

                    if all(os.path.exists(p) for p in paths.values()):
                        self.samples.append(paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        sar_complex_tensor = read_sar_complex_tensor(sample_paths["sar_path"], config.IMG_HEIGHT, config.IMG_WIDTH)

        # If a file is corrupted, load the next one instead
        if sar_complex_tensor is None:
            return self.__getitem__((idx + 1) % len(self))

        label_A = np.load(sample_paths["label_A_path"])
        label_alpha = np.load(sample_paths["label_alpha_path"])

        # Apply log1p transform to amplitude for training stability
        label_A[label_A > 0] = np.log1p(label_A[label_A > 0])

        label_tensor = torch.from_numpy(np.stack([label_A, label_alpha], axis=0)).float()

        return sar_complex_tensor, label_tensor


# Add this helper method to the MSTAR_ASC_Dataset class
def get_by_path(self, sar_path):
    for s in self.samples:
        if s["sar_path"] == sar_path:
            return self.__getitem__(self.samples.index(s))
    return None, None


MSTAR_ASC_Dataset.get_by_path = get_by_path
