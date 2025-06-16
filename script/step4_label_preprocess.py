import os
import sys
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# --- Setup Project Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils import config  # noqa: E402


def gaussian_2d(shape, sigma=1):
    """Generates a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def model_to_pixel(x_model, y_model):
    """
    Converts physical model coordinates to pixel coordinates.
    This function must be consistent across the project.
    """
    C1 = config.IMG_HEIGHT / (0.3 * config.P_GRID_SIZE)
    C2 = config.IMG_WIDTH / 2 + 1

    col_pixel = (x_model * C1) + C2
    row_pixel = C2 - (y_model * C1)
    return row_pixel, col_pixel


def main():
    # Ensure output directory exists
    os.makedirs(config.LABEL_SAVE_ROOT_5CH, exist_ok=True)

    print(f"Reading .mat files from: {config.ASC_MAT_ROOT}")
    print(f"Saving 5-channel .npy labels to: {config.LABEL_SAVE_ROOT_5CH}")

    # Process all subdirectories
    for root, _, fnames in sorted(os.walk(config.ASC_MAT_ROOT)):
        print(f"\nProcessing directory: {root}")
        for fname in tqdm(sorted(fnames), desc="Generating Labels"):
            if not fname.endswith("_yang.mat"):
                continue

            mat_path = os.path.join(root, fname)

            try:
                mat_data = sio.loadmat(mat_path)
                scatter_all = mat_data["scatter_all"][0]
            except (KeyError, IndexError, FileNotFoundError) as e:
                print(f"Warning: Could not process {fname}. Error: {e}")
                continue

            # Initialize the 5-channel label map
            # 0: Heatmap, 1: Amplitude, 2: Alpha, 3: X-offset, 4: Y-offset
            label_map = np.zeros((5, config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.float32)

            # Generate a 7x7 Gaussian kernel for the heatmap
            gaussian_kernel = gaussian_2d((7, 7), sigma=1)

            for item in scatter_all:
                params = item[0]
                # [x, y, alpha, gamma, phi_prime, L, A]
                x_m, y_m, alpha, A = params[0], params[1], params[2], params[6]

                # We only model localized scatterers, so we can ignore L > 0 if needed
                # if params[5] > 0: continue

                # --- Calculate precise pixel location and integer center ---
                row_f, col_f = model_to_pixel(x_m, y_m)
                row_c, col_c = int(row_f), int(col_f)

                # Check if the center is within the image bounds
                if not (0 <= row_c < config.IMG_HEIGHT and 0 <= col_c < config.IMG_WIDTH):
                    continue

                # --- Calculate sub-pixel offsets ---
                dy = row_f - row_c
                dx = col_f - col_c

                # --- Draw the Gaussian on the heatmap ---
                # This part places the Gaussian kernel onto the heatmap channel
                left, right = min(col_c, 3), min(config.IMG_WIDTH - 1 - col_c, 3)
                top, bottom = min(row_c, 3), min(config.IMG_HEIGHT - 1 - row_c, 3)

                masked_kernel = gaussian_kernel[3 - top : 3 + bottom + 1, 3 - left : 3 + right + 1]

                # Ensure we don't overwrite a stronger peak
                label_map[0, row_c - top : row_c + bottom + 1, col_c - left : col_c + right + 1] = np.maximum(
                    label_map[0, row_c - top : row_c + bottom + 1, col_c - left : col_c + right + 1], masked_kernel
                )

                # --- Populate the regression channels at the integer center pixel ---
                label_map[1, row_c, col_c] = np.log1p(A)  # Use log1p for amplitude stability
                label_map[2, row_c, col_c] = alpha
                label_map[3, row_c, col_c] = dx
                label_map[4, row_c, col_c] = dy

            # Save the 5-channel .npy file
            output_filename = fname.replace("_yang.mat", "_5ch.npy")
            relative_path = os.path.relpath(root, config.ASC_MAT_ROOT)
            output_dir = os.path.join(config.LABEL_SAVE_ROOT_5CH, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, output_filename), label_map)

    print("\nLabel generation complete.")


if __name__ == "__main__":
    main()
