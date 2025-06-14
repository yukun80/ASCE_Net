import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal.windows import taylor
from scipy.ndimage import maximum_filter
from tqdm import tqdm

# Ensure project root is in path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model.asc_net import ASCNetComplex_v2
from utils import config
from utils.dataset import read_sar_complex_tensor

# Set the OMP environment variable to avoid the library conflict error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def reconstruct_image_from_asc(asc_params, params):
    """
    Reconstructs a SAR image from a list of ASC parameters with corrected imaging algorithm.
    """
    # Unpack parameters
    fc, c, B, phi_m, N, M = params["fc"], params["c"], params["B"], params["phi_m"], params["N"], params["M"]

    # 1. Create frequency and angle vectors
    f = np.linspace(fc - B / 2, fc + B / 2, N)
    phi = np.linspace(-phi_m / 2, phi_m / 2, M)

    # 2. Simulate the backscattered signal in polar k-space
    E_polar = np.zeros((N, M), dtype=np.complex128)
    for i in range(asc_params.shape[0]):
        A, alpha, x, y = asc_params[i, :]
        phase_term = np.exp(-1j * 4 * np.pi / c * (x * np.cos(phi) + y * np.sin(phi)) * f[:, np.newaxis])
        freq_term = (1j * f / fc) ** alpha
        E_polar += A * np.multiply(phase_term.T, freq_term).T

    # 3. Define k-space coordinates for interpolation
    k = 2 * np.pi * f / c
    # Polar grid coordinates from our signal
    k_range_polar = k[:, np.newaxis] * np.cos(phi)
    k_azimuth_polar = k[:, np.newaxis] * np.sin(phi)

    # Target Cartesian grid
    k_range_max = np.max(np.abs(k_range_polar))
    k_azimuth_max = np.max(np.abs(k_azimuth_polar))
    k_range_vec = np.linspace(-k_range_max, k_range_max, N)
    k_azimuth_vec = np.linspace(-k_azimuth_max, k_azimuth_max, M)
    k_azimuth_grid, k_range_grid = np.meshgrid(k_azimuth_vec, k_range_vec)

    # Interpolate from polar grid to Cartesian grid
    points_polar = np.array([k_range_polar.flatten(), k_azimuth_polar.flatten()]).T
    values_polar = E_polar.flatten()
    E_cartesian = griddata(points_polar, values_polar, (k_range_grid, k_azimuth_grid), method="cubic", fill_value=0)

    # 4. Apply a 2D Taylor window
    win_range = taylor(N, nbar=4, sll=35)
    win_azimuth = taylor(M, nbar=4, sll=35)
    win_2d = np.outer(win_range, win_azimuth)
    E_cartesian_win = E_cartesian * win_2d

    # 5. Perform 2D IFFT to get the image
    # BUG FIX: Added a critical np.fft.ifftshift before the ifft2 call
    img = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(E_cartesian_win)))
    img_mag = np.abs(img)

    return img_mag


# --- The rest of the script (extract_asc_from_maps, main loop, etc.) remains the same ---
# (Helper functions from the previous response are omitted here for brevity but should be kept in your file)
def extract_asc_from_maps(pred_A, pred_alpha, threshold=0.1, peak_box_size=5):
    """
    Extracts sparse ASC parameters from dense prediction maps by finding peaks.
    """
    # Normalize amplitude for thresholding
    max_A = np.max(pred_A)
    if max_A == 0:
        return np.empty((0, 4))

    # Find peaks using a maximum filter
    local_max = maximum_filter(pred_A, footprint=np.ones((peak_box_size, peak_box_size)), mode="constant")
    peaks = pred_A == local_max

    # Apply threshold
    peaks[pred_A < (max_A * threshold)] = 0

    # Get coordinates of the peaks
    peak_coords = np.argwhere(peaks)

    asc_params = []
    for r, c in peak_coords:
        A = pred_A[r, c]
        alpha = pred_alpha[r, c]

        # Convert pixel coordinates (r, c) to spatial coordinates (x, y)
        x = (c - config.IMG_WIDTH / 2) * config.PIXEL_SPACING
        y = (config.IMG_HEIGHT / 2 - r) * config.PIXEL_SPACING  # Invert y-axis

        asc_params.append([A, alpha, x, y])

    return np.array(asc_params)


def extract_asc_from_labels(true_A, true_alpha):
    """
    Extracts sparse ASC parameters from ground truth label maps.
    """
    scatter_coords = np.argwhere(true_A > 0)

    asc_params = []
    for r, c in scatter_coords:
        A = true_A[r, c]
        alpha = true_alpha[r, c]

        x = (c - config.IMG_WIDTH / 2) * config.PIXEL_SPACING
        y = (config.IMG_HEIGHT / 2 - r) * config.PIXEL_SPACING

        asc_params.append([A, alpha, x, y])

    return np.array(asc_params)


if __name__ == "__main__":
    # --- Setup ---
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}")

    output_dir = "./datasets/SAR_ASC_Project/tmp_MSTAR_ASC_Reconstruct"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reconstructions will be saved in: {output_dir}")

    sim_params = {
        "fc": config.F_C,
        "c": config.C,
        "B": config.BANDWIDTH,
        "phi_m": config.PHI_M,
        "N": config.IMG_HEIGHT,
        "M": config.IMG_WIDTH,
    }

    # --- Batch Processing ---
    valid_samples = []
    # ... (Discovery logic remains the same) ...
    for dirpath, _, filenames in os.walk(config.LABEL_SAVE_ROOT):
        for filename in filenames:
            if filename.endswith("_A.npy"):
                base_name = filename.replace("_A.npy", "")
                raw_base_name = base_name.replace("_yang", "")

                paths = {
                    "A": os.path.join(dirpath, filename),
                    "alpha": os.path.join(dirpath, base_name + "_alpha.npy"),
                    "sar": os.path.join(
                        config.SAR_RAW_ROOT,
                        os.path.relpath(dirpath, config.LABEL_SAVE_ROOT),
                        raw_base_name + ".128x128.raw",
                    ),
                }

                if all(os.path.exists(p) for p in paths.values()):
                    valid_samples.append({"base_name": raw_base_name, "paths": paths})

    print(f"\nFound {len(valid_samples)} valid samples to process.")

    for sample in tqdm(valid_samples, desc="Reconstructing Images"):
        # ... (Main loop logic remains the same) ...
        paths = sample["paths"]

        sar_tensor = read_sar_complex_tensor(paths["sar"], config.IMG_HEIGHT, config.IMG_WIDTH)
        sar_tensor = sar_tensor.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            predicted_maps = model(sar_tensor).squeeze(0).cpu().numpy()

        pred_A = np.expm1(predicted_maps[0])
        pred_alpha = predicted_maps[1]

        asc_pred = extract_asc_from_maps(pred_A, pred_alpha, threshold=0.2)

        true_A = np.load(paths["A"])
        true_alpha = np.load(paths["alpha"])
        asc_gt = extract_asc_from_labels(true_A, true_alpha)

        if asc_pred.shape[0] > 0:
            img_pred = reconstruct_image_from_asc(asc_pred, sim_params)
        else:
            img_pred = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH))

        if asc_gt.shape[0] > 0:
            img_gt = reconstruct_image_from_asc(asc_gt, sim_params)
        else:
            img_gt = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH))

        vmax = max(np.max(img_gt), np.max(img_pred), 1e-9)

        plt.imsave(
            os.path.join(output_dir, f"{sample['base_name']}_gt_recon.png"), img_gt, cmap="gray", vmin=0, vmax=vmax
        )
        plt.imsave(
            os.path.join(output_dir, f"{sample['base_name']}_pred_recon.png"), img_pred, cmap="gray", vmin=0, vmax=vmax
        )

    print("\nBatch reconstruction complete.")
