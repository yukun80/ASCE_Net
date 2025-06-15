import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Setup Project Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports from your project ---
from model.asc_net import ASCNetComplex_v2
from utils import config
from utils.dataset import MSTAR_ASC_Dataset, read_sar_complex_tensor
from utils.reconstruction import extract_scatterers_from_maps, reconstruct_sar_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # --- 1. Setup Model ---
    print("Loading the trained model...")
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Setup Output Directory ---
    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reconstruction plots will be saved in: {output_dir}")

    # --- 3. Define SAR Imaging Parameters ---
    img_params = {"fc": 1e10, "B": 5e8, "om_deg": 2.86, "p": 84, "q": 128}

    # --- 4. Discover Samples ---
    print("Discovering dataset samples...")
    dataset = MSTAR_ASC_Dataset(sar_root=config.SAR_RAW_ROOT, label_root=config.LABEL_SAVE_ROOT)
    if not dataset.samples:
        print("No valid samples found. Check paths.")
        return
    print(f"Found {len(dataset.samples)} samples.")

    # --- 5. Process and Visualize ---
    for sample_info in tqdm(dataset.samples, desc="Generating Reconstructions"):
        label_a_path = os.path.normpath(sample_info["label_A_path"])
        base_name = os.path.basename(label_a_path).replace("_A.npy", "").replace("_yang", "")

        # a. Load original data and run inference
        sar_tensor, label_tensor = dataset.get_by_path(sample_info["sar_path"])  # Use a helper if needed or manual load
        if sar_tensor is None:
            continue

        true_A_log = label_tensor[0].numpy()
        true_alpha = label_tensor[1].numpy()
        true_A = np.expm1(true_A_log)

        with torch.no_grad():
            predicted_maps = model(sar_tensor.unsqueeze(0).to(config.DEVICE)).squeeze(0).cpu().numpy()
        pred_A = np.expm1(predicted_maps[0])
        pred_alpha = predicted_maps[1]

        # b. Extract scatterers from both ground truth and prediction
        # --- FIX: Call the function with the correct arguments ---
        gt_scatterers = extract_scatterers_from_maps(true_A, true_alpha, is_gt=True)
        pred_scatterers = extract_scatterers_from_maps(pred_A, pred_alpha, is_gt=False, threshold_abs=0.1)
        # --- END FIX ---

        # c. Reconstruct images
        recon_gt_img = reconstruct_sar_image(gt_scatterers, img_params)
        recon_pred_img = reconstruct_sar_image(pred_scatterers, img_params)

        # d. Visualize and Save
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Image Reconstruction Comparison: {base_name}", fontsize=16)

        # Define a common intensity range for better comparison
        vmax = np.percentile(np.abs(sar_tensor[0].numpy()), 99.5)

        axes[0].imshow(np.abs(sar_tensor[0].numpy()), cmap="gray", vmax=vmax)
        axes[0].set_title("Original SAR Image")
        axes[0].axis("off")

        axes[1].imshow(np.abs(recon_gt_img), cmap="gray", vmax=vmax)
        axes[1].set_title("Reconstruction from GT")
        axes[1].axis("off")

        axes[2].imshow(np.abs(recon_pred_img), cmap="gray", vmax=vmax)
        axes[2].set_title("Reconstruction from Prediction")
        axes[2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"reconstruction_{base_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print("\nReconstruction and visualization complete.")


# We need a helper in the Dataset class to load a single item by path for the script
def get_by_path(self, sar_path):
    # Find the corresponding sample dict
    for s in self.samples:
        if s["sar_path"] == sar_path:
            # Re-use the existing __getitem__ logic
            return self.__getitem__(self.samples.index(s))
    return None, None


# Add this helper method to the MSTAR_ASC_Dataset class in utils/dataset.py
MSTAR_ASC_Dataset.get_by_path = get_by_path

if __name__ == "__main__":
    main()
