import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

# Ensure project root is in path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model.asc_net import ASCNetComplex_v2
from utils import config
from utils.dataset import read_sar_complex_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def visualize_single_prediction(model, sample, output_dir):
    paths = sample["paths"]
    base_name = sample["base_name"]

    # 1. Load data
    sar_tensor = (
        read_sar_complex_tensor(paths["sar"], config.IMG_HEIGHT, config.IMG_WIDTH).unsqueeze(0).to(config.DEVICE)
    )
    true_A = np.load(paths["A"])
    true_alpha = np.load(paths["alpha"])

    # 2. Predict
    with torch.no_grad():
        predicted_maps = model(sar_tensor).squeeze(0).cpu().numpy()

    pred_A = np.expm1(predicted_maps[0])
    pred_alpha = predicted_maps[1]

    # Mask out predictions where the ground truth is zero for a cleaner alpha comparison
    pred_alpha[true_A == 0] = 0

    # 3. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Parameter Map Comparison for: {base_name}", fontsize=16)

    # --- Input SAR Image ---
    ax = axes[0, 0]
    sar_mag = torch.abs(sar_tensor.squeeze(0).cpu()).numpy()[0]
    im = ax.imshow(sar_mag, cmap="gray")
    ax.set_title("Input SAR Image (Magnitude)")
    ax.axis("off")

    # --- Amplitude (A) Comparison ---
    norm_A = SymLogNorm(linthresh=0.01, vmin=0, vmax=max(np.max(true_A), np.max(pred_A), 1e-9))
    ax = axes[0, 1]
    ax.imshow(true_A, cmap="viridis", norm=norm_A)
    ax.set_title("Ground Truth Amplitude (A)")
    ax.axis("off")

    ax = axes[0, 2]
    im = ax.imshow(pred_A, cmap="viridis", norm=norm_A)
    ax.set_title("Predicted Amplitude (A)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Alpha (α) Comparison ---
    axes[1, 0].axis("off")  # Empty plot for alignment
    norm_alpha = plt.Normalize(vmin=-1, vmax=1)

    ax = axes[1, 1]
    ax.imshow(true_alpha, cmap="coolwarm", norm=norm_alpha)
    ax.set_title("Ground Truth Alpha (α)")
    ax.axis("off")

    ax = axes[1, 2]
    im = ax.imshow(pred_alpha, cmap="coolwarm", norm=norm_alpha)
    ax.set_title("Predicted Alpha (α)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"param_viz_{base_name}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to save memory


if __name__ == "__main__":
    # --- Setup ---
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}")

    output_dir = "./datasets/SAR_ASC_Project/tmp_Parameter_Visualization"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Parameter visualizations will be saved in: {output_dir}")

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

    for sample in tqdm(valid_samples, desc="Visualizing Parameters"):
        visualize_single_prediction(model, sample, output_dir)

    print("\nBatch visualization complete.")
