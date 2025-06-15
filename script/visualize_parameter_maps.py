import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from tqdm import tqdm

# --- Setup Project Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports from your project ---
from model.asc_net import ASCNetComplex_v2
from utils import config
from utils.dataset import MSTAR_ASC_Dataset, read_sar_complex_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def visualize_single_prediction(model, sample, output_dir):
    base_name = sample["base_name"]
    paths = sample["paths"]

    # 1. Load Data
    # --- FIX: Use the correct dictionary keys ---
    sar_tensor = read_sar_complex_tensor(paths["sar_path"], config.IMG_HEIGHT, config.IMG_WIDTH)
    if sar_tensor is None:
        print(f"Skipping {base_name} due to loading error.")
        return
    sar_tensor = sar_tensor.unsqueeze(0).to(config.DEVICE)

    true_A = np.load(paths["label_A_path"])
    true_alpha = np.load(paths["label_alpha_path"])
    # --- END FIX ---

    # 2. Perform Inference
    with torch.no_grad():
        predicted_maps = model(sar_tensor).squeeze(0).cpu().numpy()

    # 3. Post-process Predictions
    pred_A = np.expm1(predicted_maps[0])
    pred_alpha = predicted_maps[1]
    pred_alpha[true_A == 0] = 0

    # 4. Create Visualization Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"ASC Parameter Map Comparison: {base_name}", fontsize=16)

    # Plotting code remains the same...
    ax = axes[0, 0]
    sar_magnitude = torch.abs(sar_tensor.squeeze(0).cpu()).numpy()[0]
    ax.imshow(sar_magnitude, cmap="gray")
    ax.set_title("Input SAR Image (Magnitude)")
    ax.axis("off")
    axes[1, 0].axis("off")

    max_val_A = max(np.max(true_A), np.max(pred_A), 1e-9)
    norm_A = SymLogNorm(linthresh=0.01, vmin=0, vmax=max_val_A)
    ax = axes[0, 1]
    ax.imshow(true_A, cmap="viridis", norm=norm_A)
    ax.set_title("Ground Truth Amplitude (A)")
    ax.axis("off")
    ax = axes[0, 2]
    im_A = ax.imshow(pred_A, cmap="viridis", norm=norm_A)
    ax.set_title("Predicted Amplitude (A)")
    ax.axis("off")
    fig.colorbar(im_A, ax=axes[0, :], shrink=0.8, pad=0.02)

    norm_alpha = plt.Normalize(vmin=-1, vmax=1)
    ax = axes[1, 1]
    ax.imshow(true_alpha, cmap="coolwarm", norm=norm_alpha)
    ax.set_title("Ground Truth Alpha (α)")
    ax.axis("off")
    ax = axes[1, 2]
    im_alpha = ax.imshow(pred_alpha, cmap="coolwarm", norm=norm_alpha)
    ax.set_title("Predicted Alpha (α)")
    ax.axis("off")
    fig.colorbar(im_alpha, ax=axes[1, :], shrink=0.8, pad=0.02)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"param_map_{base_name}.png")
    plt.savefig(save_path)
    plt.close(fig)


# main function remains the same as previously provided
def main():
    print("Loading the trained model...")
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"Model loaded successfully from {model_path}")

    output_dir = os.path.join(project_root, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualization plots will be saved in: {output_dir}")

    print("Discovering dataset samples...")
    dataset = MSTAR_ASC_Dataset(sar_root=config.SAR_RAW_ROOT, label_root=config.LABEL_SAVE_ROOT)
    if len(dataset.samples) == 0:
        print("No valid samples found. Check paths in 'utils/config.py'.")
        return
    print(f"Found {len(dataset.samples)} valid samples to process.")

    for sample_info in tqdm(dataset.samples, desc="Generating Visualizations"):
        label_a_path = os.path.normpath(sample_info["label_A_path"])
        base_name = os.path.basename(label_a_path).replace("_A.npy", "").replace("_yang", "")
        full_sample_details = {"base_name": base_name, "paths": sample_info}
        visualize_single_prediction(model, full_sample_details, output_dir)

    print(f"\nParameter map visualization complete. Plots saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
