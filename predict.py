import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# --- Ensure project root is in path ---
# This allows the script to find your custom modules like utils and model
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.append(project_root)  # 将项目根目录添加到系统路径
# --- End path setup ---

from model.asc_net import ASCNetComplex_v2  # noqa: E402
from utils import config  # noqa: E402
from utils.dataset import read_sar_complex_tensor  # noqa: E402


def predict_and_visualize(model, sar_image_path, label_A_path, label_alpha_path):
    """
    Loads a single SAR image, performs inference, and visualizes the results.
    """
    # 1. Set model to evaluation mode
    model.eval()

    # 2. Load and preprocess the input SAR image
    sar_tensor = read_sar_complex_tensor(sar_image_path, config.IMG_HEIGHT, config.IMG_WIDTH)
    if sar_tensor is None:
        print(f"Failed to load or process SAR image: {sar_image_path}")
        return

    # Add a batch dimension and send to device
    sar_tensor = sar_tensor.unsqueeze(0).to(config.DEVICE)

    # 3. Load ground truth labels
    true_A = np.load(label_A_path)
    true_alpha = np.load(label_alpha_path)

    # 4. Perform inference
    with torch.no_grad():
        predicted_params = model(sar_tensor)

    # Move predicted tensor to CPU and remove batch dimension
    predicted_params_np = predicted_params.squeeze(0).cpu().numpy()

    # 5. Post-process the predictions
    # The model predicts log(1+A), so we need to revert it
    # We also clip to avoid potential numerical instability from exp()
    predicted_A = np.expm1(predicted_params_np[0])
    predicted_A = np.clip(predicted_A, 0, 1e6)  # Clip at a large value

    # Alpha is already in the correct range due to tanh activation
    predicted_alpha = predicted_params_np[1]

    # Mask out predictions where the ground truth is zero for a cleaner alpha comparison
    zero_mask = true_A == 0
    predicted_alpha[zero_mask] = 0

    # 6. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Model Prediction for: {os.path.basename(sar_image_path)}", fontsize=16)

    # --- Row 1: Input and Amplitude ---
    # Input SAR Image Magnitude
    ax = axes[0, 0]
    im = ax.imshow(torch.abs(sar_tensor.squeeze(0).cpu()).numpy()[0], cmap="gray")
    ax.set_title("Input SAR Image (Magnitude)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    # Ground Truth Amplitude
    ax = axes[0, 1]
    # Use LogNorm to better visualize sparse, high-dynamic-range data
    norm_A = SymLogNorm(linthresh=0.01, vmin=np.min(true_A), vmax=np.max(true_A))
    im = ax.imshow(true_A, cmap="viridis", norm=norm_A)
    ax.set_title("Ground Truth Amplitude (A)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    # Predicted Amplitude
    ax = axes[0, 2]
    im = ax.imshow(predicted_A, cmap="viridis", norm=norm_A)
    ax.set_title("Predicted Amplitude (A)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    # --- Row 2: Alpha Parameter ---
    axes[1, 0].axis("off")  # Empty plot for alignment

    # Ground Truth Alpha
    ax = axes[1, 1]
    # Use a diverging colormap since alpha is in [-1, 1]
    im = ax.imshow(true_alpha, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Ground Truth Alpha (α)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    # Predicted Alpha
    ax = axes[1, 2]
    im = ax.imshow(predicted_alpha, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Predicted Alpha (α)")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"prediction_{os.path.basename(sar_image_path)}.png")
    plt.savefig(save_path)
    print(f"Prediction saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    # 1. Initialize the model
    model = ASCNetComplex_v2(n_channels=1, n_params=2).to(config.DEVICE)

    # 2. Load the trained weights
    model_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using train.py")
    else:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print(f"Model loaded from {model_path}")

        # 3. Find a sample to test
        # We walk through the label directory to find a sample that exists
        # This is more robust than hardcoding a path
        sample_found = False
        for dirpath, _, filenames in os.walk(config.LABEL_SAVE_ROOT):
            for filename in filenames:
                if filename.endswith("_A.npy"):
                    # Construct paths for all necessary files
                    label_A_path = os.path.join(dirpath, filename)
                    base_name = filename.replace("_A.npy", "")
                    label_alpha_path = os.path.join(dirpath, base_name + "_alpha.npy")

                    relative_path = os.path.relpath(dirpath, config.LABEL_SAVE_ROOT)
                    raw_base_name = base_name.replace("_yang", "")
                    sar_raw_filename = raw_base_name + ".128x128.raw"
                    sar_image_path = os.path.join(config.SAR_RAW_ROOT, relative_path, sar_raw_filename)

                    if os.path.exists(label_alpha_path) and os.path.exists(sar_image_path):
                        print("\n--- Found a valid sample to test ---")
                        print(f"SAR Image: {sar_image_path}")
                        print(f"Label A: {label_A_path}")
                        print(f"Label Alpha: {label_alpha_path}")

                        # Use the first valid sample found
                        predict_and_visualize(model, sar_image_path, label_A_path, label_alpha_path)
                        sample_found = True
                        break
            if sample_found:
                break

        if not sample_found:
            print("\nCould not find any valid samples to process.")
            print("Please ensure your dataset is correctly preprocessed and paths in config.py are correct.")
