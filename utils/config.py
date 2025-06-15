import os
import torch

# --- File and Directory Paths ---
# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for saving ASC .mat files
ASC_MAT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")

# Raw MSTAR data directory (containing .raw files)
SAR_RAW_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Data_Processed_raw")

# Directory to save processed labels (as .npy files)
LABEL_SAVE_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_MSTAR_ASC_LABELS")

# Directory for saving model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model & Training Parameters ---
MODEL_NAME = "best_model.pth"
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Add this line to define the pixel size in meters
PIXEL_SPACING = 0.1  # Physical size of a pixel in meters (m/pixel)

# --- Loss Function Parameters ---
# Weight for the foreground (scatterer) pixels in the loss function
# This helps combat the class imbalance from sparse labels
LOSS_WEIGHT_FOREGROUND = 10.0

# --- Data Loader Parameters ---
BATCH_SIZE = 16
NUM_WORKERS = 4
