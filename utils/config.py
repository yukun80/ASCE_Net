import os
import torch

# --- File and Directory Paths ---
# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- NEW: Complete Dataset Paths ---
# Raw MSTAR data directory (containing .raw files with multi-level structure)
SAR_RAW_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "02_Data_Processed_raw")

# Directory for saving ASC .mat files (with multi-level structure)
ASC_MAT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "03_Training_ASC")

# Directory to save processed labels (as .npy files with multi-level structure)
LABEL_SAVE_ROOT_5CH = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "04_MSTAR_ASC_LABELS")

# --- Legacy paths for testing/debugging (keep for compatibility) ---
# Directory for saving ASC .mat files (testing)
ASC_MAT_ROOT_TMP = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")

# Raw MSTAR data directory (testing)
SAR_RAW_ROOT_TMP = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Data_Processed_raw")

# Directory to save processed labels (testing)
LABEL_SAVE_ROOT_5CH_TMP = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_MSTAR_ASC_LABELS")

# --- NEW: Dataset Split Configuration ---
TRAIN_FOLDER = "train_17_deg"
TEST_FOLDER = "test_15_deg"
TARGET_CLASSES = ["BTR70", "BMP2", "T72"]

# Directory for saving model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# --- NEW: Mode Configuration ---
# Set to True to use complete dataset, False to use tmp testing dataset
USE_COMPLETE_DATASET = True

# --- Model and Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Reduced batch size for the larger model/labels
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200
MODEL_NAME = "asc_net_v3_5param.pth"

# --- Image and Model Physical Parameters ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
P_GRID_SIZE = 84
PIXEL_SPACING = 0.1

# --- NEW: Weights for the Multi-Task Loss Function ---
LOSS_WEIGHT_HEATMAP = 1.0
LOSS_WEIGHT_AMP = 0.1
LOSS_WEIGHT_ALPHA = 0.5
LOSS_WEIGHT_OFFSET = 1.0


# --- Helper function to get current paths based on mode ---
def get_current_paths():
    """Returns current dataset paths based on USE_COMPLETE_DATASET flag"""
    if USE_COMPLETE_DATASET:
        return {"sar_raw_root": SAR_RAW_ROOT, "asc_mat_root": ASC_MAT_ROOT, "label_save_root": LABEL_SAVE_ROOT_5CH}
    else:
        return {
            "sar_raw_root": SAR_RAW_ROOT_TMP,
            "asc_mat_root": ASC_MAT_ROOT_TMP,
            "label_save_root": LABEL_SAVE_ROOT_5CH_TMP,
        }
