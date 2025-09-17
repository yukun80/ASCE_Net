import os
import torch

"""
tmp_路径是把所有类型的数据都统一放在一个文件夹下，方便进行可视化查看
"""

# --- File and Directory Paths ---
# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for saving ASC .mat files
ASC_MAT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Training_ASC")

# Raw MSTAR data directory (containing .raw files)
SAR_RAW_ROOT = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_Data_Processed_raw")

# Directory to save processed labels (as .npy files)
LABEL_SAVE_ROOT_5CH = os.path.join(PROJECT_ROOT, "datasets", "SAR_ASC_Project", "tmp_MSTAR_ASC_LABELS")

# Directory for saving model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


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
