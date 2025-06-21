# filename: train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset
from utils.loss import ASCMultiTaskLoss
from model.asc_net import ASCNet_v3_5param


def validate_model(model, loader, criterion, device):
    """Function to perform validation on the model."""
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for sar_images, labels in loader:
            sar_images = sar_images.to(device)
            labels = labels.to(device)

            predictions = model(sar_images)
            loss, _, _, _, _ = criterion(predictions, labels)

            if not torch.isnan(loss):
                total_val_loss += loss.item()

    return total_val_loss / len(loader)


def main():
    # --- Setup ---
    writer = SummaryWriter(log_dir=f"runs/{os.path.basename(config.MODEL_NAME).replace('.pth', '')}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Data Loading ---
    print("Loading datasets...")
    try:
        train_dataset = MSTAR_ASC_5CH_Dataset(mode="train")
        val_dataset = MSTAR_ASC_5CH_Dataset(mode="val")
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        print(
            f"Please ensure you have run 'script/step4_label_preprocess.py' and that paths in 'utils/config.py' are correct."
        )
        return

    if len(train_dataset) == 0:
        print(
            f"Error: No samples were found for mode 'train'. Please check your dataset at {os.path.join(config.LABEL_SAVE_ROOT_5CH, 'train_17_deg')}."
        )
        return
    if len(val_dataset) == 0:
        print(
            f"Warning: No samples were found for mode 'val'. Validation will be skipped. Check {os.path.join(config.LABEL_SAVE_ROOT_5CH, 'test_15_deg')}."
        )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")

    # --- Model, Loss, and Optimizer ---
    print("Initializing model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    criterion = ASCMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- THIS IS THE CORRECTED LINE ---
    # Removed the 'verbose=True' argument to support older PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.1)

    # --- Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]", leave=False)
        for sar_images, labels in progress_bar:
            sar_images = sar_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions = model(sar_images)
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected during training at epoch {epoch+1}. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)

        # --- Validation ---
        avg_val_loss = validate_model(model, val_loader, criterion, config.DEVICE)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- Manual 'verbose' for learning rate scheduler ---
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"  -> Learning rate reduced from {old_lr} to {new_lr}")

        # --- Logging ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        writer.add_scalar("LearningRate", new_lr, epoch)
        writer.add_scalar("Loss_components/heatmap", h_loss.item(), epoch)
        writer.add_scalar("Loss_components/amplitude", amp_loss.item(), epoch)
        writer.add_scalar("Loss_components/alpha", alpha_loss.item(), epoch)
        writer.add_scalar("Loss_components/offset", off_loss.item(), epoch)

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} with val loss: {best_val_loss:.6f}")

        # --- Save Periodic Checkpoint ---
        if (epoch + 1) % 10 == 0:
            periodic_save_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), periodic_save_path)
            print(f"  -> Periodic checkpoint saved to {periodic_save_path}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
