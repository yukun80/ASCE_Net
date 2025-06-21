import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset
from utils.loss import ASCMultiTaskLoss
from model.asc_net import ASCNet_v3_5param


# --- MODIFICATION: Added a dedicated validation function ---
def validate_model(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    # Store losses for each component
    h_loss_total, amp_loss_total, alpha_loss_total, off_loss_total = 0, 0, 0, 0

    with torch.no_grad():  # Disable gradient calculation
        for sar_images, labels in loader:
            sar_images = sar_images.to(device)
            labels = labels.to(device)

            predictions = model(sar_images)
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)

            if not torch.isnan(loss):
                total_loss += loss.item()
                h_loss_total += h_loss.item()
                amp_loss_total += amp_loss.item()
                alpha_loss_total += alpha_loss.item()
                off_loss_total += off_loss.item()

    avg_loss = total_loss / len(loader)
    avg_losses_components = {
        "heatmap": h_loss_total / len(loader),
        "amplitude": amp_loss_total / len(loader),
        "alpha": alpha_loss_total / len(loader),
        "offset": off_loss_total / len(loader),
    }

    return avg_loss, avg_losses_components


def main():
    # --- Setup ---
    writer = SummaryWriter(log_dir=f"runs/{config.MODEL_NAME[:-4]}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Data ---
    print("Loading datasets...")
    # --- MODIFICATION: Create separate train and validation datasets ---
    try:
        train_dataset = MSTAR_ASC_5CH_Dataset(mode="train")
        val_dataset = MSTAR_ASC_5CH_Dataset(mode="val")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing datasets: {e}")
        print(
            "Please ensure you have run 'script/step4_label_preprocess.py' and that paths in 'utils/config.py' are correct."
        )
        return

    # Note: Set num_workers=0 on Windows to avoid potential issues
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- Model ---
    print("Initializing model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)

    # --- Loss and Optimizer ---
    criterion = ASCMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.1, verbose=True)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float("inf")  # --- MODIFICATION: Track best VALIDATION loss

    for epoch in range(config.NUM_EPOCHS):
        model.train()  # Set model to training mode
        epoch_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]", leave=False)
        for sar_images, labels in progress_bar:
            sar_images = sar_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions = model(sar_images)
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected during training. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            progress_bar.set_postfix(train_loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)

        # --- MODIFICATION: Run validation at the end of each epoch ---
        avg_val_loss, val_loss_components = validate_model(model, val_loader, criterion, config.DEVICE)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- Logging to TensorBoard ---
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        # Log validation loss components for better analysis
        for name, value in val_loss_components.items():
            writer.add_scalar(f"Loss_Val/{name}", value, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(avg_val_loss)

        # --- MODIFICATION: Advanced model saving logic ---
        # 1. Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path} with val_loss: {best_val_loss:.6f}")

        # 2. Save a checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"  -> Periodic checkpoint saved to {checkpoint_path}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
