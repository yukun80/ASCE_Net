import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset
from utils.loss import ASCMultiTaskLoss
from model.asc_net import ASCNet_v3_5param  # Import the new model


def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    val_loss = 0
    val_h_loss = 0
    val_amp_loss = 0
    val_alpha_loss = 0
    val_off_loss = 0

    with torch.no_grad():
        for sar_images, labels in val_loader:
            sar_images = sar_images.to(device)
            labels = labels.to(device)

            predictions = model(sar_images)
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)

            val_loss += loss.item()
            val_h_loss += h_loss.item()
            val_amp_loss += amp_loss.item()
            val_alpha_loss += alpha_loss.item()
            val_off_loss += off_loss.item()

    num_batches = len(val_loader)
    return {
        "total": val_loss / num_batches,
        "heatmap": val_h_loss / num_batches,
        "amplitude": val_amp_loss / num_batches,
        "alpha": val_alpha_loss / num_batches,
        "offset": val_off_loss / num_batches,
    }


def main():
    # --- Setup ---
    writer = SummaryWriter(log_dir=f"runs/{config.MODEL_NAME[:-4]}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Data ---
    print("Loading datasets...")
    print(f"Dataset mode: {'Complete Dataset' if config.USE_COMPLETE_DATASET else 'Testing Dataset'}")

    if config.USE_COMPLETE_DATASET:
        # Load separate train and validation sets
        train_dataset = MSTAR_ASC_5CH_Dataset(split="train")
        val_dataset = MSTAR_ASC_5CH_Dataset(split="test")  # Use test set as validation

        if len(train_dataset) == 0:
            print("Error: No training samples found. Please run 'script/step4_label_preprocess.py' first.")
            return
        if len(val_dataset) == 0:
            print("Error: No validation samples found. Please run 'script/step4_label_preprocess.py' first.")
            return

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
        )

    else:
        # Use legacy approach for testing dataset
        dataset = MSTAR_ASC_5CH_Dataset()
        if len(dataset) == 0:
            print(
                "Error: No samples found. Please run 'script/step4_label_preprocess.py' and check paths in 'utils/config.py'."
            )
            return

        train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = None
        print(f"Dataset loaded with {len(dataset)} samples (no validation split).")

    # --- Model ---
    print("Initializing model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)

    # --- Loss and Optimizer ---
    criterion = ASCMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.1)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_h_loss = 0
        epoch_amp_loss = 0
        epoch_alpha_loss = 0
        epoch_off_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for i, (sar_images, labels) in enumerate(progress_bar):
            sar_images = sar_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(sar_images)

            # Calculate loss
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {i}. Skipping batch.")
                continue

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_h_loss += h_loss.item()
            epoch_amp_loss += amp_loss.item()
            epoch_alpha_loss += alpha_loss.item()
            epoch_off_loss += off_loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        avg_h_loss = epoch_h_loss / len(train_loader)
        avg_amp_loss = epoch_amp_loss / len(train_loader)
        avg_alpha_loss = epoch_alpha_loss / len(train_loader)
        avg_off_loss = epoch_off_loss / len(train_loader)

        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}")

        # --- Validation phase ---
        if val_loader is not None:
            val_losses = evaluate_model(model, val_loader, criterion, config.DEVICE)
            print(f"Epoch {epoch+1} Validation Loss: {val_losses['total']:.6f}")

            # Use validation loss for scheduling and saving
            scheduler.step(val_losses["total"])
            current_loss = val_losses["total"]

            # Log validation losses
            writer.add_scalar("Loss/val_total", val_losses["total"], epoch)
            writer.add_scalar("Loss/val_heatmap", val_losses["heatmap"], epoch)
            writer.add_scalar("Loss/val_amplitude", val_losses["amplitude"], epoch)
            writer.add_scalar("Loss/val_alpha", val_losses["alpha"], epoch)
            writer.add_scalar("Loss/val_offset", val_losses["offset"], epoch)

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
                print(f"Model saved with new best validation loss: {best_val_loss:.6f}")
        else:
            # Use training loss for scheduling and saving (legacy mode)
            scheduler.step(avg_train_loss)
            current_loss = avg_train_loss

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
                print(f"Model saved with new best training loss: {best_train_loss:.6f}")

        # --- Logging ---
        writer.add_scalar("Loss/train_total", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_heatmap", avg_h_loss, epoch)
        writer.add_scalar("Loss/train_amplitude", avg_amp_loss, epoch)
        writer.add_scalar("Loss/train_alpha", avg_alpha_loss, epoch)
        writer.add_scalar("Loss/train_offset", avg_off_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    writer.close()
    print("Training finished.")
    print(f"Best loss: {best_val_loss:.6f}" if val_loader else f"Best training loss: {best_train_loss:.6f}")


if __name__ == "__main__":
    main()
