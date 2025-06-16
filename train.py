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


def main():
    # --- Setup ---
    writer = SummaryWriter(log_dir=f"runs/{config.MODEL_NAME[:-4]}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Data ---
    print("Loading dataset...")
    dataset = MSTAR_ASC_5CH_Dataset()
    if len(dataset) == 0:
        print(
            "Error: No samples found. Please run 'script/step4_label_preprocess_5channel.py' and check paths in 'utils/config.py'."
        )
        return

    # Note: Set num_workers=0 if you are on Windows to avoid potential issues
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- Model ---
    print("Initializing model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)

    # --- Loss and Optimizer ---
    criterion = ASCMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.1)

    # --- Training Loop ---
    print("Starting training...")
    best_loss = float("inf")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for i, (sar_images, labels) in enumerate(progress_bar):
            # --- FIX: Removed the erroneous .squeeze(1) operation ---
            # The DataLoader already provides the correct [N, C, H, W] shape.
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
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # --- Logging ---
        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/heatmap", h_loss.item(), epoch)
        writer.add_scalar("Loss/amplitude", amp_loss.item(), epoch)
        writer.add_scalar("Loss/alpha", alpha_loss.item(), epoch)
        writer.add_scalar("Loss/offset", off_loss.item(), epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))
            print(f"Model saved to {config.CHECKPOINT_DIR}/{config.MODEL_NAME} with new best loss: {best_loss:.6f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
