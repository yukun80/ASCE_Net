import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import config
from utils.dataset import MSTAR_ASC_5CH_Dataset
from utils.loss import ASCMultiTaskLoss
from model.asc_net import ASCNet_v3_5param # Import the new model

def main():
    # --- Setup ---
    writer = SummaryWriter(log_dir=f'runs/{config.MODEL_NAME[:-4]}')
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # --- Data ---
    print("Loading dataset...")
    dataset = MSTAR_ASC_5CH_Dataset()
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # --- Model ---
    print("Initializing model...")
    model = ASCNet_v3_5param(n_channels=1, n_params=5).to(config.DEVICE)
    
    # --- Loss and Optimizer ---
    criterion = ASCMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        for sar_images, labels in progress_bar:
            sar_images = sar_images.squeeze(1).to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(sar_images)
            
            # Calculate loss
            loss, h_loss, amp_loss, alpha_loss, off_loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # --- Logging ---
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/heatmap', h_loss.item(), epoch)
        writer.add_scalar('Loss/amplitude', amp_loss.item(), epoch)
        writer.add_scalar('Loss/alpha', alpha_loss.item(), epoch)
        writer.add_scalar('Loss/offset', off_loss.item(), epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # --- Validation & Saving ---
        # (A simple validation step could be added here if you split your data)
        scheduler.step(avg_loss)
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME))

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()