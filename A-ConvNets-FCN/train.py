import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from dataset import ASC2ChannelDataset  # noqa: E402
from losses import asc_regression_loss  # noqa: E402
from model import AConvFCN  # noqa: E402
from utils.io import ensure_dir, load_config  # noqa: E402
from utils.seed import seed_worker, set_seed  # noqa: E402

"""
python A-ConvNets-FCN/train.py --config config.yaml
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A-ConvNets-FCN comparison model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label-on-the-fly", action="store_true", help="Convert labels from 5-channel during loading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device_cfg = cfg["training"].get("device", "auto")
    device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else device_cfg)

    dataset = ASC2ChannelDataset(config_path=args.config, label_on_the_fly=args.label_on_the_fly)
    val_split = cfg["training"].get("val_split", 0.2)
    # Build validation subset indices (overlap with training allowed)
    total_len = len(dataset)
    val_len = max(1, int(total_len * val_split))
    perm = torch.randperm(total_len)
    val_indices = perm[:val_len].tolist()
    val_set = Subset(dataset, val_indices)

    batch_size = cfg["training"].get("batch_size", 32)
    num_workers = cfg["training"].get("num_workers", 0)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = AConvFCN(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=cfg["model"].get("out_channels", 2),
        kernel_size=cfg["model"].get("conv_kernel", 5),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"].get("epochs", 100))

    save_dir = ensure_dir(cfg["training"].get("save_dir", "outputs/checkpoints"))
    log_dir = ensure_dir(cfg["training"].get("log_dir", "outputs/logs"))
    writer = SummaryWriter(log_dir=log_dir)

    best_val = float("inf")
    epochs = cfg["training"].get("epochs", 100)
    train_len = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss, channel_losses = asc_regression_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / train_len

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss, _ = asc_regression_loss(preds, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= max(val_len, 1)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path(save_dir) / "aconv_fcn_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val": best_val,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"Epoch {epoch+1}: new best val loss {best_val:.6f}, saved to {ckpt_path}")
        else:
            print(f"Epoch {epoch+1}: train {train_loss:.6f}, val {val_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
