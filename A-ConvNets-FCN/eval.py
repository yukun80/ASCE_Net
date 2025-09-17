import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from dataset import ASC2ChannelDataset  # noqa: E402
from model import AConvFCN  # noqa: E402
from utils.io import load_config  # noqa: E402
from utils.peaks import argmax2d  # noqa: E402
from utils.seed import seed_worker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-ConvNets-FCN model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-on-the-fly", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    dataset = ASC2ChannelDataset(config_path=args.config, label_on_the_fly=args.label_on_the_fly)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = AConvFCN(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=cfg["model"].get("out_channels", 2),
        kernel_size=cfg["model"].get("conv_kernel", 5),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    amp_mae = 0.0
    alpha_mae = 0.0
    pos_mae = 0.0
    samples = 0

    height = cfg["data"].get("image_height", 128)
    width = cfg["data"].get("image_width", 128)
    center_x = width // 2
    center_y = height // 2

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            for pred_map, gt_map in zip(preds, labels):
                amp_pred = pred_map[0].cpu()
                alpha_pred = pred_map[1].cpu()
                amp_gt = gt_map[0].cpu()
                alpha_gt = gt_map[1].cpu()

                pr, pc = argmax2d(amp_pred)
                gr, gc = argmax2d(amp_gt)

                amp_mae += abs(amp_pred[pr, pc].item() - amp_gt[gr, gc].item())
                alpha_mae += abs(alpha_pred[pr, pc].item() - alpha_gt[gr, gc].item())

                pred_pos = (pc - center_x, pr - center_y)
                gt_pos = (gc - center_x, gr - center_y)
                pos_mae += abs(pred_pos[0] - gt_pos[0]) + abs(pred_pos[1] - gt_pos[1])
                samples += 1

    if samples == 0:
        print("No samples evaluated.")
        return

    print(f"Amplitude MAE: {amp_mae / samples:.6f}")
    print(f"Alpha MAE: {alpha_mae / samples:.6f}")
    print(f"Position L1 (pixels): {pos_mae / samples:.6f}")


if __name__ == "__main__":
    main()
