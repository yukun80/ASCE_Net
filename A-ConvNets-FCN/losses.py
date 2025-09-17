from typing import Tuple

import torch
import torch.nn.functional as F


def mse_with_mask(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error over non-zero target elements."""
    mask = target != 0
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred[mask], target[mask])


def asc_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loss = sum(MSE_full + MSE_nz) for each channel."""
    if pred.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape")

    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    channel_losses = []

    for ch in range(pred.shape[1]):
        pred_ch = pred[:, ch]
        target_ch = target[:, ch]
        loss_full = F.mse_loss(pred_ch, target_ch)
        loss_nz = mse_with_mask(pred_ch, target_ch)
        channel_loss = loss_full + loss_nz
        channel_losses.append(channel_loss.detach())
        total_loss = total_loss + channel_loss

    return total_loss, torch.stack(channel_losses)
