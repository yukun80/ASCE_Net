import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.sum()  # Return sum to normalize by number of scatterers later


class ASCMultiTaskLoss(nn.Module):
    """Multi-task loss for the 5-parameter ASC prediction."""

    def __init__(self):
        super(ASCMultiTaskLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = nn.L1Loss(reduction="sum")

        # Load weights from config for easy tuning
        self.w_heatmap = config.LOSS_WEIGHT_HEATMAP
        self.w_amp = config.LOSS_WEIGHT_AMP
        self.w_alpha = config.LOSS_WEIGHT_ALPHA
        self.w_offset = config.LOSS_WEIGHT_OFFSET

    def forward(self, prediction, target):
        # Unpack predictions and targets
        pred_heatmap, pred_A, pred_alpha, pred_dx, pred_dy = torch.split(prediction, 1, dim=1)
        gt_heatmap, gt_A, gt_alpha, gt_dx, gt_dy = torch.split(target, 1, dim=1)

        # --- Create mask to apply regression loss only at scatterer locations ---
        # The ground truth heatmap is 1 at peaks and decays, so we use a threshold.
        # Amplitude channel (gt_A) is a more reliable mask source.
        mask = gt_A > 0
        num_scatterers = mask.sum().float().clamp(min=1)

        # --- 1. Heatmap Loss (Focal Loss) ---
        loss_h = self.focal_loss(pred_heatmap, gt_heatmap) / num_scatterers

        # --- 2. Regression Losses (Masked L1) ---
        loss_amp = self.l1_loss(pred_A[mask], gt_A[mask]) / num_scatterers
        loss_alpha = self.l1_loss(pred_alpha[mask], gt_alpha[mask]) / num_scatterers
        loss_off = (
            self.l1_loss(pred_dx[mask], gt_dx[mask]) + self.l1_loss(pred_dy[mask], gt_dy[mask])
        ) / num_scatterers

        # --- 3. Total Weighted Loss ---
        total_loss = (
            self.w_heatmap * loss_h + self.w_amp * loss_amp + self.w_alpha * loss_alpha + self.w_offset * loss_off
        )

        return total_loss, loss_h, loss_amp, loss_alpha, loss_off
