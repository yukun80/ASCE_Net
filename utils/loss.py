# utils/loss.py (优化版)
import torch
import torch.nn as nn


class FocusedASCLoss(nn.Module):
    def __init__(self, background_sample_ratio=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.background_sample_ratio = background_sample_ratio

    def forward(self, predicted_params, true_params):
        pred_A, pred_alpha = predicted_params[:, 0], predicted_params[:, 1]
        true_A, true_alpha = true_params[:, 0], true_params[:, 1]

        # 1. 找到前景（非零）和背景（零）像素的掩码
        foreground_mask = true_A > 0
        background_mask = ~foreground_mask

        # 2. 计算前景loss (这是最重要的部分)
        if foreground_mask.sum() > 0:
            loss_fg_A = self.mse_loss(pred_A[foreground_mask], true_A[foreground_mask])
            loss_fg_alpha = self.mse_loss(pred_alpha[foreground_mask], true_alpha[foreground_mask])
            foreground_loss = (loss_fg_A + loss_fg_alpha) / 2
        else:
            foreground_loss = 0.0

        # 3. 随机采样一部分背景像素来计算背景loss
        num_background = background_mask.sum()
        num_sample = int(num_background * self.background_sample_ratio)

        if num_sample > 0:
            # 随机选择背景像素的索引
            bg_indices = torch.where(background_mask)
            rand_indices = torch.randperm(len(bg_indices[0]))[:num_sample]

            sampled_bg_indices = (bg_indices[0][rand_indices], bg_indices[1][rand_indices], bg_indices[2][rand_indices])

            loss_bg_A = self.mse_loss(pred_A[sampled_bg_indices], true_A[sampled_bg_indices])
            loss_bg_alpha = self.mse_loss(pred_alpha[sampled_bg_indices], true_alpha[sampled_bg_indices])
            background_loss = (loss_bg_A + loss_bg_alpha) / 2
        else:
            background_loss = 0.0

        # 总loss是前景和背景loss的加权和，可以给予前景更高的权重
        total_loss = foreground_loss * 10.0 + background_loss

        return total_loss
