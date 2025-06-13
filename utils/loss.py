# utils/loss.py

import torch
import torch.nn as nn


class ASCLoss(nn.Module):
    def __init__(self, weight_nonzero=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.weight_nonzero = weight_nonzero

    def forward(self, predicted_params, true_params):
        """
        predicted_params: 模型输出的参数图像张量, shape (B, 2, H, W)
        true_params: 真实的参数图像标签, shape (B, 2, H, W)
        """
        pred_A, pred_alpha = predicted_params[:, 0, :, :], predicted_params[:, 1, :, :]
        true_A, true_alpha = true_params[:, 0, :, :], true_params[:, 1, :, :]

        # 1. 计算整个图像的MSE loss (对应论文中的 L_MSE)
        loss_mse_A = self.mse_loss(pred_A, true_A)
        loss_mse_alpha = self.mse_loss(pred_alpha, true_alpha)
        total_mse_loss = (loss_mse_A + loss_mse_alpha) / 2

        # 2. 计算非零区域的MSE loss (对应论文中的 L_MSE_nonzero)
        # 创建一个mask，其中散射中心位置为True
        nonzero_mask = true_A > 0

        # 如果一个batch中没有任何散射中心，非零loss为0
        if nonzero_mask.sum() > 0:
            loss_nonzero_A = self.mse_loss(pred_A[nonzero_mask], true_A[nonzero_mask])
            loss_nonzero_alpha = self.mse_loss(pred_alpha[nonzero_mask], true_alpha[nonzero_mask])
            total_nonzero_loss = (loss_nonzero_A + loss_nonzero_alpha) / 2
        else:
            total_nonzero_loss = 0.0

        # 最终loss是两部分之和 (对应论文中的 Equation 8)
        final_loss = total_mse_loss + self.weight_nonzero * total_nonzero_loss

        return final_loss
