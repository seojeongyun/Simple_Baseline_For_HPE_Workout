# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none') if use_target_weight else nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        """
        output, target: [B, J, H, W]
        target_weight: [B, J] ?? [B, J, 1] (float, 0/1)
        """
        eps = 1e-6
        B, J, H, W = output.shape
        HW = H * W

        # [B, J, HW]
        pred = output.view(B, J, HW)
        gt = target.view(B, J, HW)

        loss = 0.0
        for idx in range(J):
            # [B, HW]
            pred_i = pred[:, idx, :]
            gt_i = gt[:, idx, :]

            # MSE: [B, HW]
            per_pix = self.criterion(pred_i, gt_i)

            if self.use_target_weight:
                # [B]
                if target_weight.dim() == 3:  # if [B, J, 1]
                    target_weight_batch = target_weight[:, idx, 0]
                else:  # if [B, J]
                    target_weight_batch = target_weight[:, idx]

                # [B] -> [B,1]
                target_weight_batch_map = target_weight_batch.view(B, 1)

                #
                num = (per_pix * target_weight_batch_map).sum()  # Sum of pixel-wise losses from visible batches
                den = (target_weight_batch_map.sum() * HW).clamp_min(eps)  # Number of pixels from visible batches
                loss_i = 0.5 * (num / den)
            else:
                loss_i = 0.5 * per_pix.mean()

            loss += loss_i

        return loss / J