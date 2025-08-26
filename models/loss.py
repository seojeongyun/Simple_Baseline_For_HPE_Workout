# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True,
                 alpha_range=(8.0, 12.0), beta_range=(1.0, 1.5), gamma=2.0,
                 normalize_weights=True, eps=1e-6):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.alpha_lo, self.alpha_hi = alpha_range
        self.beta_lo, self.beta_hi = beta_range
        self.gamma = gamma
        self.normalize_weights = normalize_weights
        self.eps = eps

        # Initial values (used until the first epoch starts)
        self.alpha_cur = (self.alpha_lo + self.alpha_hi) * 0.5
        self.beta_cur = (self.beta_lo + self.beta_hi) * 0.5

    @torch.no_grad()
    def on_new_epoch(self, device=None):
        """Call once at the beginning of each epoch to resample alpha_cur and beta_cur."""
        device = device or torch.device('cpu')
        self.alpha_cur = torch.empty((), device=device).uniform_(self.alpha_lo, self.alpha_hi).item()
        self.beta_cur = torch.empty((), device=device).uniform_(self.beta_lo, self.beta_hi).item()

    def forward(self, output, target, target_weight):
        """
        output, target: [B, J, H, W], target in [0,1]
        target_weight: [B, J] or [B, J, 1]
        """
        # Key idea:
        # 1. per_pix = (pred - gt)^2
        #    - Background pixels (gt ~= 0) -> small error
        #    - Foreground pixels (gt ~= 1) -> large error if prediction is wrong
        # 2. w = beta + (alpha - beta) * (gt^gamma)
        #    - Background (gt ~= 0) -> w ~= beta (small weight)
        #    - Foreground (gt ~= 1) -> w ~= alpha (large weight)
        # 3. per_pix * w
        #    - Background: small error * small weight -> almost ignored
        #    - Foreground: large error * large weight -> strongly emphasized
        # 4. den = (w * tw).sum()
        #    - Normalize by total weight -> compute a weighted average
        #    - Keeps loss scale stable regardless of foreground/background ratio
        #
        B, J, H, W = output.shape
        HW = H * W
        pred = output.view(B, J, HW)
        gt = target.view(B, J, HW)

        # Use fixed (alpha_cur, beta_cur)
        alpha = torch.tensor(self.alpha_cur, device=output.device)
        beta = torch.tensor(self.beta_cur, device=output.device)

        loss = 0.0
        for idx in range(J):
            pred_i = pred[:, idx, :]  # [B, HW]
            gt_i = gt[:, idx, :]  # [B, HW]
            per_pix = (pred_i - gt_i) ** 2  # [B, HW]

            # soft weight: peak ?, background ?
            w_raw = beta + (alpha - beta) * (gt_i.clamp(0, 1) ** self.gamma)  # [B, HW]

            # Normalize to have mean 1 per batch (stabilize scale)
            if self.normalize_weights:
                w = w_raw / (w_raw.mean(dim=1, keepdim=True).clamp_min(self.eps))
            else:
                w = w_raw

            if self.use_target_weight:
                if target_weight.dim() == 3:  # [B, J, 1]
                    tw = target_weight[:, idx, 0]
                else:  # [B, J]
                    tw = target_weight[:, idx]
                tw = tw.view(B, 1)  # [B, 1]

                num = (per_pix * w * tw).sum()
                den = (w * tw).sum().clamp_min(self.eps)
                loss_i = 0.5 * (num / den)
            else:
                num = (per_pix * w).sum()
                den = w.sum().clamp_min(self.eps)
                loss_i = 0.5 * (num / den)

            loss += loss_i

        return loss / J