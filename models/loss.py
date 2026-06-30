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
    def __init__(self, use_target_weight=True, use_joint_weighted_loss=False,
                 alpha_range=(8.0, 12.0), beta_range=(1.0, 1.5), gamma=2.0,
                 normalize_weights=True, eps=1e-6):
        super().__init__()
        self.register_buffer(
            'joint_weights',
            torch.tensor([
                1.0,  # Nose
                1.0,  # Left Eye
                1.0,  # Right Eye
                1.0,  # Left Ear
                1.0,  # Right Ear
                2.0,  # Left Shoulder
                2.0,  # Right Shoulder
                2.0,  # Left Elbow
                2.0,  # Right Elbow
                1.5,  # Left Wrist
                1.5,  # Right Wrist
                1.5,  # Left Hip
                1.5,  # Right Hip
                3.0,  # Left Knee
                3.0,  # Right Knee
                2.0,  # Left Ankle
                2.0,  # Right Ankle
                1.0,  # Neck
                1.5,  # Left Palm
                1.5,  # Right Palm
                1.0,  # Back
                1.0,  # Waist
                2.0,  # Left Foot
                2.0,  # Right Foot
            ], dtype=torch.float32))
        #
        self.use_target_weight = use_target_weight
        self.joint_weighted_loss = use_joint_weighted_loss

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
        alpha = self.alpha_cur
        beta = self.beta_cur

        loss = output.new_tensor(0.0)
        weight_sum = output.new_tensor(0.0)

        for idx in range(J):
            pred_i = pred[:, idx, :]
            gt_i = gt[:, idx, :]
            per_pix = (pred_i - gt_i) ** 2

            w_raw = beta + (alpha - beta) * (gt_i.clamp(0, 1) ** self.gamma)

            if self.normalize_weights:
                w = w_raw / (
                    w_raw.mean(dim=1, keepdim=True).clamp_min(self.eps)
                )
            else:
                w = w_raw

            if self.use_target_weight:
                if target_weight.dim() == 3:
                    tw = target_weight[:, idx, 0]
                else:
                    tw = target_weight[:, idx]

                tw = tw.view(B, 1)

                num = (per_pix * w * tw).sum()
                den = (w * tw).sum().clamp_min(self.eps)
                loss_i = 0.5 * (num / den)

            else:
                num = (per_pix * w).sum()
                den = w.sum().clamp_min(self.eps)
                loss_i = 0.5 * (num / den)

            # Increase the contribution of difficult joints
            # (knees, shoulders, elbows, ankles, etc.)
            if self.joint_weighted_loss:
                jw = self.joint_weights[idx]
                loss += jw * loss_i
                weight_sum += jw
            else:
                loss += loss_i
                weight_sum += 1

        return loss / weight_sum.clamp_min(self.eps)