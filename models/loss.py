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
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        # output.reshape((batch_size, num_joints, -1)).shape is torch.Size([32, 16, 4096])
        # The type of result of applying .split(1,1) is tuple.
        # Each element shape of the tuple is torch.Size([32, 1, 4096])
        # The tuple = ( ([32, 1, 4096]),       ([32, 1, 4096]),          ([32, 1, 4096]),       ...     ([32, 1, 4096]))
        # it means        only a head        only a left shoulder     only a right shoulder            only a right ankle
        # The length of the tuple is 17.
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
    #
        for idx in range(num_joints): # the number of iteration is 17
            # heatmap_pred = heatmap's'_pred[idx] -> if idx is 1, the heatmap_pred means flattend heatmap for only a head.
            heatmap_pred = heatmaps_pred[idx].squeeze() # shape -> torch.Size([Batch, 4096])
            # heatmaps_pred[idx].shape is torch.Size([Batch, 1, 4096])
            # 4096 means flattened heatmap for a joint
            # ex) heatmaps_pred[0].squeeze().shape is torch.Size([Batch, 4096])
            # Maybe, heatmaps_pred[1] means a left shoulder, heatmap_pred[2] means a right shoulder.

            heatmap_gt = heatmaps_gt[idx].squeeze() # shape -> torch.Size([Batch, 4096])
            if self.use_target_weight: #
                loss += 0.5 * self.criterion( # self.criterion is MSE Loss
                    heatmap_pred.mul(target_weight[:, idx]),
                    # The shape of target_weight[:, idx] is torch.Size([Batch, 1])
                    # target_weight[:, idx] means whether the joint for idx detected or not.
                    heatmap_gt.mul(target_weight[:, idx])
                )
                # heatmap_pred.mul(target_weight[:, idx]) => [Batch, 4096] * [Batch , 1] => multiply (constant value = [Batch, 1]) to (each of the 4096 pixels = [Batch, 4096])

                # if heatmap_pred = heatmaps_pred[0].squeeze() and target_weight[:, 0],
                # this multiply consider visibility for a head on predicted heatmap
                # if idx = 1, consider visibility of a left shoulder.

                # < Predict a head >
                # The network may predict that a head exist when a head not exist in a picture.
                # The target_weight[:, idx] that has information of visibility for each joint prevent this case.

            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
        # Maybe not reflect loss for the each joint..??