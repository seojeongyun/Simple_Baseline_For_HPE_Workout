# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
    # The shape of batch_heatmaps is (32, 17, 64, 64)
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    # (#, 17, 4096)
    idx = np.argmax(heatmaps_reshaped, 2)
    # (32, 17)
    # Extracting the largest index on heatmap using argmax means finding that where is a joint i need
    # The range of idx is 0 ~ 4095.

    maxvals = np.amax(heatmaps_reshaped, 2)
    # (32, 17)
    # For the largest index, it extract max value on heatmap

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) # (32, 17, 2). 2(index[1]) -> copy of same value for index [0]
    # preds have idx. -> this mean where is a joint on heatmap?

    preds[:, :, 0] = (preds[:, :, 0]) % width # column
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) # row
    # np.floor(1.999) = 1.0
    # return to the original (before flatten) index

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    # np.greater(maxvals, 0.0) ----- if maxvals larger than 0.0, the result is True, if not, false

    pred_mask = pred_mask.astype(np.float32)
    # True and False is replaced 1.0 and 0.0

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p] # [Batch, 17, 64, 64] -> batch_heatmaps[n][p] means a heatmap for [batch = n][what joint? = p]
                px = int(math.floor(coords[n][p][0] + 0.5)) # coords[batch][number of joints][x]
                py = int(math.floor(coords[n][p][1] + 0.5)) # coords[batch][number of joints][y]
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]]) # Calculating difference of activation value for x axis and y axis on heatmap
                    coords[n][p] += np.sign(diff) * .25
                    # np.sign -> if a value is positive, the result is 1
                    # if a value is negative, the result is -1
                    # if a value is zero, the result is zero
    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]): # coords.shape[0] = batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals