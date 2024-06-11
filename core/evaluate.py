# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0])) # [17, 32]
    for n in range(preds.shape[0]): # 32
        for c in range(preds.shape[1]): # 17
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                # dists[c,n] -> the distance between two vector (preds and targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1) # 32 -> True or False
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
        # np.less(dists[dist_cal] , thr) -> compare value of dists[dist_cal] and thr and return True or False
    else:
        return -1
    # 1. Check the number of True (if the value is not -1 == target[Batch, joint, x] > 1 and target[Batch, joint, y] > 1, It is True)
    # 2. Count the number of True
    # 3. When there is at least one True, compare value of dists[dist_cal] and thr and return True or False, counting the number of True and * 1.0 / num_dist_cal
    # Finally, the number of value more 0.5 / the number of dists which not have -1 = percentage

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    # output.shape = [Batch, 17, 64, 64]
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output) # the shape of pred is [Batch, 17, 2]
        target, _ = get_max_preds(target)
        h = output.shape[2] # 64
        w = output.shape[3] # 64
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        # [Batch, 2] all the value is 6.4
    dists = calc_dists(pred, target, norm) # calculate distance between pred and target

    acc = np.zeros((len(idx) + 1)) # 17 + 1
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)): # 17
        acc[i + 1] = dist_acc(dists[idx[i]]) # idx[i]  = batch number
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred