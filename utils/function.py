# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import cv2
import numpy as np
import torch

from config.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.events import NCOLS, load_yaml, write_tbimg

logger = logging.getLogger(__name__)


def train(config, train_loader, valid_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input.to('cuda'))
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train/loss', losses.val, global_steps)
            writer.add_scalar('train/acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            result, ori, hm = plot_train_batch(config, input, output)
            train_result = [result, ori, hm]
            write_tbimg(writer_dict['writer'], imgs=train_result, step=epoch, type='train')

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

        if i % (config.PRINT_FREQ * 3) == 0:
            validate(config=config, val_loader=valid_loader, model=model,
                     criterion=criterion, epoch=epoch, output_dir=output_dir, tb_log_dir=tb_log_dir,
                     writer_dict=writer_dict)



def validate(config, val_loader, model, criterion, epoch, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # c = meta['center'].numpy()
            # s = meta['scale'].numpy()
            # score = meta['score'].numpy()
            #
            # preds, maxvals = get_final_preds(
            #     config, output.clone().cpu().numpy(), c, s)

            # all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            # all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # # double check this all_boxes parts
            # all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            # all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            # all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            # all_boxes[idx:idx + num_images, 5] = score


            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)

                result, ori, hm = plot_train_batch(config, input, output)
                valid_result = [result, ori, hm]
                write_tbimg(writer_dict['writer'], imgs=valid_result, step=epoch, type='val')
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid/loss', losses.avg, global_steps)
            writer.add_scalar('valid/acc', acc.avg, global_steps)



def plot_train_batch(config, input, output, gamma=0.5, max_subplots=16):
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
        # img = input[0].transpose(2, 0).transpose(1, 0)
        # img = img.type(torch.uint8)
        # plt.imshow(img)
    if isinstance(output, torch.Tensor):
        upsample = torch.nn.Upsample(size=(config.MODEL.IMAGE_SIZE,config.MODEL.IMAGE_SIZE), mode='nearest')
        output = upsample(output)
        output = output.cpu().detach().numpy()

    all = np.zeros([output.shape[0], output.shape[2], output.shape[3]])
    for i in range(output.shape[1]):
        all += output[:,i,:,:]

    from copy import deepcopy

    temp = []
    for i in range(output.shape[0]):
        max = all[i].max()
        all[i] /= max
        all[i] *= 255
        temp.append(np.expand_dims(deepcopy(all[i]).astype(np.uint8), axis=0))

    # zeros = np.zeros(all.shape)
    # all = np.concatenate((all,zeros,zeros), axis=1)

    for i in range(output.shape[0]):
        temp[i] = cv2.applyColorMap(temp[i].transpose(1, 2, 0), cv2.COLORMAP_JET).transpose(2, 0, 1)
        temp[i] = np.stack([temp[i][2, :, :], temp[i][1, :, :], temp[i][0, :, :]], axis=0)
    # result = input + gamma * np.stack(temp, axis=0)

    result = (deepcopy(input)*(1-gamma) + np.stack(temp, axis=0) * gamma).astype('uint8')
    return result, input.astype(np.uint8), np.stack(temp, axis=0)
        # rgb img + htmap / rgb img / htmap


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0