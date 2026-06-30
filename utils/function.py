# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
import time
import os
import json

import cv2
import numpy as np
import torch
import torch.distributed as dist

from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from config.config import get_model_name, POSE_RESNET
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.events import NCOLS, load_yaml, write_tbimg
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


def train(config, train_loader, valid_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, acc_list, use_amp=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    use_bf16 = use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if (not use_amp or use_bf16) else GradScaler()
    val_iter = iter(valid_loader)
    #
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #
    one_epoch_start_time = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #
        optimizer.zero_grad()
        # switch to train mode
        model.train()
        #
        if config.USE_DDP:
            input = input.cuda(config.DDP_OPTS.GPU, non_blocking=True)
            target = target.cuda(config.DDP_OPTS.GPU, non_blocking=True)
            target_weight = target_weight.cuda(config.DDP_OPTS.GPU, non_blocking=True)

        else:
            input = input.cuda(int(config.GPUS), non_blocking=True)
            target = target.cuda(int(config.GPUS), non_blocking=True)
            target_weight = target_weight.cuda(int(config.GPUS), non_blocking=True)

        # compute output
        # --- forward: toggle autocast on/off based on use_amp ---
        if config.USE_AMP:
            # IN THIS CODE, USE ONLY FP16
            # THIS IS BECAUSE NUMPY NOT SUPPORT BF16
            with autocast(dtype=amp_dtype if not use_bf16 else torch.float16):
                output = model(input)

        else:
            output = model(input)

        # extract joint coordinates
        flatten = output.detach().cpu().reshape(config.TRAIN.BATCH_SIZE, config.MODEL.NUM_JOINTS, -1).argmax(axis=2)
        y, x = torch.div(flatten, POSE_RESNET.HEATMAP_SIZE[0], rounding_mode='floor'), flatten % POSE_RESNET.HEATMAP_SIZE[0]
        coords = torch.stack([x, y], dim=-1)  # [B, J, 2]

        # DEBUG: Visualization of extracted joint coordinates on the input image
        if config.DEBUG.VISUALIZATION:
            if isinstance(input, torch.Tensor):
                input_img = input.cpu().detach().numpy()
                for batch_idx in range(input.shape[0]):
                    img = input_img[batch_idx].transpose(1, 2, 0)
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    #
                    plt.figure()
                    plt.imshow(img)
                    for x, y in coords[batch_idx]:
                        x, y = x.float() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0], y.float() / POSE_RESNET.HEATMAP_SIZE[1] * config.MODEL.IMAGE_SIZE[1]
                        plt.scatter(x, y)
                    plt.axis('off')
                    plt.show()
        # ---------- DEBUG ----------


        # --- compute loss in FP32 for stable convergence ---
        loss = criterion(output.float(), target, target_weight)

        # --- backward + optimizer step ---
        # FP16 AMP path (requires GradScaler)
        if use_amp and not use_bf16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # BF16 AMP or plain FP32 path
        else:
            loss.backward()
            optimizer.step()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy(),
                                         thr = config.ACC_THR)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                  'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            #
            if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train/loss_val', losses.val, global_steps)
                writer.add_scalar('train/acc_val', acc.val, global_steps)
                writer.add_scalar('train/loss_avg', losses.avg, global_steps)
                writer.add_scalar('train/acc_avg', acc.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
                #
                # visualization exatracted joint coords on original image
                if isinstance(input, torch.Tensor):
                    input_img = input.cpu().detach().numpy()
                    img = input_img[0].transpose(1, 2, 0)
                    # De-normalize: normalized -> [0, 1]
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    #
                    for x, y in coords[0]:
                        x = int(x.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                        y = int(y.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                        y1 = max(0, y - 5)
                        y2 = min(img.shape[0], y + 5)
                        x1 = max(0, x - 5)
                        x2 = min(img.shape[1], x + 5)
                        img[y1:y2, x1:x2] = np.array([1.0, 0.0, 0.0], dtype=np.uint8)
                    img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    write_tbimg(config, writer_dict['writer'], imgs=img_chw, step=i, type='vis_joint_coords')

                result, ori, hm = plot_train_batch(config, input, output)
                train_result = [result, ori, hm]
                write_tbimg(config, writer_dict['writer'], imgs=train_result, step=i, type='train')

                if acc.val < 0.9:
                    result, ori, hm = plot_train_batch(config, input, output)
                    train_result = [result, ori, hm]
                    write_tbimg(config, writer_dict['writer'], imgs=train_result, step=i, type='lower_perf')

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

    # Training time measurement for one epoch
    one_epoch_end_time = time.time()
    train_one_epoch_time = one_epoch_end_time - one_epoch_start_time
    print(f"[Epoch {epoch}] Train Time: {train_one_epoch_time:.2f} sec")

    # Validation time measurement
    valid_start_time = time.time()
    acc_val, val_iter = validate(config=config, val_loader=valid_loader, model=model,
             criterion=criterion, epoch=epoch, output_dir=output_dir, tb_log_dir=tb_log_dir,
             val_iter='', writer_dict=writer_dict, is_training=True)
    valid_end_time = time.time()
    epoch_time = valid_end_time - valid_start_time
    print(f"[Epoch {epoch}] Valid Time: {epoch_time:.2f} sec")
    #
    return acc_val

def validate(config, val_loader, model, criterion, epoch, output_dir,
             tb_log_dir,  val_iter, writer_dict=None, is_training=False, ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    #
    FULL_EVAL = config.TEST.FULL_EVAL

    # switch to evaluate mode
    model.eval()

    # To check various test images in the tensorboard, we make randint value.
    max_val_images = random.randint(2, 10) if is_training else len(val_loader)

    # To Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Collect lower Perf samples on validation
    lower_perf_samples = {}
    lower_perf_samples['json_path'] = []
    lower_perf_samples['avg_acc'] = []
    lower_perf_samples['cnt'] = []
    #
    with torch.no_grad():
        end = time.time()
        if not FULL_EVAL:
            for i in range(max_val_images):
                try:
                    input, target, target_weight, meta = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    input, target, target_weight, meta = next(val_iter)

                # measure data loading time
                data_time.update(time.time() - end)
                #
                if config.USE_DDP:
                    input = input.cuda(config.DDP_OPTS.GPU, non_blocking=True)
                    target = target.cuda(config.DDP_OPTS.GPU, non_blocking=True)
                    target_weight = target_weight.cuda(config.DDP_OPTS.GPU, non_blocking=True)

                else:
                    input = input.cuda(int(config.GPUS), non_blocking=True)
                    target = target.cuda(int(config.GPUS), non_blocking=True)
                    target_weight = target_weight.cuda(int(config.GPUS), non_blocking=True)

                # compute output
                output = model(input)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)

                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy(), thr = config.ACC_THR)

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % config.PRINT_FREQ == 0:
                if (i + 1) % max_val_images == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                          'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                        0, i, len(val_loader), batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)

                    if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
                        writer = writer_dict['writer']
                        global_steps = writer_dict['train_global_steps']
                        writer.add_scalar('val/loss', losses.val, global_steps)
                        writer.add_scalar('val/acc', acc.val, global_steps)
                        writer_dict['train_global_steps'] = global_steps + 1

                        result, ori, hm = plot_train_batch(config, input, output)
                        valid_result = [result, ori, hm]
                        write_tbimg(config, writer_dict['writer'], imgs=valid_result, step=i, type='validation')

                        prefix = '{}_{}'.format(os.path.join(output_dir, 'validation'), i)
                        save_debug_images(config, input, meta, target, pred * 4, output,
                                          prefix)

        else:
            for i, (input, target, target_weight, meta, json_path) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                #
                if config.USE_DDP:
                    input = input.cuda(config.DDP_OPTS.GPU, non_blocking=True)
                    target = target.cuda(config.DDP_OPTS.GPU, non_blocking=True)
                    target_weight = target_weight.cuda(config.DDP_OPTS.GPU, non_blocking=True)

                else:
                    input = input.cuda(int(config.GPUS), non_blocking=True)
                    target = target.cuda(int(config.GPUS), non_blocking=True)
                    target_weight = target_weight.cuda(int(config.GPUS), non_blocking=True)

                # compute output
                output = model(input)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)

                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy(), thr=config.ACC_THR)

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if acc.val < 0.8:
                    # [DEBUG] Collect Lower Perf Samples on Valid
                    # lower_perf_samples['json_path'].append(json_path)
                    # lower_perf_samples['avg_acc'].append(avg_acc)
                    # lower_perf_samples['cnt'].append(cnt)
                    #
                    # Drawing Heatmap on Original Image
                    result, ori, hm = plot_train_batch(config, input, output)

                    # Extract joint coordinates
                    flatten = output.detach().cpu().reshape(config.TEST.BATCH_SIZE, config.MODEL.NUM_JOINTS, -1).argmax(axis=2)
                    y, x = torch.div(flatten, POSE_RESNET.HEATMAP_SIZE[0], rounding_mode='floor'), flatten % POSE_RESNET.HEATMAP_SIZE[0]
                    coords = torch.stack([x, y], dim=-1)  # [B, J, 2]

                    input_img = input.cpu().detach().numpy()
                    img = input_img[0].transpose(1, 2, 0)

                    # De-normalize: normalized -> [0, 1]
                    img = img * std + mean
                    img = np.clip(img, 0, 1)

                    for x, y in coords[0]:
                        x = int(x.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                        y = int(y.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                        y1 = max(0, y - 5)
                        y2 = min(img.shape[0], y + 5)
                        x1 = max(0, x - 5)
                        x2 = min(img.shape[1], x + 5)
                        img[y1:y2, x1:x2] = np.array([1.0, 0.0, 0.0], dtype=np.uint8)
                    img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    train_result = [result, ori, hm, img_chw]

                    write_tbimg(config=config, tblogger=writer_dict['writer'], imgs=train_result, step=i,
                                type='lower_performance_on_valid')

                if i % 10 == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                          'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                        0, i, len(val_loader), batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)

                    if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
                        writer = writer_dict['writer']
                        global_steps = writer_dict['valid_global_steps']
                        writer.add_scalar('val/loss_val', losses.val, global_steps)
                        writer.add_scalar('val/acc_val', acc.val, global_steps)
                        writer.add_scalar('val/loss_avg', losses.avg, global_steps)
                        writer.add_scalar('val/acc_avg', acc.avg, global_steps)
                        writer_dict['valid_global_steps'] = global_steps + 1

                        result, ori, hm = plot_train_batch(config, input, output)
                        valid_result = [result, ori, hm]
                        write_tbimg(config, writer_dict['writer'], imgs=valid_result, step=i, type='validation')

                        prefix = '{}_{}'.format(os.path.join(output_dir, 'validation'), i)
                        save_debug_images(config, input, meta, target, pred * 4, output,
                                          prefix)

    # [DEBUG] Collect Lower Perf Samples on Valid
    # with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/[DEBUG_] Lower_Perf_Samples_on_Valid.json', 'w', encoding='utf-8') as f:
    #     json.dump(lower_perf_samples, f, ensure_ascii=False, indent=4)

    if not FULL_EVAL:
        return acc, val_iter
    else:
        return acc, None



def hard_exercise_finetune(config, train_loader, valid_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, acc_list, use_amp=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    #
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #
    one_epoch_start_time = time.time()
    for i, (data, json_file, workout_name, conditions) in enumerate(train_loader):
        optimizer.zero_grad()
        # switch to train mode
        model.train()
        # Generate Image Path from meta
        label2image = path[0].replace('label', 'image').split('/')
        base_path = os.path.join('/'.join(label2image[:7]), (label2image[8]))
        for view_idx in view_keys:
            a_video = {}
            for frame_idx in range(len(data['frames'])):
                a_frame_pts = {k: [] for k in config.DEMO.JOINTS_NAME}
                img_path = base_path + '/' + data['frames'][frame_idx][view_idx]['img_key'][0]

                # Debug
                if img_path == '/storage/jysuh/fitness/fitness/validation/image/babel_01/Day07_200929_F/5/A/011-1-1-01-Z21_A/011-1-1-01-Z21_A-0000002.jpg':
                    pass

                # Load an Image
                data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if data_numpy is None:
                    print(f"Failed to read image: {img_path}")
                    raise ValueError("Failed to read image")

                # Image Crop and Resize
                CROP = 1080
                H, W = data_numpy.shape[:2]  # H=1080, W=1920
                cx, cy = W / 2.0, H / 2.0  # (960, 540)

                # Crop
                x0, x1 = int(round(cx - CROP / 2)), int(round(cx + CROP / 2))
                y0, y1 = int(round(cy - CROP / 2)), int(round(cy + CROP / 2))
                cropped = data_numpy[y0:y1, x0:x1]

                # Resize
                out_w, out_h = config.MODEL.IMAGE_SIZE
                data_numpy = cv2.resize(cropped, (out_w, out_h))

                # BGR2RGB
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

                # Forward pass
                data_tensor = to_tensor(data_numpy)  # [H,W,C] uint8 0~255 -> [C,H,W] float 0~1
                input_data = normalize(data_tensor)  # Normalize
                input_data = input_data.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                input_data = input_data.to(device)
                output = model(input_data)

                # Extract Joints Points
                hm_flatten = output.detach().cpu(). \
                    reshape(config.DEMO.BATCH_SIZE, config.MODEL.NUM_JOINTS, -1).argmax(axis=2)
                y, x = torch.div(hm_flatten, POSE_RESNET.HEATMAP_SIZE[0], rounding_mode='floor'), \
                    hm_flatten % POSE_RESNET.HEATMAP_SIZE[0]
                coords = torch.stack([x, y], dim=-1)  # [B, J, 2]

                # DEBUG: Visualization
                if config.DEBUG.VISUALIZATION:
                    import matplotlib.pyplot as plt
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    #
                    if isinstance(input_data, torch.Tensor):
                        input_img = input_data.cpu().detach().numpy()
                        for batch_idx in range(input_data.shape[0]):
                            img = input_img[batch_idx].transpose(1, 2, 0)
                            img = img * std + mean
                            img = np.clip(img, 0, 1)
                            #
                            plt.figure()
                            plt.imshow(img)
                            for x, y in coords[batch_idx]:
                                x, y = x.float() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0], y.float() / \
                                       POSE_RESNET.HEATMAP_SIZE[1] * config.MODEL.IMAGE_SIZE[1]
                                plt.scatter(x, y)
                            plt.axis('off')
                            plt.show()

                # --- compute loss in FP32 for stable convergence ---
                loss = criterion(output.float(), target, target_weight)

                # --- backward + optimizer step ---
                loss.backward()
                optimizer.step()
                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy(),
                                                 thr=config.ACC_THR)
                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                          'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)
                    #
                    if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
                        writer = writer_dict['writer']
                        global_steps = writer_dict['train_global_steps']
                        writer.add_scalar('train/loss_val', losses.val, global_steps)
                        writer.add_scalar('train/acc_val', acc.val, global_steps)
                        writer.add_scalar('train/loss_avg', losses.avg, global_steps)
                        writer.add_scalar('train/acc_avg', acc.avg, global_steps)
                        writer_dict['train_global_steps'] = global_steps + 1
                        #
                        # visualization exatracted joint coords on original image
                        if isinstance(input, torch.Tensor):
                            input_img = input.cpu().detach().numpy()
                            img = input_img[0].transpose(1, 2, 0)
                            # De-normalize: normalized -> [0, 1]
                            img = img * std + mean
                            img = np.clip(img, 0, 1)
                            #
                            for x, y in coords[0]:
                                x = int(x.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                                y = int(y.item() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0])
                                y1 = max(0, y - 5)
                                y2 = min(img.shape[0], y + 5)
                                x1 = max(0, x - 5)
                                x2 = min(img.shape[1], x + 5)
                                img[y1:y2, x1:x2] = np.array([1.0, 0.0, 0.0], dtype=np.uint8)
                            img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                            write_tbimg(config, writer_dict['writer'], imgs=img_chw, step=i, type='vis_joint_coords')

                        result, ori, hm = plot_train_batch(config, input, output)
                        train_result = [result, ori, hm]
                        write_tbimg(config, writer_dict['writer'], imgs=train_result, step=i, type='train')

                        if acc.val < 0.9:
                            result, ori, hm = plot_train_batch(config, input, output)
                            train_result = [result, ori, hm]
                            write_tbimg(config, writer_dict['writer'], imgs=train_result, step=i, type='lower_perf')

                        prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                        save_debug_images(config, input, meta, target, pred * 4, output,
                                          prefix)

            # Training time measurement for one epoch
            one_epoch_end_time = time.time()
            train_one_epoch_time = one_epoch_end_time - one_epoch_start_time
            print(f"[Epoch {epoch}] Train Time: {train_one_epoch_time:.2f} sec")

            # Validation time measurement
            valid_start_time = time.time()
            acc_val, val_iter = validate(config=config, val_loader=valid_loader, model=model,
                                         criterion=criterion, epoch=epoch, output_dir=output_dir, tb_log_dir=tb_log_dir,
                                         val_iter='', writer_dict=writer_dict, is_training=True)
            valid_end_time = time.time()
            epoch_time = valid_end_time - valid_start_time
            print(f"[Epoch {epoch}] Valid Time: {epoch_time:.2f} sec")
            #
            return acc_val


def plot_train_batch(config, input, output, gamma=0.5, max_subplots=16):
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
        # img = input[0].transpose(2, 0).transpose(1, 0)
        # img = img.type(torch.uint8)
        # plt.imshow(img)

        # De-normalize input: normalized -> [0, 1] -> [0, 255]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        input_denorm = input * std + mean
        input_denorm = np.clip(input_denorm, 0, 1)
        input_uint8 = (input_denorm * 255).astype(np.uint8)  # [B, C, H, W]

    if isinstance(output, torch.Tensor):
        upsample = torch.nn.Upsample(size=(config.MODEL.IMAGE_SIZE[0],config.MODEL.IMAGE_SIZE[1]), mode='nearest')
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

    result = (deepcopy(input_uint8)*(1-gamma) + np.stack(temp, axis=0) * gamma).astype('uint8')
    return result, input_uint8.astype(np.uint8), np.stack(temp, axis=0)
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