#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil

NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tbimg(config, tblogger, imgs, step, type='train'):
    cnt = 0
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        # tblogger.add_image(f'train_batch', valid_img, step + 1, dataformats='HWC')
        for idx, img in enumerate(imgs[0]):
            cnt += 1
            if cnt % config.TRAIN.BATCH_SIZE == 0:
                tblogger.add_image(f'train_vis/img_result', imgs[0][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'train_vis/rgb_img', imgs[1][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'train_vis/heat_map', imgs[2][idx], step + 1, dataformats='CHW')

    elif type == 'vis_joint_coords':
        tblogger.add_image(f'train_vis/vis_joint_coords', imgs, step + 1, dataformats='CHW')

    elif type == 'validation':
        for idx, img in enumerate(imgs[0]):
            cnt += 1
            if cnt % config.TEST.BATCH_SIZE == 0:
                tblogger.add_image(f'val_vis/img_result', imgs[0][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'val_vis/rgb_img', imgs[1][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'val_vis/heat_map', imgs[2][idx], step + 1, dataformats='CHW')

    elif type == 'lower_perf':
        for idx, img in enumerate(imgs[0]):
            cnt += 1
            if cnt % config.TRAIN.BATCH_SIZE == 0:
                tblogger.add_image(f'lower_perf_pos/img_result', imgs[0][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'lower_perf_pos/rgb_img', imgs[1][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'lower_perf_pos/heat_map', imgs[2][idx], step + 1, dataformats='CHW')

    elif type == 'lower_performance_on_valid':
        for idx, img in enumerate(imgs[0]):
            cnt += 1
            if cnt % config.TRAIN.BATCH_SIZE == 0:
                tblogger.add_image(f'lower_performance_on_valid/img_result', imgs[0][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'lower_performance_on_valid/rgb_img', imgs[1][idx], step + 1, dataformats='CHW')
                tblogger.add_image(f'lower_performance_on_valid/heat_map', imgs[2][idx], step + 1, dataformats='CHW')

    else:
        print(f'Invalid type: "{type}". Cannot log images to TensorBoard.')
