# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import json

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import tensorboard
import yaml
import _init_paths
import torch.distributed as dist

from tensorboardX import SummaryWriter
from config.config import config
from config.config import update_config
from config.config import update_dir
from config.config import get_model_name
from models.loss import JointsMSELoss
from utils.function import train
from utils.function import validate
from utils.function import get_sequences
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from easydict import EasyDict as edict
from cmd_in import get_args_parser
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import dataset
import models


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

def main():
    gen_config('/storage/jysuh/Simple_Baseline_For_HPE_Workout/config/workout.yaml')

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg=config, cfg_name=config.CONFIG_FILE_PATH.split('/')[2], phase=config.TASK)

    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    from models.pose_resnet import get_pose_net
    model = get_pose_net(config, is_train=True)
    # Check the is_train -> whether your purpose is train or validation

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, 'models', config.MODEL.NAME + '.py'),    # pose_resnet.py copy to final_output_dir
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)
    # To make calculation graph in the tensorboard, dump_input is forwarding.

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda() # To use multi gpus / gpus = [0, 1]

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    from dataset.JointsDataset import JointsDataset
    if config.TASK == 'train':
        train_dataset = JointsDataset(cfg=config,
                             root=config.DATASET.ROOT,
                             task=config.TASK,
                             transform=transforms.Compose([transforms.ToTensor(), normalize]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True
        )

    valid_dataset = JointsDataset(cfg=config,
                         root=config.DATASET.ROOT_VALID_LABEL,
                         task='validation' if config.TASK == 'train' else config.TASK,
                         transform=transforms.Compose([transforms.ToTensor(), normalize]))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    val_acc = 0.0
    best_model = False
    if config.TASK == 'train':
        for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
            acc_list = []
            # train for one epoch
            train(config, train_loader, valid_loader, model, criterion, optimizer, epoch,
                                 final_output_dir, tb_log_dir, writer_dict, acc_list, use_amp=config.TRAIN.USE_AMP)

            # if perf_indicator > best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'model': get_model_name(config),
            #     'state_dict': model.state_dict(),
            #     'perf': perf_indicator,
            #     'optimizer': optimizer.state_dict(),
            # }, best_model, final_output_dir)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

            lr_scheduler.step()
        #
        final_model_state_file = os.path.join('/storage/jysuh/fitness_weights/',
                                              'final_state.pth.tar')

        logger.info('saving final model state to {}'.format(
            final_model_state_file))

        torch.save(model.module.state_dict(), final_model_state_file)

        writer_dict['writer'].close()

    elif config.TASK == 'validation':
        epoch = 0
        acc = validate(config=config, val_loader=valid_loader, model=model,
                     criterion=criterion, epoch=epoch, output_dir=final_output_dir, tb_log_dir=tb_log_dir,
                     writer_dict=writer_dict, is_training=False)

    elif config.TASK == 'get_sequences_for_tf':
        sequences_data_to_tf = get_sequences(config, valid_loader, model)
        with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/sequences_data_to_tf.json','w') as f:
            json.dump(sequences_data_to_tf, f)

    else:
        raise ValueError("{} is wrong task.".format(config.TASK))

if __name__ == '__main__':
    from setproctitle import *
    setproctitle('Simple_Baseline : Workout [1024, 1024]')
    # setproctitle('Generate Sequences information for transformer')
    main()