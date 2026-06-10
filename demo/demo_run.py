# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import shutil
import json
import random
import numpy as np
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import yaml
import torch.distributed as dist

from utils.function import AverageMeter
from tensorboardX import SummaryWriter
from demo.demo_config import config
from models.loss import JointsMSELoss
from utils.function import train
from utils.function import validate
from utils.function import get_sequences
from utils.utils import get_optimizer

from utils.utils import create_logger
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


def cleanup():
    """ Destroy process group """
    dist.destroy_process_group()

def init_distributed_training(rank):
    # 1. setting for distributed training
    config.DDP_OPTS.RANK = rank
    config.DDP_OPTS.GPU = config.DDP_OPTS.RANK % torch.cuda.device_count()
    config.DDP_OPTS.LOCAL_GPU_ID = int(config.DDP_OPTS.GPU_IDS[config.DDP_OPTS.RANK])
    torch.cuda.set_device(config.DDP_OPTS.LOCAL_GPU_ID)

    if config.DDP_OPTS.RANK is not None:
        print("Use GPU: {} for training".format(config.DDP_OPTS.LOCAL_GPU_ID))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://202.31.136.169:' + str(config.DDP_OPTS.PORT_NUM),
                                         world_size=config.DDP_OPTS.NGPUS_PER_NODE,
                                         rank=config.DDP_OPTS.RANK)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(config.DDP_OPTS.RANK == 0)
    print('CONFIG.DDP_OPTS :', config.DDP_OPTS)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def gen_config(config_save_path):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_save_path, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def set_seed(seed: int = 42):
    # Python random
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # CUDA ?? determinism ??
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (??) ?? deterministic (?? ??? ? ??)
    torch.use_deterministic_algorithms(True)

    # (??) hashing ??
    os.environ["PYTHONHASHSEED"] = str(seed)

def main(rank):
    set_seed(42)

    # Generate configuration
    gen_config(config_save_path='/storage/jysuh/Simple_Baseline_For_HPE_Workout/config/workout.yaml')

    # Set Device
    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    # Set Logger
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg=config, cfg_name=config.CONFIG_FILE_PATH.split('/')[2], phase=config.TASK)

    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

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

    # =*=*=*=* Set Model =*=*=*=*
    from models.pose_resnet import get_pose_net
    model = get_pose_net(config).to(device)

    #
    # To make calculation graph in the tensorboard, dump_input is forwarding.
    # dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
    #                          3,
    #                          config.MODEL.IMAGE_SIZE[1],
    #                          config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model.cpu(), (dump_input, ), verbose=False)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    from demo.JointsDataset_demo import JointsDataset

    dataset = JointsDataset(cfg=config,
                         root=config.DEMO.JSON_PATH,
                         transform=transforms.Compose([transforms.ToTensor(), normalize]))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DEMO.BS,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # Set Average Meter (Metric)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, meta in enumerate(dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

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
            for i, (input, target, target_weight, meta) in enumerate(val_loader):
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

                if i % config.PRINT_FREQ == 0:
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
                        writer.add_scalar('val/loss', losses.val, global_steps)
                        writer.add_scalar('val/acc', acc.val, global_steps)
                        writer_dict['valid_global_steps'] = global_steps + 1

                        result, ori, hm = plot_train_batch(config, input, output)
                        valid_result = [result, ori, hm]
                        write_tbimg(config, writer_dict['writer'], imgs=valid_result, step=i, type='validation')

                        prefix = '{}_{}'.format(os.path.join(output_dir, 'validation'), i)
                        save_debug_images(config, input, meta, target, pred * 4, output,
                                          prefix)

                ####
            # if i % config.PRINT_FREQ == 0:
            #     msg = 'Test: [{0}/{1}]\t' \
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
            #           'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time,
            #               loss=losses, acc=acc)
            #     logger.info(msg)
            #
            #     prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
            #
            #     result, ori, hm = plot_train_batch(config, input, output)
            #     valid_result = [result, ori, hm]
            #     epoch = i if is_training else epoch
            #
            #     write_tbimg(writer_dict['writer'], imgs=valid_result, step=epoch, type='val')
            #     save_debug_images(config, input, meta, target, pred*4, output,
            #                       prefix)
            #
            # if writer_dict:
            #     writer = writer_dict['writer']
            #     global_steps = writer_dict['valid_global_steps']
            #     # writer.add_scalar('valid/loss', losses.avg, global_steps)
            #     # writer.add_scalar('valid/acc', acc.avg, global_steps)
            #     writer.add_scalar('valid/loss', losses.avg, global_step=epoch)
            #     writer.add_scalar('valid/acc', acc.avg, global_step=epoch)

            # break

if __name__ == '__main__':
    from setproctitle import *
    setproctitle('HPE : sigma=0.1 / lr=0.001*0.7 / thr = 0.2')
    # setproctitle('Generate Sequences information for transformer')

    main(rank=None)