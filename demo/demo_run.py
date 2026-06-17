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
import cv2

from utils.function import AverageMeter
from tensorboardX import SummaryWriter
from demo.demo_config import config, POSE_RESNET
from models.loss import JointsMSELoss
from utils.function import train
from utils.function import validate
from utils.function import get_sequences
from utils.utils import get_optimizer

from utils.utils import create_logger
from easydict import EasyDict as edict


def cleanup():
    """ Destroy process group """
    dist.destroy_process_group()

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

    # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)

def main(rank):
    set_seed(42)

    # Generate configuration
    gen_config(config_save_path='/storage/jysuh/Simple_Baseline_For_HPE_Workout/demo/workout.yaml')

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
    shutil.copy2(
        os.path.join('/storage/jysuh/Simple_Baseline_For_HPE_Workout', 'models', config.MODEL.NAME + '.py'),    # pose_resnet.py copy to final_output_dir
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
        shuffle=True,
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

    # View idx
    view_keys = ['view1', 'view2', 'view3', 'view4', 'view5']
    with torch.no_grad():
        for step, (data, path) in enumerate(dataloader):
            # Generate Image Path from meta
            label2image = path[0].replace('label', 'image').split('/')
            base_path = os.path.join('/'.join(label2image[:7]),(label2image[8]))
            for view_idx in view_keys:
                for frame_idx in range(len(data['frames'])):
                    img_path = base_path + '/' + data['frames'][frame_idx][view_idx]['img_key'][0]
                    print(img_path)
                    if img_path == '/storage/jysuh/fitness/fitness/validation/image/babel_01/Day07_200929_F/5/A/011-1-1-01-Z21_A/011-1-1-01-Z21_A-0000002.jpg':
                        pass
                    # Load an Image
                    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if data_numpy is None:
                        print(f"Failed to read image: {img_path}")
                        continue

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
                    output = model(data_numpy)

                    # Extract Joints Points
                    hm_flatten = output.detach().cpu().\
                        reshape(config.TRAIN.BATCH_SIZE, config.MODEL.NUM_JOINTS,-1).argmax(axis=2)
                    y, x = torch.div(hm_flatten, POSE_RESNET.HEATMAP_SIZE[0], rounding_mode='floor'), \
                        hm_flatten % POSE_RESNET.HEATMAP_SIZE[0]
                    coords = torch.stack([x, y], dim=-1)  # [B, J, 2]

                    for x, y in coords[0]:
                        x, y = x.float() / POSE_RESNET.HEATMAP_SIZE[0] * config.MODEL.IMAGE_SIZE[0], y.float() / \
                               POSE_RESNET.HEATMAP_SIZE[1] * config.MODEL.IMAGE_SIZE[1]

                    # Pts Save

                #
                # # measure accuracy and record loss
                # losses.update(loss.item(), num_images)
                # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                #                                  target.cpu().numpy(), thr=config.ACC_THR)
                #
                # acc.update(avg_acc, cnt)
                #
                # # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()
                #
                # # if i % config.PRINT_FREQ == 0:
                # if (i + 1) % max_val_images == 0:
                #     msg = 'Epoch: [{0}][{1}/{2}]\t' \
                #           'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                #           'Speed {speed:.1f} samples/s\t' \
                #           'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                #           'Loss {loss.val:.7f} ({loss.avg:.7f})\t' \
                #           'Accuracy {acc.val:.5f} ({acc.avg:.5f})'.format(
                #         0, i, len(val_loader), batch_time=batch_time,
                #         speed=input.size(0) / batch_time.val,
                #         data_time=data_time, loss=losses, acc=acc)
                #     logger.info(msg)
                #
                #     if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
                #         writer = writer_dict['writer']
                #         global_steps = writer_dict['train_global_steps']
                #         writer.add_scalar('val/loss', losses.val, global_steps)
                #         writer.add_scalar('val/acc', acc.val, global_steps)
                #         writer_dict['train_global_steps'] = global_steps + 1
                #
                #         result, ori, hm = plot_train_batch(config, input, output)
                #         valid_result = [result, ori, hm]
                #         write_tbimg(config, writer_dict['writer'], imgs=valid_result, step=i, type='validation')
                #
                #         prefix = '{}_{}'.format(os.path.join(output_dir, 'validation'), i)
                #         save_debug_images(config, input, meta, target, pred * 4, output,
                #                           prefix)

if __name__ == '__main__':
    from setproctitle import *
    setproctitle('HPE : sigma=0.1 / lr=0.001*0.7 / thr = 0.2')
    # setproctitle('Generate Sequences information for transformer')

    main(rank=None)