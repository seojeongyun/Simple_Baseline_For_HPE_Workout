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

from tensorboardX import SummaryWriter
from config.config import config
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

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
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
    gen_config('/storage/jysuh/Simple_Baseline_For_HPE_Workout/config/workout.yaml')
    #
    if config.USE_DDP:
        init_distributed_training(rank)
        local_gpu_id = config.DDP_OPTS.GPU
    else:
        device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")
    #
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
    #
    from models.pose_resnet import get_pose_net
    model = get_pose_net(config)
    #
    if config.USE_DDP:
        # --- Distributed Data Parallel ---
        model = model.cuda(local_gpu_id)
        model = DistributedDataParallel(module=model, device_ids=[local_gpu_id], output_device=local_gpu_id)
    else:
        # --- Single GPU ---
        model = model.to(device)

    #
    # To make calculation graph in the tensorboard, dump_input is forwarding.
    # dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
    #                          3,
    #                          config.MODEL.IMAGE_SIZE[1],
    #                          config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model.cpu(), (dump_input, ), verbose=False)


    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT)
    criterion = criterion.to(local_gpu_id) if config.USE_DDP else criterion.to(device)

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
        if config.USE_DDP:
            train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
            batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, config.TRAIN.BATCH_SIZE, drop_last=True)
            train_loader = DataLoader(train_dataset,
                                      batch_sampler=batch_sampler_train,
                                      num_workers=config.DDP_OPTS.NUM_WORKERS,
                                      pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=config.TRAIN.SHUFFLE,
                num_workers=config.WORKERS,
                pin_memory=True
            )
    if config.TASK != 'append_pred_to_label_json_file':
        valid_dataset = JointsDataset(cfg=config,
                             root=config.DATASET.ROOT_VALID_LABEL,
                             task='validation' if config.TASK == 'train' else config.TASK,
                             transform=transforms.Compose([transforms.ToTensor(), normalize]))
        if config.USE_DDP:
            valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True)
            batch_sampler_valid = torch.utils.data.BatchSampler(valid_sampler, config.TEST.BATCH_SIZE, drop_last=True)
            valid_loader = DataLoader(valid_dataset,
                                      batch_sampler=batch_sampler_valid,
                                      num_workers=config.DDP_OPTS.NUM_WORKERS,
                                      pin_memory=True)
        else:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=config.TEST.BATCH_SIZE,
                shuffle=config.TEST.SHUFFLE,
                num_workers=config.WORKERS,
                pin_memory=True
            )

    if config.TASK == 'append_pred_to_label_json_file':
        from demo.JointsDataset_demo import JointsDataset
        dataset = JointsDataset(cfg=config,
                             root=config.DATASET.ROOT_VALID_LABEL,
                             dataset_type='validation' if config.TASK == 'train' else config.TASK,
                             transform=transforms.Compose([transforms.ToTensor(), normalize]))
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.TEST.BATCH_SIZE,
                shuffle=config.TEST.SHUFFLE,
                num_workers=config.WORKERS,
                pin_memory=True
            )
    best_perf = 0.0
    if config.TASK == 'train':
        for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
            #
            acc_list = []
            #
            criterion.on_new_epoch(device=config.DDP_OPTS.GPU if config.USE_DDP else device)

            start_time = time.time()
            # train for one epoch
            val_acc = train(config, train_loader, valid_loader, model, criterion, optimizer, epoch,
                                 final_output_dir, tb_log_dir, writer_dict, acc_list, use_amp=config.USE_AMP)

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"[Epoch {epoch}] Train Time: {epoch_time:.2f} sec")

            if val_acc.avg > best_perf:
                best_perf = val_acc.avg
                best_model = True
            else:
                best_model = False

            if best_model:
                if not config.MODEL.PRETRAINED:
                    torch.save(model.state_dict(),os.path.join(final_output_dir, 'from_scratch_512_128.pth.tar'))
                if config.MODEL.PRETRAINED:
                    torch.save(model.state_dict(),os.path.join(final_output_dir, 'finetune.pth_input_size_512_512.tar'))

                logger.info('=> saving checkpoint to {}'.format(final_output_dir))

            lr_scheduler.step()
        #
        final_model_state_file = os.path.join('/storage/jysuh/fitness_weights/',
                                              'final_state.pth.tar')

        logger.info('saving final model state to {}'.format(
            final_model_state_file))

        if not config.USE_DDP or (dist.is_initialized() and dist.get_rank() == 0):
            torch.save(model.module.state_dict(), final_model_state_file)

        writer_dict['writer'].close()

    elif config.TASK == 'validation':
        epoch = 0
        acc, _ = validate(config=config, val_loader=valid_loader, model=model,
                     criterion=criterion, epoch=epoch, output_dir=final_output_dir, tb_log_dir=tb_log_dir,
                     writer_dict=writer_dict, is_training=False)

    elif config.TASK == 'get_sequences_for_tf':
        sequences_data_to_tf = get_sequences(config, valid_loader, model)
        with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/json_files/sequences_data_to_tf.json','w') as f:
            json.dump(sequences_data_to_tf, f)

    elif config.TASK == 'append_pred_to_label_json_file':
        append_pred_to_label_json_file(config, dataloader, model)

    else:
        raise ValueError("{} is wrong task.".format(config.TASK))

if __name__ == '__main__':
    from setproctitle import *
    setproctitle('HPE : sigma=0.1 / lr=0.001*0.7 / thr = 0.2')
    # setproctitle('Generate Sequences information for transformer')
    if config.USE_DDP:
        torch.multiprocessing.spawn(main,nprocs=config.DDP_OPTS.NGPUS_PER_NODE,join=True)
        cleanup()

    else:
        main(rank=None)