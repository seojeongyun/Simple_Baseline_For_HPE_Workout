import argparse

import torch
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet18

# NOTE:
# - This function must be called in each process participating in distributed training.
# - Execution will block until all processes have called init_process_group.
# - Call only after setting up all necessary configurations for distributed training.

torch.distributed.init_process_group(
    backend='nccl',  # Backend for distributed training
    # - 'nccl': Recommended for GPU training
    # - 'gloo': Recommended for CPU training

    init_method='tcp://202.31.136.169:17771',  # Initialization method
    # - For single-machine multi-GPU with NCCL,
    #   use 'tcp://localhost:<port>'

    world_size=ngpus_per_node,  # Total number of processes in the job
    # - For single machine: equal to the number of GPUs

    rank=process_id  # Unique ID for the current process
    # - Ranges from 0 to (world_size - 1)
)

# Create a DistributedSampler for distributed data parallel training
# - Splits the dataset across multiple processes so that each process
#   gets a unique subset of data (no overlap between processes)
# - Generates a new shuffled index order each epoch to ensure randomness
# - Helps balance the workload evenly among all processes
train_sampler = DistributedSampler(
    dataset=train_set,
    shuffle=True
)

# Wrap the sampler with a BatchSampler
# - Groups the indices from the DistributedSampler into batches
# - drop_last=True: drops the last incomplete batch if dataset size is not divisible by batch_size
batch_sampler_train = torch.utils.data.BatchSampler(
    train_sampler,
    opts.batch_size,
    drop_last=True
)

# Create the DataLoader for training
# - Uses the batch sampler to load data in distributed training
# - num_workers: number of subprocesses for data loading (parallel I/O)
train_loader = DataLoader(
    train_set,
    batch_sampler=batch_sampler_train,
    num_workers=opts.num_workers
)

# Wrap the model with DistributedDataParallel (DDP)
# - Replicates the model on each GPU and splits the input data across GPUs
# - Each GPU computes its own forward and backward pass
# - Gradients from all GPUs are synchronized and averaged across processes
# - Model weights are updated consistently on all GPUs
# - More efficient and scalable than DataParallel (DP), especially for multi-GPU training
model = DistributedDataParallel(
    module=model,
    device_ids=[local_gpu_id]  # Specifies the GPU device for this process
)

import torch.multiprocessing as mp

def train(rank, world_size):
    # Training logic for each process in distributed training
    # - 'rank' is the unique process ID (0 to world_size-1)
    # - This function should include process group initialization
    #   and the model/data setup for the given rank
    pass

if __name__ == '__main__':
    world_size = 4  # Total number of processes to launch

    # Launch multiple processes for distributed training
    # - fn: function to run in each process (first argument must be 'rank')
    # - args: arguments passed to 'fn' (excluding 'rank')
    # - nprocs: number of processes to spawn
    mp.spawn(
        train,              # Function to execute in each process
        args=(world_size,), # Arguments for 'train' (rank is passed automatically)
        nprocs=world_size   # Number of processes to launch
    )















def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument('--root', type=str, default='./cifar')
    parser.add_argument('--local_rank', type=int)
    return parser


def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://202.31.136.169:' + str(opts.port),
                                         world_size=opts.ngpus_per_node,
                                         rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)


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

def main(rank, opts):
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    train_set = CIFAR10(root=opts.root,
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)

    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)

    model = resnet18(pretrained=False)
    model = model.cuda(local_gpu_id)
    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])

    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01,
                                weight_decay=0.0005,
                                momentum=0.9)

    print(f'[INFO] : ���� ����')
    for epoch in range(opts.epoch):

        model.train()
        train_sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'[INFO] : {epoch} ���� epoch ����')

    print(f'[INFO] : Distributed ���� ����������')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(main,
             args=(opts,),
             nprocs=opts.ngpus_per_node,
             join=True)
