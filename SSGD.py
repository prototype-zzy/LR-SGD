import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.multiprocessing import Process
from math import ceil
from timer import Timer

config = dict(
    epoch_num=2,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    backend="nccl",
    workers=4,
    gpu=[0, 1, 2, 3],
    datasets="cifar10",
    model="restnet18"
)

if config["datasets"] == "cifar10":
    from cifar10 import *
else:
    from qmnist import *
config["lr"] *= config["workers"]


def average_gradients(model):
    """ Gradient averaging. 利用all_reduce求梯度平均值"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size



def average_weight(model):
    size = float(dist.get_world_size())
    rank = dist.get_rank()
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            param.data /= size


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    return group_decay, group_no_decay


""" Distributed Synchronous SGD Example  分布式同步SGD方式例子"""
def run(rank, size):
    log = open("logssgd"+str(rank)+".txt", "a")
    print(config, file=log)
    torch.manual_seed(1234)
    train_set, test_set, bsz = partition_dataset()

    device = torch.device("cuda:{}".format(config["gpu"][rank]) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["gpu"][rank])
    print(device)

    if config["model"] == "restnet18":
        model = ResNet18().to(device)
    else:
        model = DenseNet121().to(device)

    group_decay, group_no_decay = group_weight(model)
    optimizer = optim.SGD([dict(params=group_decay, weight_decay=config["weight_decay"]), dict(params=group_no_decay, weight_decay=.0)], lr=config["lr"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["epoch_num"]/2), int(config["epoch_num"] * 0.833)], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(train_set.dataset) / float(bsz))

    timer = Timer(rank)

    for epoch in range(config["epoch_num"]):
        #with timer("epoch spend", epoch):
        epoch_loss = 0.0
        for data, target in train_set:
            with timer("forward & backward", epoch):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                epoch_loss += loss.item()
                loss.backward()
            with timer("all reduce & average", epoch):
                average_gradients(model)
            with timer("optimizer step", epoch):
                optimizer.step()

        scheduler.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches, 'lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        # print(optimizer.state_dict()['param_groups'])
        with timer("test accuracy", epoch):
            timer.test_accuracy(epoch, model, test_set, device, log)
    if rank == 0:
        print(timer.summary())
    print(timer.summary(), file=log)
    log.close()




def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = config["workers"]
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run, config["backend"]))
        p.start()
        processes.append(p)

    for p in processes:
        # print(p.name)
        p.join()

