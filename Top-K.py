import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from topk_reducer import TopKReducer

from torch.multiprocessing import Process
from math import ceil
from timer import Timer

config = dict(
    compression=1/20,
    epoch_num=100,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    backend="gloo",
    workers=8,
    gpu=[0, 0, 1, 1, 2, 2, 3, 3],
    datasets="qmnist",
    model="resnet18"
)

if config["datasets"] == "cifar10":
    from cifar10 import *
else:
    from qmnist import *
config["lr"] *= config["workers"]


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


def run(rank, size):
    """ Distributed S 分布式同步参数平均"""
    log = open("logtopk"+str(rank)+".txt", "a")
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
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    """新加的损失函数"""
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["epoch_num"]/2), int(config["epoch_num"] * 0.833)], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    timer = Timer(rank)
    reducer = TopKReducer(device, timer, config["compression"])
    for epoch in range(config["epoch_num"]):
        with timer("epoch spend", epoch):
            epoch_loss = 0.0
            for data, target in train_set:
                with timer("forward & backward", epoch):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    """改变的损失函数"""
                    # loss = F.nll_loss(output, target)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    loss.backward()

                grad_in = [param.grad.data for param in model.parameters()]
                grad_out = grad_in
                reducer.reduce(grad_in, grad_out, epoch)

                with timer("optimizer step", epoch):
                    optimizer.step()
            scheduler.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches, 'lr:', optimizer.state_dict()['param_groups'][0]['lr'])
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
    if len(sys.argv) > 2:
        print("rank:", int(sys.argv[1]), "size:", int(sys.argv[2]))
        init_process(int(sys.argv[1]), int(sys.argv[2]), run)

    else:
        size = config["workers"]
        processes = []
        for rank in range(size):
            p = Process(target=init_process, args=(rank, size, run, config["backend"]))
            p.start()
            processes.append(p)

        for p in processes:
            # print(p.name)
            p.join()
