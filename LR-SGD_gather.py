import os
import sys
import time
import random
import torch
import math
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch.multiprocessing import Process
from math import ceil
from timer import Timer


config = dict(
    synLayerNum=10,
    epoch_num=100,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    backend="nccl",
    workers=4,
    gpu=[0, 1, 2, 3, 0, 0, 0, 0],
    datasets="cifar10",
    model="restnet18"
)

if config["datasets"] == "cifar10":
    from cifar10 import *
else:
    from qmnist import *
config["lr"] *= config["workers"]


def sin_index(generatNum, totalLayerNum, device):
    if totalLayerNum < generatNum:                                          # 如果生成数量大于层数，变成同步所有层
        generatNum = totalLayerNum
    x = [random.uniform(-1.570796326794897, 1.570796326794897) for _ in range(generatNum)]
    seq = [int((math.sin(xi) + 1) / 2 * totalLayerNum) for xi in x]
    indexTensor = torch.zeros((totalLayerNum)).to(device)
    for index in seq:
        indexTensor[index] = 1
    return indexTensor


def rand_index(generatNum, totalLayerNum, device):                          # 传入要同步的层的数量, 返回生成的index,类型为tensor
    """随机决定同步哪些层的参数"""
    if totalLayerNum < generatNum:                                          # 如果生成数量大于层数，变成同步所有层
        generatNum = totalLayerNum
    seq = random.sample(range(totalLayerNum), generatNum)
    indexTensor = torch.zeros((totalLayerNum)).to(device)
    for index in seq:
        indexTensor[index] = 1
    return indexTensor


def average_gradient(model, device, layerSize, totalLayerNum, timer, epoch, maxLen):
    """权重平均"""

    with timer("generate index", epoch):
        indexTensor = rand_index(config["synLayerNum"], totalLayerNum, device)                  # 设定 同步哪几层

    with timer("process param", epoch):
        flatParam = torch.empty(maxLen, device=device)
        i = 0
        startId = 0
        for param in model.parameters():
            if indexTensor[i] == 1:
                length = layerSize[i]
                flatParam[startId: startId + length] = param.grad.data.view(-1)

                startId += length
            i += 1

        workerParams = [torch.empty_like(flatParam).to(device) for _ in range(size)]
        workerIndexes = [torch.empty_like(indexTensor) for _ in range(size)]

    with timer("all gather", epoch):                                     # 开始gather
        h1 = dist.all_gather(workerParams, flatParam, async_op=True)
        h2 = dist.all_gather(workerIndexes, indexTensor, async_op=True)
        h1.wait()
        h2.wait()

    with timer("average", epoch):
        indexCount = torch.ones_like(indexTensor)                          # 用于存index计数
        for r, flatParamFromWorkers, indexFormWorkers in zip(range(size), workerParams, workerIndexes):
            if r == dist.get_rank():                                        # 自己的数据不需要处理
                continue
            indexCount.add_(indexFormWorkers)
            i = 0                                                           # i用于计层数
            startId = 0                                                     # startId 保存flatParam的开头位置
            for param in model.parameters():
                if indexFormWorkers[i].item() == 1:                         # 第i层需要更新
                    param.grad.view(-1).add_(flatParamFromWorkers[startId: startId + layerSize[i]])
                    startId += layerSize[i]                                 # 开头位置更新
                i += 1
        # print(indexCount)
        i = 0
        for param in model.parameters():
            param.grad.data /= indexCount[i]
            i += 1


def average_weight(model, device, layerSize, totalLayerNum, timer, epoch, maxLen):
    """权重平均"""
    with torch.no_grad():
        with timer("generate index", epoch):
            indexTensor = sin_index(config["synLayerNum"], totalLayerNum, device)                     # 设定 同步哪几层
            # indexTensor = rand_index(config["synLayerNum"], totalLayerNum, device)                  # 设定 同步哪几层

        with timer("process param", epoch):
            flatParam = torch.empty(maxLen, device=device)
            i = 0
            startId = 0
            for param in model.parameters():
                if indexTensor[i] == 1:
                    length = layerSize[i]
                    flatParam[startId: startId + length] = param.data.view(-1)

                    startId += length
                i += 1

            workerParams = [torch.empty_like(flatParam).to(device) for _ in range(size)]
            workerIndexes = [torch.empty_like(indexTensor) for _ in range(size)]

        with timer("all gather", epoch):                                     # 开始gather
            h1 = dist.all_gather(workerParams, flatParam, async_op=True)
            h2 = dist.all_gather(workerIndexes, indexTensor, async_op=True)
            h1.wait()
            h2.wait()
            # print("workerIndexes ", workerIndexes)
            # print("worker param", workerParams)

        with timer("average", epoch):
            indexCount = torch.ones_like(indexTensor)                          # 用于存index计数
            for r, flatParamFromWorkers, indexFormWorkers in zip(range(size), workerParams, workerIndexes):
                if r == dist.get_rank():                                        # 自己的数据不需要处理
                    continue
                indexCount.add_(indexFormWorkers)
                i = 0                                                           # i用于计层数
                startId = 0                                                     # startId 保存flatParam的开头位置
                for param in model.parameters():
                    if indexFormWorkers[i].item() == 1:                         # 第i层需要更新
                        param.data.view(-1).add_(flatParamFromWorkers[startId: startId + layerSize[i]])
                        startId += layerSize[i]                                 # 开头位置更新
                    i += 1
            # print(indexCount)
            i = 0
            for param in model.parameters():
                param.data /= indexCount[i]
                i += 1

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
    log = open("loggather"+str(rank)+".txt", "a")
    print(config, file=log)
    train_set, test_set, bsz = partition_dataset()

    device = torch.device("cuda:{}".format(config["gpu"][rank]) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["gpu"][rank])

    torch.cuda.set_device(device)
    print(device)

    if config["model"] == "restnet18":
        model = ResNet18().to(device)
    else:
        model = DenseNet121().to(device)
    # group_decay, group_no_decay = group_weight(model)
    # optimizer = optim.SGD([dict(params=group_decay, weight_decay=config["weight_decay"]), dict(params=group_no_decay, weight_decay=.0)], lr=config["lr"], momentum=config["momentum"])
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(config["epoch_num"]/2), int(config["epoch_num"] * 0.833)], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(train_set.dataset) / float(bsz))

    totalLayerNum = 0
    layerSize = []
    for param in model.parameters():
        totalLayerNum += 1
        layerSize.append(len(param.view(-1)))
    sizeTensor = torch.Tensor(layerSize)
    maxSizes, _ = torch.topk(sizeTensor, config["synLayerNum"], largest=True, sorted=False)
    maxLen = torch.sum(maxSizes).int()
    print("totalLayerNum", totalLayerNum)

    timer = Timer(rank)
    for epoch in range(config["epoch_num"]):
        with timer("epoch spend", epoch):
            epoch_loss = 0.0
            for data, target in train_set:
                with timer("forward & backward", epoch):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    loss.backward()

                with timer("optimizer step", epoch):
                    optimizer.step()
                average_weight(model, device, layerSize, totalLayerNum, timer, epoch, maxLen.item())  # 与average_gradient二选一
            scheduler.step()

        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

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
            p.join()
