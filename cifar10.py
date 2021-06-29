import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torchvision
import models
from torchvision import datasets, transforms
from partitioner import DataPartitioner


def partition_dataset(bsz=128):
    """ Partitioning Cifar10 """

    trainset = datasets.CIFAR10('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                               ]))
    testset = datasets.CIFAR10('./data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                               ]))

    size = dist.get_world_size()
    # bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=int(bsz), shuffle=True)
    partition = DataPartitioner(testset, partition_sizes)
    partition = partition.use((dist.get_rank()))
    test_set = torch.utils.data.DataLoader(partition, batch_size=int(bsz), shuffle=True)
    return train_set, test_set, bsz


def DenseNet121():
    return models.DenseNet121()

def DenseNet169():
    return models.DenseNet169()

def DenseNet201():
    return models.DenseNet201()

def DenseNet161():
    return models.DenseNet161()

def densenet_cifar():
    return models.densenet_cifar()

def ResNet18(use_batchnorm=True):
    return models.ResNet18(use_batchnorm)

def ResNet34(use_batchnorm=True):
    return models.ResNet34(use_batchnorm)

def ResNet50(use_batchnorm=True):
    return models.ResNet50(use_batchnorm)

def ResNet101(use_batchnorm=True):
    return models.ResNet101(use_batchnorm)

def ResNet152(use_batchnorm=True):
    return models.ResNet152(use_batchnorm)

def GoogLeNet():
    return models.GoogLeNet()
