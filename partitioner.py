import random

class Partition(object):
    """ Dataset partitioning helper"""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):     # 如果不带sizes参数则将划分样本为7:2:1的比例
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])             # 将样本划分为0~该分组的长度
            indexes = indexes[part_len:]                            # 下一组的开头

    def use(self, partition):                                       # 获取数据集划分为partition的分组
        return Partition(self.data, self.partitions[partition])
