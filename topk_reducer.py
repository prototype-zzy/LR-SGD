import torch
import torch.distributed
import numpy as np
from timer import Timer

class Reducer:
    def __init__(self, device):
        self.rng = np.random.RandomState()
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out):
        """Return communicated bits"""
        raise NotImplementedError()


class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, device, timer, compression=1 / 244):
        super().__init__(device)
        self.timer = timer
        self.compression = compression

    def reduce(self, grad_in, grad_out, epoch):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("prepare grad size", epoch):
            bits_communicated = 0

            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]                                                            # 记录每一层tensor的开始位置
            for tensor in grad_in:
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))      # 最大取数数量 max(0.5 * 压缩率 * 一层参数梯度tensor的元素数量)
                flatgrad_size += top_size                                               # 最大数量累加
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]                                        # 展平后起始位置合集: 0 ~ n-1
            flatgrad_end_idx = tensor_idx[1:]                                           # 展平后结束位置合集: 1 ~ n
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("prepare grad value", epoch):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)    # 取tenor中绝对值最大的top_size个的位置，放入positions中
                values = tensor.view(-1)[positions].contiguous()                            # 把值放进values中
                flat_values[start:end] = values
                flat_positions[start:end] = positions

            for tensor, start, end in zip(
                grad_in, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]

        with self.timer("all gather", epoch):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated += n_bits(flat_values) + n_bits(flat_positions)

        with self.timer("average", epoch):

            for tensor, out, start, end in zip(grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers

        return bits_communicated


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()
