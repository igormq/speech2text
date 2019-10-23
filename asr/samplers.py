import math

import torch
from torch.utils.data.sampler import Sampler

from asr.utils import klass_fullname
import logging

logger = logging.getLogger(__name__)


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, first_epoch='asc'):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples
        together.

        Args:
            data_source (dataset): dataset source. Assuming they are in order.
            batch_size (int): batch size
            first_epoch (str): First epoch behavior, one of [`asc`, `desc`, None]. If `asc`, will
                kept the bins in ascending order, elif `desc` will kept the bins in descending
                order, else will shuffle the bins in the first epoch. Default `asc`
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.first_epoch = first_epoch
        self.epoch = 0

        if first_epoch not in ('asc', 'desc', None):
            raise ValueError(
                'Arg. `first_epoch` must be one of [`asc`, `desc`, None].')

        ids = list(range(0, len(data_source)))
        self.bins = [
            ids[i:i + self.batch_size]
            for i in range(0, len(ids), self.batch_size)
        ]

        if self.first_epoch == 'desc':
            self.bins = self.bins[::-1]

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.epoch or (self.epoch == 0 and self.first_epoch is None):
            g = torch.Generator()
            g.manual_seed(self.epoch)
            bin_ids = list(torch.randperm(len(self.bins), generator=g))
            self.bins = [self.bins[i] for i in bin_ids]

        for ids in self.bins:
            yield iter(ids)

    def __len__(self):
        return len(self.bins)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedBucketingSampler(BucketingSampler):
    def __init__(self,
                 data_source,
                 batch_size=1,
                 first_epoch='asc',
                 num_replicas=None,
                 rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples
        together.

        Args:
            data_source (dataset): dataset source. Assuming they are in order.
            batch_size (int): batch size
            first_epoch (str): First epoch behavior, one of [`asc`, `desc`, None]. If `asc`, will
                kept the bins in ascending order, elif `desc` will kept the bins in descending
                order, else will shuffle the bins in the first epoch. Default `asc`
            num_replicas (int): number of processes in the current process group
            rank (int): rank of the current process group
        """
        super().__init__(data_source, batch_size, first_epoch)

        import torch.distributed as dist
        if not dist.is_initialized():
            logger.warning(
                'torch.distributed was not initialized. Setting `num_replicas=1` and `rank=0`'
            )
            num_replicas = 1
            rank = 0
        else:
            if num_replicas is None:
                try:
                    num_replicas = dist.get_world_size()
                except ValueError:
                    raise RuntimeError('Horovod has not been initialized')
            if rank is None:
                try:
                    rank = dist.get_rank()
                except ValueError:
                    raise RuntimeError('Horovod has not been initialized')

        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = int(
            math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.epoch or (self.epoch == 0 and self.first_epoch is None):
            g = torch.Generator()
            g.manual_seed(self.epoch)
            bin_ids = list(torch.randperm(len(self.bins), generator=g))
            self.bins = [self.bins[i] for i in bin_ids]

        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size

        # Get every Nth bin, starting from rank
        samples = bins[self.rank:self.total_size:self.num_replicas]

        return iter(samples)

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return (
            f'BucketingSampler({klass_fullname(self.data_source)}, '
            f'batch_size={self.batch_size}, '
            f'first_epoch={self.first_epoch}, num_replicas={self.num_replicas}, '
            f'rank={self.rank})')
