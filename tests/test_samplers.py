import pytest
from asr.samplers import BucketingSampler, DistributedBucketingSampler
import math


@pytest.mark.parametrize('size', [4, 12])
def test_bucketing_sampler(size):
    data_source = list(range(size))

    sampler = BucketingSampler(data_source, batch_size=1)
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source == list(list(s)[0] for s in sampler)

    sampler = BucketingSampler(data_source, batch_size=1)
    assert len(sampler) == size
    assert data_source == list(list(s)[0] for s in sampler)

    sampler.set_epoch(1)
    assert data_source != list(list(s)[0] for s in sampler)
    assert data_source == list(sorted(list(list(s)[0] for s in sampler)))

    sampler = BucketingSampler(data_source, batch_size=1, first_epoch='desc')
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source[::-1] == list(list(s)[0] for s in sampler)

    with pytest.raises(ValueError):
        sampler = BucketingSampler(data_source,
                                   batch_size=1,
                                   first_epoch='invalid')

    sampler = BucketingSampler(data_source, batch_size=1, first_epoch=None)
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source[::-1] != list(list(s)[0] for s in sampler)

    sampler = BucketingSampler(data_source, batch_size=2, first_epoch='asc')
    sampler.set_epoch(0)
    assert len(sampler) == size // 2
    assert [data_source[i:i + 2] for i in range(0, len(data_source), 2)
            ] == list(list(s) for s in sampler)


@pytest.mark.parametrize('size', [4, 12])
def test_dist_bucketing_sampler(size):
    data_source = list(range(size))

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=1,
                                          rank=0)
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source == list(list(s)[0] for s in sampler)

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=1,
                                          rank=0)
    assert len(sampler) == size
    assert data_source == list(list(s)[0] for s in sampler)

    sampler.set_epoch(1)
    assert data_source != list(list(s)[0] for s in sampler)
    assert data_source == list(sorted(list(list(s)[0] for s in sampler)))

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=1,
                                          rank=0,
                                          first_epoch='desc')
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source[::-1] == list(list(s)[0] for s in sampler)

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=1,
                                          rank=0,
                                          first_epoch=None)
    sampler.set_epoch(0)
    assert len(sampler) == size
    assert data_source[::-1] != list(list(s)[0] for s in sampler)

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=2,
                                          num_replicas=1,
                                          rank=0,
                                          first_epoch='asc')
    sampler.set_epoch(0)
    assert len(sampler) == size // 2
    assert [data_source[i:i + 2] for i in range(0, len(data_source), 2)
            ] == list(list(s) for s in sampler)

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=2,
                                          rank=0)
    sampler.set_epoch(0)
    assert len(sampler) == size // 2
    assert data_source[::2] == list(list(s)[0] for s in sampler)

    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=2,
                                          rank=1)
    sampler.set_epoch(0)
    assert len(sampler) == size // 2
    assert data_source[1::2] == list(list(s)[0] for s in sampler)

    data_source = list(range(size - 1))
    sampler = DistributedBucketingSampler(data_source,
                                          batch_size=1,
                                          num_replicas=2,
                                          rank=1)
    sampler.set_epoch(0)
    assert len(sampler) == int(math.ceil((size - 1) * 1.0 / 2))
    assert data_source[1::2] + [data_source[0]] == list(
        list(s)[0] for s in sampler)
