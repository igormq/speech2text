from asr.data import datasets, loaders
from asr import samplers
from asr.exceptions import ConfigurationError
from asr.utils.checks import check_for_data_path
import logging

logger = logging.getLogger()


def datasets_from_params(params):
    """
    Load all the datasets specified by the config.
    """
    sets = {}
    for split in ['train', 'val', 'test']:
        dataset_params = params.pop(f'{split}_dataset', None)

        if dataset_params is None:
            if split == 'train':
                ConfigurationError('Must provide train_dataset params.')
            continue

        data_path = dataset_params.get('manifest_filepath', None)
        if data_path is not None:
            check_for_data_path(data_path, 'manifest_filepath')

        sets[split] = datasets.from_params(dataset_params)

    return sets


def loaders_from_params(params,
                        distributed=False,
                        world_size=1,
                        first_epoch='asc'):
    """
    Load all loaders specified by the config.
    """

    sets = datasets_from_params(params)

    data_loaders = {}
    for split in ['train', 'val', 'test']:
        if split not in sets:
            continue

        loader_params = params.pop(f'{split}_data_loader', None)
        if not loader_params:
            loader_params = params.get('data_loader')

        # TODO: put it in a better place
        if distributed:
            logger.info('Using distributed bucketing sampler')
            sampler = samplers.DistributedBucketingSampler
        else:
            logger.info('Using normal bucketing sampler')
            sampler = samplers.BucketingSampler

        batch_sampler = sampler(sets[split],
                                batch_size=params['trainer']['batch_size'],
                                first_epoch=first_epoch)

        data_loaders[split] = loaders.from_params(loader_params,
                                                  dataset=sets[split],
                                                  batch_sampler=batch_sampler)
    return data_loaders
