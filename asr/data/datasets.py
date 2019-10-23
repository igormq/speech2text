from asr.data import speech2text

from asr.utils import klass_fullname
import copy
from asr.common import Params
from asr.data import transforms
import logging

logger = logging.getLogger(__name__)
dataset_names = {**speech2text.dataset_names}


def list_available():
    return list(dataset_names.keys())


def by_name(name):
    try:
        return dataset_names[name]
    except KeyError:
        raise ValueError(
            f'Dataset name {name} not found. '
            f'Datasets available: [{",".join(dataset_names.keys())}]')


def from_params(params, **extra):
    params = copy.copy(params)

    klass_name = params.pop('type') if isinstance(params,
                                                  (dict, Params)) else params
    if isinstance(params, str):
        params = {}
    elif isinstance(params, Params):
        params = params.as_dict(quiet=True)

    transforms_params = params.get('transforms', None)
    if transforms_params:
        transform = transforms.from_params(transforms_params)
        params['transforms'] = transform

    target_transforms_params = params.get('target_transforms', None)
    if target_transforms_params:
        target_transform = transforms.from_params(target_transforms_params)
        params['target_transforms'] = target_transform

    klass = by_name(klass_name)
    params = {**params, **extra}

    logger.info(
        f'Instantiating class `{klass_fullname(klass)}` with params {params}')

    return klass(**params)
