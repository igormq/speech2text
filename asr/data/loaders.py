from asr.data import speech2text

from asr.utils import klass_fullname
import copy
from asr.common import Params
import logging
from asr import samplers
logger = logging.getLogger(__name__)
loader_names = {**speech2text.loader_names}


def by_name(name):
    try:
        return loader_names[name]
    except KeyError:
        raise ValueError(
            f'Loader name {name} not found. '
            f'Loaders available: [{",".join(loader_names.keys())}]')


def from_params(params, **extra):
    params = copy.copy(params)

    klass_name = params.pop('type') if isinstance(params,
                                                  (dict, Params)) else params
    if isinstance(params, str):
        params = {}

    klass = by_name(klass_name)
    params = {**params, **extra}

    logger.info(
        f'Instantiating class `{klass_fullname(klass)}` with params {params}')

    return klass(**params)
