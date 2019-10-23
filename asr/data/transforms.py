from asr.data import speech2text

from asr.utils import klass_fullname
import copy
from asr.common import Params
import logging

logger = logging.getLogger(__name__)


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


default_transforms = {'compose': Compose}
transform_names = {**speech2text.transform_names, **default_transforms}


def by_name(name):
    try:
        return transform_names[name]
    except KeyError:
        raise ValueError(
            f'Tranform name {name} not found. '
            f'Tranforms available: [{",".join(transform_names.keys())}]')


def from_params(params, **extra):
    params = copy.copy(params)

    klass_name = params.pop('type') if isinstance(params,
                                                  (dict, Params)) else params
    if isinstance(params, str):
        params = {}

    klass = by_name(klass_name)

    if klass_name == 'compose':
        for i in range(len(params['transforms'])):
            params['transforms'][i] = from_params(params['transforms'][i],
                                                  **extra)

    params = {**params, **extra}

    logger.info(
        f'Instantiating class `{klass_fullname(klass)}` with params {params}')

    return klass(**params)
