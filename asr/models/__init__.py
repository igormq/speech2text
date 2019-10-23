# flake8: noqa
import logging
import os
import re

import torch

from asr.common import Params
from asr.data import Alphabet

from .speech2text import DeepSpeech2, DeepSpeech2SeanNaren

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = 'best-model.pth'
CONFIG_NAME = 'config.yaml'

model_names = {'ds2-seannaren': DeepSpeech2SeanNaren, 'ds2': DeepSpeech2}


def by_name(model_name):
    try:
        return model_names[model_name]
    except KeyError:
        raise ValueError(f'Model name {model_name} not found. ' f'Models available: [{",".join(model_names.keys())}]')


def from_params(params, **extra):
    klass_name = params.pop('type', params)

    alphabet = extra.pop('alphabet', None)
    if params is not None:
        extra.setdefault('num_classes', len(alphabet))

    if not len(params):
        params = {}

    params = {**params, **extra}

    klass = by_name(klass_name)

    return klass(**params)


def load(model_params, serialization_dir, weights_file=None, device='cpu'):
    weights_file = weights_file or os.path.join(serialization_dir, DEFAULT_WEIGHTS)

    # Load vocabulary from file
    alphabet_file = os.path.join(serialization_dir, 'vocabulary', 'alphabet')
    # If the config specifies a vocabulary subclass, we need to use it.
    alphabet = Alphabet.from_file(alphabet_file)

    default_params = {'num_classes': len(alphabet)}

    # Loading weights
    logger.info(f'Loading weights from {weights_file}.')
    state_dict = torch.load(weights_file, map_location='cpu')
    state_dict = {re.sub(r'^module.', '', k): v for k, v in state_dict.items()}

    model_name = model_params.pop('type')
    model_params = {**default_params, **model_params}

    model = by_name(model_name)(**model_params)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def load_archive(serialization_dir, overrides="{}", weights_file=None):

    if not serialization_dir.endswith('.pth'):
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            logger.error(f'config file {config_file} does not exist, unable to load params')

        model_params = Params.load(config_file, overrides).get('model')

        weights_file = weights_file or os.path.join(serialization_dir, 'models', DEFAULT_WEIGHTS)

        return model_params, weights_file

    return None, serialization_dir
