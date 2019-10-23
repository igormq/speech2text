import torch
import torch.optim.lr_scheduler
import copy
import logging
from asr.utils import klass_fullname
from asr.common import Params

logger = logging.getLogger(__name__)

lr_schedulers_names = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multi_step": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cyclic": torch.optim.lr_scheduler.CyclicLR
}


def by_name(name):
    try:
        return lr_schedulers_names[name]
    except KeyError:
        raise ValueError(
            f'Scheduler name {name} not found. '
            f'Schedulers available: [{",".join(lr_schedulers_names.keys())}]')


def from_params(params, optimizer, **extra):
    params = copy.copy(params)

    klass_name = params.pop('type') if isinstance(params,
                                                  (dict, Params)) else params
    if isinstance(params, str):
        params = {}

    klass = by_name(klass_name)
    params = {**params, **extra}

    logger.info(
        f'Instantiating class `{klass_fullname(klass)}` with params {params}')

    return klass(optimizer, **params)
