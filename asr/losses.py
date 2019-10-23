import logging

import torch
import torch.nn.functional as F
from asr.utils import klass_fullname
import copy
from asr.common import Params
logger = logging.getLogger(__name__)


class CTCLoss(torch.nn.Module):
    def __init__(self, blank=0, backend='baidu'):
        super().__init__()

        if backend not in ('pytorch', 'baidu'):
            raise ValueError('Backend must be one of [`pytorch`, `baidu`]')

        self.blank = blank
        self.backend = backend

        if self.backend == 'baidu':
            from warpctc_pytorch import _CTC
            logger.info('Using CTCLoss Baidu backend.')

            self.ctc_loss = _CTC.apply
        else:
            logger.info('Using CTCLoss PyTorch backend.')
            self.ctc_loss = F.ctc_loss

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        log_probs = log_probs.transpose(0, 1)  # B, N, * to N, B, *
        if self.backend == 'baidu':
            assert len(targets.size()) == 1  # labels must be 1 dimensional
            loss = self.ctc_loss(log_probs,
                                 targets.to('cpu').type(torch.int),
                                 input_lengths.to('cpu').type(torch.int),
                                 target_lengths.to('cpu').type(torch.int),
                                 False, False, self.blank)
        else:
            loss = self.ctc_loss(log_probs,
                                 targets,
                                 input_lengths,
                                 target_lengths,
                                 blank=self.blank,
                                 reduction='sum',
                                 zero_infinity=True)

        return (loss / log_probs.shape[1]).to(log_probs.device)


losses_names = {'ctc': CTCLoss}


def by_name(name):
    try:
        return losses_names[name]
    except KeyError:
        raise ValueError(
            f'Loss name {name} not found. '
            f'Losses available: [{",".join(losses_names.keys())}]')


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
