import logging
import re

import torch
import copy
from asr.utils import klass_fullname
from asr.common import Params

logger = logging.getLogger(__name__)

optimizers_names = {
    "adam": torch.optim.Adam,
    "sparse_adam": torch.optim.SparseAdam,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "adamax": torch.optim.Adamax,
    "averaged_sgd": torch.optim.ASGD
}


class LARC(torch.optim.Optimizer):
    r"""Implements Layer-wise Adaptive Rate Control

    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.

    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.
    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim
    Optimizer.

    It can even be used in conjunction with Apex

    Args:
        optimizer (torch.optim.Optimizer): Pytorch optimizer to wrap and modify learning rate for.
        eta: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is
            set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning
            rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr

    Example:
        >>> model = ...
        >>> optim = torch.optim.Adam(model.parameters(), lr=...)
        >>> optim = LARC(optim)

    """

    def __init__(self, optimizer, eta=0.02, clip=True, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group[
                    'weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.eta * (param_norm) / (
                            grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals
                            # `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]


def by_name(name):
    try:
        return optimizers_names[name]
    except KeyError:
        raise ValueError(
            f'Optimizer name {name} not found. '
            f'Optimizers available: [{",".join(optimizers_names.keys())}]')


def from_params(params, model_params, **extra):
    params = copy.copy(params)

    klass_name = params.pop('type') if isinstance(params,
                                                  (dict, Params)) else params
    world_size = extra.pop('world_size', 1)
    scale = params.pop('scale', False)
    larc_eta = params.pop('larc_eta', None)

    if scale and world_size > 1:
        logger.info(f'Scaling base lr by a factor of {world_size:d}.')

    if isinstance(params, str):
        params = {}

    klass = by_name(klass_name)
    params = {**params, **extra}

    groups = params.pop("param_groups", None)
    if groups:
        # The input to the optimizer is list of dict.
        # Each dict contains a "parameter group" and groups specific options,
        # e.g., {'params': [list of parameters], 'lr': 1e-3, ...}
        # Any config option not specified in the additional options (e.g.
        # for the default group) is inherited from the top level config.
        # see: http://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options
        #
        # groups contains something like:
        # "param_groups": [
        #       [["regex1", "regex2"], {"lr": 1e-3}],
        #       [["regex3"], {"lr": 1e-4}]
        # ]
        # In addition to any parameters that match group specific regex,
        # we also need a group for the remaining "default" group.
        # Those will be included in the last entry of param_groups.
        param_groups = [{'params': []} for _ in range(len(groups) + 1)]
        # add the group specific kwargs
        for k in range(len(groups)):  # pylint: disable=consider-using-enumerate
            group_kwargs = groups[k][1]
            if isinstance(group_kwargs, Params):
                group_kwargs = group_kwargs.as_dict()

            # Multiply the learning rate by the world_size
            if group_kwargs.get('lr') and scale:
                group_kwargs['lr'] *= world_size

            param_groups[k].update(group_kwargs)

        regex_use_counts = {}
        param_group_names = [set() for _ in range(len(groups) + 1)]
        for name, param in model_params:
            # Determine the group for this param.
            group_index = None
            for k, group_regexes in enumerate(groups):
                for regex in group_regexes[0]:
                    if regex not in regex_use_counts:
                        regex_use_counts[regex] = 0
                    if re.search(regex, name):
                        if group_index is not None and group_index != k:
                            raise ValueError(
                                f"{name} was specified in two separate parameter groups"
                            )
                        group_index = k
                        regex_use_counts[regex] += 1

            if group_index is not None:
                param_groups[group_index]['params'].append(param)
                param_group_names[group_index].add(name)
            else:
                # the default group
                param_groups[-1]['params'].append(param)
                param_group_names[-1].add(name)

        # log the param groups
        logger.info("Done constructing parameter groups.")
        for k in range(len(groups) + 1):
            group_options = {
                key: val
                for key, val in param_groups[k].items() if key != 'params'
            }
            logger.info("Group %s: %s, %s", k, list(param_group_names[k]),
                        group_options)
        # check for unused regex
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning(
                    "When constructing parameter groups, "
                    " %s not match any parameter name", regex)

    else:
        param_groups = [param for name, param in model_params]

    if 'lr' in params and scale:
        params['lr'] *= world_size

    # Log the number of params to optimize
    num_params = 0
    for param_group in param_groups:
        if isinstance(param_group, dict):
            num_params += sum(param.numel() for param in param_group["params"])
        else:
            num_params += param_group.numel()
    logger.info(f"Number of trainable parameters: {num_params}")

    logger.info(
        f'Instantiating class `{klass_fullname(klass)}` with params {params}')

    optim = klass(param_groups, **params)
    if larc_eta:
        logger.info('LARC optimizer enabled. Wrapping up.')
        return LARC(optim, eta=larc_eta)

    return optim
