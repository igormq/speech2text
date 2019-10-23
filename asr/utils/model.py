import logging
import re

from asr.utils.exp_utils import get_frozen_and_tunable_parameter_names

logger = logging.getLogger(__name__)


def freeze_params(model, no_grad_regexes=()):
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(
        model)
    logger.info("Following parameters are Frozen  (without gradient):")
    logger.info(', '.join(frozen_parameter_names))
    logger.info("Following parameters are Tunable (with gradient):")
    logger.info(', '.join(tunable_parameter_names))
