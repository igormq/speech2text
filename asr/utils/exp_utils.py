import logging
import os
import random
import shutil

import numpy
import torch

from asr.common import Params
from asr.exceptions import ConfigurationError
from asr.models import CONFIG_NAME
from asr.utils.checks import log_pytorch_version_info

logger = logging.getLogger(__name__)


def create_serialization_dir(params, serialization_dir, reset):
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical
    configuration.

    Parameters:
        params (Params): A parameter object specifying an AllenNLP Experiment.
        serialization_dir (str): The directory in which to save results and logs.
        reset (bool): If ``True``, we will overwrite the serialization directory if it already
            exists.
    """

    if os.path.exists(serialization_dir) and reset:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir):
        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError(
                "The serialization directory already exists but doesn't "
                f"contain a {CONFIG_NAME}. You probably gave the wrong directory."
            )

        loaded_params = Params.load(recovered_config_file)

        # Check whether any of the training configuration differs from the configuration we are
        # resuming.  If so, warn the user that training may fail.
        fail = False
        flat_params = params.as_flat_dict()
        flat_loaded = loaded_params.as_flat_dict()
        for key in flat_params.keys() - flat_loaded.keys():
            logger.error(
                f"Key '{key}' found in training configuration but not in the "
                f"serialization directory we're recovering from.")
            fail = True
        for key in flat_loaded.keys() - flat_params.keys():
            logger.error(
                f"Key '{key}' found in the serialization directory we're recovering "
                f"from but not in the training config.")
            fail = True
        for key in flat_params.keys():
            if flat_params.get(key, None) != flat_loaded.get(key, None):
                logger.error(
                    f"Value for '{key}' in training configuration does not match that "
                    f"the value in the serialization directory we're recovering from: "
                    f"{flat_params[key]} != {flat_loaded[key]}")
                fail = True

        if fail:
            raise ConfigurationError(
                "Training configuration does not match the configuration we're "
                "recovering from.")

    os.makedirs(serialization_dir, exist_ok=True)


def get_frozen_and_tunable_parameter_names(model):
    frozen_parameter_names = []
    tunable_parameter_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameter_names.append(name)
        else:
            tunable_parameter_names.append(name)
    return [frozen_parameter_names, tunable_parameter_names]


def dump_metrics(file_path, metrics, log=False):
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


def prepare_environment(params):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.

    Params:
        params: Params object or dict, required.
            A ``Params`` object or dict holding the json parameters.
    """

    seed = params.pop_int("random_seed", None)
    numpy_seed = params.pop_int("numpy_seed", None)
    torch_seed = params.pop_int("pytorch_seed", None)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    log_pytorch_version_info()
