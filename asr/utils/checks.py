import logging
import os

import torch

from asr.exceptions import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def log_pytorch_version_info():
    import torch
    logger.info(f"Pytorch version: {torch.__version__}")


def check_loss(loss):
    """
    Check that loss is valid and will not break training

    Return:
        `True` if loss if valid, `False` therwise.
    """
    is_valid = True
    if torch.isinf(loss):
        is_valid = False
        logger.warning('Received an inf loss.')
    elif torch.isnan(loss):
        is_valid = False
        logger.warning('Received a nan loss.')
    elif loss < 0:
        is_valid = False
        logger.warning('Received a negative loss.')
    return is_valid


def check_for_gpu(device_id):
    from torch import cuda
    if device_id == 'cuda':
        device_id = 0
    elif device_id == 'cpu':
        device_id = -1

    if device_id is not None and (device_id >= 0):
        num_devices_available = cuda.device_count()
        if num_devices_available == 0:
            raise ConfigurationError(
                "Experiment specified a GPU but none are available;"
                " if you want to run on CPU use the override"
                " 'trainer.cuda_device=-1' in the json config file.")
        elif device_id >= num_devices_available:
            raise ConfigurationError(
                f"Experiment specified GPU device {device_id}"
                f" but there are only {num_devices_available} devices "
                f" available.")


def check_for_data_path(data_path: str, dataset_name: str):
    if not os.path.exists(data_path):
        raise ConfigurationError(f"Experiment specified {dataset_name}, "
                                 f"but {data_path} doesn't exist.")
