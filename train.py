""" Command to train a model
It requires a configuration file and a directory in which to write the results.


Heavily inspired on SeanNaren DeepSpeech2, AllenNLP, and NVIDIA/OpenSeq2Seq
"""

import argparse
import logging
import os
import re
from distutils.dir_util import copy_tree

import torch
import torchaudio

from asr import losses, models
from asr.common.params import Params
from asr.data import Alphabet
from asr.engine import Trainer
from asr.exceptions import ConfigurationError
from asr.utils.dataset import loaders_from_params
from asr.utils.exp_utils import create_serialization_dir, prepare_environment
from asr.utils.io_utils import prepare_global_logging
from asr.utils.model import freeze_params
import torch.distributed as dist

logger = logging.getLogger('asr')

CONFIG_NAME = 'config.yaml'


def train_model_from_args(args):

    if args.local_rank == 0 and args.prev_output_dir is not None:
        logger.info('Copying results from {} to {}...'.format(args.prev_output_dir, args.serialization_dir))

        copy_tree(args.prev_output_dir, args.serialization_dir, update=True, verbose=True)

    if not os.path.isfile(args.param_path):
        raise ConfigurationError(f'Parameters file {args.param_path} not found.')

    logger.info(f'Loading experiment from {args.param_path} with overrides `{args.overrides}`.')

    params = Params.load(args.param_path, args.overrides)

    prepare_environment(params)

    logger.info(args.local_rank)
    if args.local_rank == 0:
        create_serialization_dir(params, args.serialization_dir, args.reset)

    if args.distributed:
        logger.info(f'World size: {dist.get_world_size()} | Rank {dist.get_rank()} | ' f'Local Rank {args.local_rank}')
        dist.barrier()

    prepare_global_logging(args.serialization_dir, local_rank=args.local_rank, verbosity=args.verbosity)

    if args.local_rank == 0:
        params.save(os.path.join(args.serialization_dir, CONFIG_NAME))

    loaders = loaders_from_params(params,
                                  distributed=args.distributed,
                                  world_size=args.world_size,
                                  first_epoch=args.first_epoch)

    if os.path.exists(os.path.join(args.serialization_dir, "alphabet")):
        alphabet = Alphabet.from_file(os.path.join(args.serialization_dir, "alphabet"))
    else:
        alphabet = Alphabet.from_params(params.pop("alphabet", {}))

    alphabet.save_to_files(os.path.join(args.serialization_dir, "alphabet"))

    loss = losses.from_params(params.pop('loss'))
    model = models.from_params(alphabet=alphabet, params=params.pop('model'))

    trainer_params = params.pop("trainer")
    if args.fine_tune:
        _, archive_weight_file = models.load_archive(args.fine_tune)

        archive_weights = torch.load(archive_weight_file, map_location=lambda storage, loc: storage)['model']

        # Avoiding initializing from archive some weights
        no_ft_regex = trainer_params.pop("no_ft", ())

        finetune_weights = {}
        random_weights = []
        for name, parameter in archive_weights.items():
            if any(re.search(regex, name) for regex in no_ft_regex):
                random_weights.append(name)
                continue
            finetune_weights[name] = parameter

        logger.info(f'Loading the following weights from archive {args.fine_tune}:')
        logger.info(','.join(finetune_weights.keys()))
        logger.info(f'The following weights are at random:')
        logger.info(','.join(random_weights))

        model.load_state_dict(finetune_weights, strict=False)

    # Freezing some parameters
    freeze_params(model, trainer_params.pop('no_grad', ()))

    trainer = Trainer(args.serialization_dir,
                      trainer_params,
                      model,
                      loss,
                      alphabet,
                      local_rank=args.local_rank,
                      world_size=args.world_size,
                      sync_bn=args.sync_bn,
                      opt_level=args.opt_level,
                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                      loss_scale=args.loss_scale)

    try:
        trainer.run(loaders['train'], val_loader=loaders.get('val'), num_epochs=trainer_params['num_epochs'])
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(args.serialization_dir, models.DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('param_path', type=str, help='path to parameter file describing the model to be trained')

    parser.add_argument('-s',
                        '--serialization-dir',
                        '--output-dir',
                        '--output_dir',
                        default=os.environ.get('PT_OUTPUT_DIR', os.path.join('results')),
                        type=str,
                        help='directory in which to save the model and its logs')

    parser.add_argument('--prev-output-dir', '--prevModelDir', default=os.environ.get('PT_PREV_OUTPUT_DIR', None))

    parser.add_argument('-r', '--reset', '--restart', action='store_true', default=False, help='Restart training')

    parser.add_argument('--fine-tune', default=None, type=str, help='Path to an archive model to fine-tune from')

    parser.add_argument('--first-epoch',
                        default='asc',
                        choices=['asc', 'desc', None],
                        help='First epoch data loader behavior')

    parser.add_argument('-o',
                        '--overrides',
                        type=str,
                        default="{}",
                        help='a JSON structure used to override the experiment configuration')

    # Deterministic
    parser.add_argument('--deterministic', '-d', action='store_true', default=False)

    # Distributed training
    parser.add_argument('--backend', default='nccl', type=str)
    parser.add_argument('--init-method', default='env://', type=str)
    parser.add_argument('--local-rank', '--local_rank', '--gpu', default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False)

    # F16 training
    parser.add_argument('--opt-level', default='O0', type=str, choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument('--keep-batchnorm-fp32', default=None, action='store_true')
    parser.add_argument('--loss-scale', type=str, default=None)

    parser.add_argument('--verbosity', '-v', action='count', default=0)

    args = parser.parse_args()

    # Initialize sox
    torchaudio.initialize_sox()

    args.world_size = 1

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(args.local_rank)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.distributed.init_process_group(backend=args.backend, init_method=args.init_method)
        args.world_size = torch.distributed.get_world_size()
    else:
        args.local_rank = 0

    train_model_from_args(args)

    torchaudio.shutdown_sox()
