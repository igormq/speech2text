import argparse
import logging
import os
from itertools import product
from timeit import default_timer

import tqdm
import pandas as pd
import torch
from torch.multiprocessing import Pool, cpu_count

from asr import samplers, models
from asr.utils.exp_utils import prepare_environment
from asr.data import loaders, datasets, Alphabet
from asr.data.speech2text.transforms import ToTensor
from asr.decoders import GreedyCTCDecoder
from asr.models import load_archive, CONFIG_NAME
from asr.common import Params

logger = logging.getLogger('asr')


def evaluate_from_args(args):
    # Disable some of the more verbose logging statements
    logging.getLogger('asr.common.params').disabled = True
    logging.getLogger('asr.common.registrable').disabled = True

    # Load from archive
    _, weights_file = load_archive(args.serialization_dir, args.overrides,
                                   args.weights_file)

    params = Params.load(os.path.join(args.serialization_dir, CONFIG_NAME),
                         args.overrides)

    prepare_environment(params)

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    dataset_params = params.pop('val_dataset', params.get('dataset_reader'))

    logger.info("Reading evaluation data from %s", args.input_file)
    dataset_params['manifest_filepath'] = args.input_file
    dataset = datasets.from_params(dataset_params)

    if os.path.exists(os.path.join(args.serialization_dir, "alphabet")):
        alphabet = Alphabet.from_file(
            os.path.join(args.serialization_dir, "alphabet", "tokens"))
    else:
        alphabet = Alphabet.from_params(params.pop("alphabet", {}))

    logits_dir = os.path.join(args.serialization_dir, 'logits')
    os.makedirs(logits_dir, exist_ok=True)

    basename = os.path.splitext(os.path.split(args.input_file)[1])[0]
    print(basename)
    logits_file = os.path.join(logits_dir, basename + '.pth')

    if not os.path.exists(logits_file):
        model = models.from_params(alphabet=alphabet,
                                   params=params.pop('model'))
        model.load_state_dict(
            torch.load(weights_file,
                       map_location=lambda storage, loc: storage)['model'])
        model.eval()

        decoder = GreedyCTCDecoder(alphabet)

        loader_params = params.pop("val_data_loader",
                                   params.get("data_loader"))
        batch_sampler = samplers.BucketingSampler(dataset,
                                                  batch_size=args.batch_size)
        loader = loaders.from_params(loader_params,
                                     dataset=dataset,
                                     batch_sampler=batch_sampler)

        logger.info(f'Logits file `{logits_file}` not found. Generating...')

        with torch.no_grad():
            model.to(args.device)

            logits = []
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            for batch in tqdm.tqdm(loader):
                sample, target, sample_lengths, target_lengths = batch
                sample = sample.to(args.device)
                sample_lengths = sample_lengths.to(args.device)

                output, output_lengths = model(sample, sample_lengths)

                output = output.to('cpu')

                references = decoder.tensor2str(target, target_lengths)

                transcripts = decoder.decode(output)[0]

                logits.extend(
                    (o[:l, ...], r)
                    for o, l, r in zip(output, output_lengths, references))

                del sample, sample_lengths, output

                for reference, transcript in zip(references, transcripts):
                    total_wer += decoder.wer(transcript, reference)
                    total_cer += decoder.cer(transcript, reference)
                    num_tokens += float(len(reference.split()))
                    num_chars += float(len(reference))

            torch.save(logits, logits_file)

            wer = float(total_wer) / num_tokens
            cer = float(total_cer) / num_chars

            print(f'WER: {wer:.02%}\nCER: {cer:.02%}')

        del model

    else:
        logger.info(f'Logits file `{logits_file}` already generated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tune language model given acoustic model and dataset')

    parser.add_argument('serialization_dir',
                        type=str,
                        help='path to an archived trained model')
    parser.add_argument('input_file',
                        type=str,
                        help='path to the file containing the evaluation data')
    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--weights-file',
                        type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--batch-size',
                        '-b',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--device',
                        '-d',
                        default='cuda',
                        type=str,
                        help='device to use')
    parser.add_argument(
        '-o',
        '--overrides',
        type=str,
        default="{} ",
        help='a JSON structure used to override the experiment paramsuration')

    args = parser.parse_args()
    evaluate_from_args(args)
