""" Code to fune-tuning the language models parameters
"""
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
from asr.decoders import BeamCTCDecoder, GreedyCTCDecoder
from asr.models import load_archive, CONFIG_NAME
from asr.common import Params

logger = logging.getLogger('asr')


def tune_from_args(args):
    # Disable some of the more verbose logging statements
    logging.getLogger('asr.common.params').disabled = True
    logging.getLogger('asr.common.registrable').disabled = True

    # Load from archive
    _, weights_file = load_archive(args.serialization_dir, args.overrides, args.weights_file)

    params = Params.load(os.path.join(args.serialization_dir, CONFIG_NAME), args.overrides)

    prepare_environment(params)

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    dataset_params = params.pop('val_dataset', params.get('dataset_reader'))

    logger.info("Reading evaluation data from %s", args.input_file)
    dataset_params['manifest_filepath'] = args.input_file
    dataset = datasets.from_params(dataset_params)

    if os.path.exists(os.path.join(args.serialization_dir, "alphabet")):
        alphabet = Alphabet.from_file(os.path.join(args.serialization_dir, "alphabet", "tokens" ))
    else:
        alphabet = Alphabet.from_params(params.pop("alphabet", {}))

    logits_dir = os.path.join(args.serialization_dir, 'logits')
    os.makedirs(logits_dir, exist_ok=True)

    basename = os.path.splitext(os.path.split(args.input_file)[1])[0]
    logits_file = os.path.join(logits_dir, basename + '.pth')

    if not os.path.exists(logits_file):
        model = models.from_params(alphabet=alphabet, params=params.pop('model'))
        model.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage)['model'])
        model.eval()

        decoder = GreedyCTCDecoder(alphabet)

        loader_params = params.pop("val_data_loader", params.get("data_loader"))
        batch_sampler = samplers.BucketingSampler(dataset, batch_size=args.batch_size)
        loader = loaders.from_params(loader_params, dataset=dataset, batch_sampler=batch_sampler)

        logger.info(f'Logits file `{logits_file}` not found. Generating...')

        with torch.no_grad():
            model.to(args.device)

            logits = []
            for batch in tqdm.tqdm(loader):
                sample, target, sample_lengths, target_lengths = batch
                sample = sample.to(args.device)
                sample_lengths = sample_lengths.to(args.device)

                output, output_lengths = model(sample, sample_lengths)

                output = output.to('cpu')

                references = decoder.tensor2str(target, target_lengths)

                logits.extend((o[:l, ...], r) for o, l, r in zip(output.to('cpu'), output_lengths, references))

                del sample, sample_lengths, output

            torch.save(logits, logits_file)

        del model

    tune_dir = os.path.join(args.serialization_dir, 'tune')
    os.makedirs(tune_dir, exist_ok=True)

    params_grid = list(
        product(torch.linspace(args.alpha_from, args.alpha_to, args.alpha_steps),
                torch.linspace(args.beta_from, args.beta_to, args.beta_steps)))

    print('Scheduling {} jobs for alphas=linspace({}, {}, {}), betas=linspace({}, {}, {})'.format(
        len(params_grid), args.alpha_from, args.alpha_to, args.alpha_steps, args.beta_from, args.beta_to,
        args.beta_steps))

    # start worker processes
    logger.info(f"Using {args.num_workers} processes and {args.lm_workers} for each CTCDecoder.")
    extract_start = default_timer()

    p = Pool(
        args.num_workers, init,
        [logits_file, alphabet, args.lm_path, args.cutoff_top_n, args.cutoff_prob, args.beam_width, args.lm_workers])

    scores = []
    best_wer = float('inf')
    with tqdm.tqdm(p.imap(tune_step, params_grid), total=len(params_grid), desc='Grid search') as pbar:
        for params in pbar:
            alpha, beta, wer, cer = params
            scores.append([alpha, beta, wer, cer])

            if wer < best_wer:
                best_wer = wer
                pbar.set_postfix(alpha=alpha, beta=beta, wer=wer, cer=cer)

    logger.info(f"Finished {len(params_grid)} processes in {default_timer() - extract_start:.1f}s")

    df = pd.DataFrame(scores, columns=['alpha', 'beta', 'wer', 'cer'])
    df.to_csv(os.path.join(tune_dir, basename + '.csv'), index=False)


def init(logits_file, alphabet, lm_path, cutoff_top_n, cutoff_prob, beam_width, workers):
    global saved_outputs
    global decoder

    saved_outputs = torch.load(logits_file)
    decoder = BeamCTCDecoder(alphabet,
                             lm_path=lm_path,
                             cutoff_top_n=cutoff_top_n,
                             cutoff_prob=cutoff_prob,
                             beam_width=beam_width,
                             num_processes=workers)


def tune_step(params):
    alpha, beta = params
    alpha = alpha.item()
    beta = beta.item()

    global decoder
    global saved_outputs

    decoder.reset_params(alpha, beta)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    for i, (output, reference) in enumerate(saved_outputs):

        transcript = decoder.decode(torch.as_tensor(output, dtype=torch.float32).unsqueeze(0))[0][0]

        total_wer += decoder.wer(transcript, reference)
        total_cer += decoder.cer(transcript, reference)
        num_tokens += float(len(reference.split()))
        num_chars += float(len(reference))

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    return alpha, beta, wer, cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune language model given acoustic model and dataset')

    parser.add_argument('serialization_dir', type=str, help='path to an archived trained model')
    parser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')
    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--weights-file', type=str, help='a path that overrides which weights file to use')
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='batch size')
    parser.add_argument('--device', '-d', default='cuda', type=str, help='device to use')
    parser.add_argument('-o',
                        '--overrides',
                        type=str,
                        default="{} ",
                        help='a JSON structure used to override the experiment paramsuration')
    parser.add_argument('--num-workers', default=cpu_count() - 1, type=int, help='Number of parallel decodes to run')

    beam_args = parser.add_argument_group("Beam Decode Options",
                                          "paramsurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--lm-path', default=None, type=str, help='Language model to use')
    beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')

    beam_args.add_argument('--alpha-from', default=0.0, type=float, help='Language model weight start tuning')
    beam_args.add_argument('--alpha-to', default=3.0, type=float, help='Language model weight end tuning')
    beam_args.add_argument('--beta-from',
                           default=0.0,
                           type=float,
                           help='Language model word bonus (all words) start tuning')
    beam_args.add_argument('--beta-to',
                           default=0.5,
                           type=float,
                           help='Language model word bonus (all words) end tuning')
    beam_args.add_argument('--alpha-steps', default=45, type=int, help='Number of alpha candidates for tuning')
    beam_args.add_argument('--beta-steps', default=8, type=int, help='Number of beta candidates for tuning')

    beam_args.add_argument('--cutoff-top-n',
                           default=40,
                           type=int,
                           help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                           'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument('--cutoff-prob',
                           default=1.0,
                           type=float,
                           help='Cutoff probability in pruning,default 1.0, no pruning.')

    beam_args.add_argument('--lm-workers', default=1, type=int)

    args = parser.parse_args()
    tune_from_args(args)
