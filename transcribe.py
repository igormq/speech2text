import argparse

import torch

from codes.decoder import GreedyDecoder
from codes.utils.model_utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser.add_argument(
        '--model-path',
        default='models/deepspeech_final.pth',
        help='Path to model file created by training')
    parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
    parser.add_argument(
        '--audio-path', metavar='AUDIO', help='Audio file to predict on', required=True)
    parser.add_argument(
        '--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
    parser.add_argument(
        '--verbose', action="store_true", help="print out decoded output and error of each sample")
    no_decoder_args = parser.add_argument_group("No Decoder Options",
                                                "Configuration options for when no decoder is "
                                                "specified")
    beam_args = parser.add_argument_group("Beam Decode Options",
                                          "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
    beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')
    beam_args.add_argument(
        '--lm-path',
        default=None,
        type=str,
        help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)'
    )
    beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
    beam_args.add_argument(
        '--beta', default=1, type=float, help='Language model word bonus (all words)')
    beam_args.add_argument(
        '--cutoff-top-n',
        default=40,
        type=int,
        help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
        'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument(
        '--cutoff-prob',
        default=1.0,
        type=float,
        help='Cutoff probability in pruning,default 1.0, no pruning.')
    beam_args.add_argument(
        '--lm-workers', default=1, type=int, help='Number of LM processes to use')
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, transforms, target_transforms = load_model(args.model_path)
    model.eval()

    label_encoder = target_transforms.label_encoder

    device = 'cpu'
    if args.cuda:
        device = 'cuda'
        model.to(device)

    if args.decoder == "beam":
        from codes.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(
            label_encoder,
            lm_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            cutoff_top_n=args.cutoff_top_n,
            cutoff_prob=args.cutoff_prob,
            beam_width=args.beam_width,
            num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(label_encoder)
    else:
        raise ValueError('Unknown decoder')

    inputs = transforms(args.audio_path).contiguous()
    inputs = inputs.view(1, inputs.shape[0], inputs.shape[1])
    inputs = inputs.to(device)
    out = model(inputs)  # NxTxH

    seq_length = out.shape[1]
    sizes = torch.tensor(seq_length)

    decoded_output, _, = decoder.decode(out, sizes)

    transcript = decoded_output[0][0]
    print("Hyp: {}".format(transcript.encode('utf8').lower()))
