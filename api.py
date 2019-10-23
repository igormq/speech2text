import argparse
import json
import os
import re

import pandas as pd
from num2words import num2words
from tqdm import tqdm

from asr.apis import SpeechRecognitionAPI
from asr.data.transforms import ToLabel
from asr.utils.error_rate import cer, wer

DEFAULT_APIS = ['gcp', 'ibm', 'bing']


def transcribe(args):
    data_dir = args.data_dir
    apis = args.apis
    lang = args.lang
    manifest = args.manifest
    file = args.file
    output = args.output

    if manifest:
        audio_files = list(
            map(
                lambda x: (os.path.join(data_dir, x[0]),
                           open(os.path.join(data_dir, x[1]), 'r', encoding='utf8').read().strip()),
                pd.read_csv(manifest).values))

    if file:
        if not os.path.isfile(file):
            raise ValueError('{} not found.'.format(file))
        audio_files = [(file, '')]

    # preload saved results
    if os.path.exists(output):
        with open(output, 'r', encoding='utf8') as f:
            out_data = json.load(f)
    else:
        out_data = {}

    # construct the api objects
    apis = {api: SpeechRecognitionAPI(api, lang=lang) for api in apis}

    for audio_file, transcription in tqdm(audio_files):
        out_data.setdefault(audio_file, {})
        out_data[audio_file]['ref'] = transcription

        for api_name, api in apis.items():
            if api_name in out_data[audio_file]:
                continue

            out_data[audio_file][api_name] = api.recognize(audio_file)

        # saving results
        with open(output, 'w', encoding='utf8') as f:
            json.dump(out_data, f, indent=4, ensure_ascii=False)


def _parse_numbers(x, lang):
    def cast_number(x):
        try:
            return int(x)
        except ValueError:
            return float(x)

    x = re.sub(r'r\$\s+([0-9]+)', r'\1 reais', x)
    x = re.sub(r'([0-9]+)r\$', r'\1 reais', x)

    if lang == 'pt_BR':
        x = x.replace('por cento', 'porcento')
        x = re.sub(r'\+([0-9]+)', r'mais \1', x)
        x = re.sub(r'([0-9]+)\%', r'\1 porcento', x)

    x = re.sub(r'([0-9]+)(?:o|a)s?.?',
               lambda x: num2words(int(x.group(1).replace(',', '.')), ordinal=True, lang=lang), x)

    return re.sub(ToLabel.GET_NUMBERS_PATTERN,
                  lambda x: num2words(cast_number(x.group().replace(',', '.')), lang=lang), x)


def evaluate(args):
    filepath = args.file
    lang = args.lang

    with open(filepath, 'r') as f:
        data = json.load(f)

    total_wer = {}
    total_cer = {}
    num_tokens = {}
    num_chars = {}
    for utterance in tqdm(data):
        ref = _parse_numbers(data[utterance]['ref'].lower(), lang)

        for api in set(data[utterance].keys()).difference(set(['ref'])):
            total_wer.setdefault(api, 0)
            total_cer.setdefault(api, 0)
            num_tokens.setdefault(api, 0)
            num_chars.setdefault(api, 0)

            transcript = _parse_numbers(data[utterance][api].lower(), lang)

            wer_inst = wer(transcript, ref)
            cer_inst = cer(transcript, ref)

            total_wer[api] += wer_inst
            total_cer[api] += cer_inst
            num_tokens[api] += len(ref.split())
            num_chars[api] += len(ref)

    for api in set(total_wer.keys()):
        print("{} - WER: {:.02f}% CER: {:.02f}%".format(
            api, (float(total_wer[api]) / num_tokens[api]) * 100,
            (float(total_cer[api]) / num_chars[api]) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    trans_parser = subparsers.add_parser('transcribe')

    trans_parser.add_argument('--api', default=DEFAULT_APIS, nargs='+')
    trans_parser.add_argument('--lang', default='pt_BR', choices=['pt_BR', 'en'])
    trans_parser.add_argument('--data-dir', default='data', type=str)
    trans_file_group = trans_parser.add_mutually_exclusive_group(required=True)
    trans_file_group.add_argument('--manifest', '-m', type=str)
    trans_file_group.add_argument('--file', '-f', type=str)
    trans_parser.add_argument('--output', default='api.results.json', type=str)
    trans_parser.set_defaults(func=transcribe)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--lang', default='pt_BR', choices=['pt_BR', 'en'])
    eval_parser.add_argument('file', type=str)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)
