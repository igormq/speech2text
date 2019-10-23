import argparse
from collections import Counter
import re
from asr.data import Alphabet

from tqdm import tqdm
import kenlm
import math


def tokenizer(args):
    if args.unit == 'word':
        raise ValueError('Not implemented yet')

    tokens = Alphabet.from_file(args.tokens)

    if '<space>' in tokens:
        raise ValueError(f'Reserved token `<space>` found in {str(args.tokens)}')

    lines = args.infile.readlines()
    for line in tqdm(lines, unit='line'):
        l = ' '.join(list(line.strip().replace(' ', '@')))
        args.outfile.write(l + '\n')


def vocab(args):
    print('Counting words... (may take a while)')
    words = re.findall(r'\w+', args.infile.read().lower())
    counter = Counter(words)
    print(f'Found {len(counter)} words')
    vocab = [w for w, c in counter.most_common(args.max_size) if c >= args.min_count]
    print(f'Vocabulary of {len(vocab)} words after pruning (word_count >= {args.min_count})')
    args.outfile.write('\n'.join(vocab))


def score(args):
    model = kenlm.Model(args.model_path)

    is_character_based = True
    for w in model.vocab:
        if w not in ('<s>', '</s>', '<unk>') and len(w) > 1:
            print(w)
            is_character_based = False
            break

    print(f'is character based? {is_character_based}. order {model.order}')

    score, num_words, num_chars = 0.0, 0, 0
    for line in args.infile:
        sentence = line.strip()

        if is_character_based:
            score += model.score(' '.join(list(sentence.replace(' ', '@'))))
        else:
            score += model.score(sentence)  # log scale
        num_words += len(sentence.split()) + 1  # EOS
        num_chars += len(list(sentence)) + 1  # eos

    if not is_character_based:
        ppl = 10**(-score / num_words)
        print(f'PPL: {ppl:.4f}')
        return

    bpc = -score / num_chars / math.log10(2)
    ppl = 2**(bpc * (num_chars / num_words))
    print(f'BPC: {bpc:.4} PPL: {ppl:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    tokenizer_parser = subparsers.add_parser('tokenizer')
    tokenizer_parser.add_argument('infile', type=argparse.FileType('r', encoding='utf8'))
    tokenizer_parser.add_argument('outfile', nargs='?', type=argparse.FileType('w', encoding='utf8'))

    tokenizer_parser.add_argument('--vocab', default=None, type=argparse.FileType('r', encoding='utf8'))

    tokenizer_parser.add_argument('--unit', default='word', choices=['char', 'word'])
    tokenizer_parser.add_argument('--tokens', required=True, default=str)

    tokenizer_parser.set_defaults(func=tokenizer)

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.add_argument('infile', type=argparse.FileType('r', encoding='utf8'))
    vocab_parser.add_argument('outfile', nargs='?', type=argparse.FileType('w', encoding='utf8'))
    vocab_parser.add_argument('--min-count', default=3)
    vocab_parser.add_argument('--max-size', default=None)
    vocab_parser.set_defaults(func=vocab)

    score_parser = subparsers.add_parser('score')
    score_parser.add_argument('model_path', type=str)
    score_parser.add_argument('infile', nargs='?', type=argparse.FileType('r', encoding='utf8'))
    score_parser.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)
