import json
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Alphabet:
    """ Alphabet of characters.
    Args:
        tokens (iterable): alphabet characters
        blank_index (int): blank index in alphabet provided
    """

    def __init__(self, tokens, blank_index):
        self.tokens = tokens
        self.blank_index = blank_index

        self._idx2str = {i: v for i, v in enumerate(tokens)}
        self._str2idx = {v: i for i, v in enumerate(tokens)}

    @staticmethod
    def word_tokenize(sentence):
        return sentence.split()

    @staticmethod
    def char_tokenize(sentence, remove_space=False):
        if remove_space:
            sentence = re.sub(r'\s+', '', sentence)
        return list(sentence)

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            json_data = json.load(f)

        return cls(json_data['tokens'],
                   json_data['tokens'].index(json_data['blank_token']))

    def save_to_files(self, directory):
        """
        Persist this Alphabet to files so it can be reloaded later.
        Each namespace corresponds to one file.

        Params:
            directory : ``str``
                The directory where we save the serialized alphabet.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logger.warn("alphabet serialization directory %s is not empty",
                        directory)

        json_data = {
            'tokens': self.tokens,
            'blank_token': self.tokens[self.blank_index]
        }

        with open(os.path.join(directory, 'tokens'), 'w',
                  encoding='utf8') as f:
            json.dump(json_data, f, indent=4)

    @classmethod
    def from_params(cls, params):
        if 'from_file' in params:
            return cls.from_file(params['from_file'])
        else:
            return cls(params['tokens'],
                       params['tokens'].index(params['blank_token']))

    def idx2str(self, sequence):
        return ''.join(self._idx2str[i.item(
        )] if isinstance(i, torch.Tensor) else self._idx2str[i]
                       for i in sequence)

    def str2idx(self, sentence):
        return [self._str2idx[s] for s in sentence]

    def __repr__(self):
        return ''.join(self.tokens)

    def __contains__(self, x):
        if isinstance(x, str):
            return x in self.tokens

        if isinstance(x, list) and isinstance(x[0], int):
            return x in list(self._idx2str.keys())

        raise ValueError('Invalid input type.')

    def __len__(self):
        return len(self.tokens)
