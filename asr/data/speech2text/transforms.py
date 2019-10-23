import logging
import os
import re

import torch
import torchaudio
from num2words import num2words
from unidecode import unidecode

from asr.data.alphabet import Alphabet

logger = logging.getLogger(__name__)

__all__ = ['ToSpectrogram', 'ToTensor', 'ToLabel']


class ToSpectrogram:
    """Create a spectrogram from a raw audio signal

    Args:
        win_len (int): window size, often called the fft size as well
        hop_len (int, optional): length of hop_len between STFT windows. default: ws // 2
        n_fft (int, optional): number of fft bins. default: ws // 2 + 1
        normalize (bool): Apply standard mean and deviation normalization to spectrogram
        window (torch windowing function or str): default: torch.hann_window
        window_params (dict, optional): arguments for window function
        librosa_compat (bool): if `True`the stft will be librosa compatible
    """

    def __init__(self,
                 win_len=320,
                 hop_len=160,
                 n_fft=None,
                 normalize=True,
                 normalize_window=False,
                 window=torch.hamming_window,
                 eps=torch.tensor(torch.finfo(torch.float).eps,
                                  dtype=torch.get_default_dtype()),
                 win_kwargs={},
                 spec_kwargs={
                     'center': True,
                     'normalized': False,
                     'onesided': True
                 },
                 device=None):

        if isinstance(window, torch.Tensor):
            self.window = window
        elif isinstance(window, str):
            self.window = getattr(torch.functional,
                                  '{}_window'.format(window))(win_len,
                                                              **win_kwargs)
        else:
            self.window = window(win_len, **win_kwargs)
            self.window = torch.as_tensor(self.window, dtype=torch.float)

        self.win_len = win_len
        self.hop_len = hop_len if hop_len is not None else win_len // 2
        self.n_fft = n_fft or self.win_len
        self.normalize = normalize
        self.normalize_window = normalize_window
        self.eps = eps
        self.device = device
        self.spec_kwargs = spec_kwargs

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (N, )
        Returns:
            spectogram (Tensor): num_hops x n_fft/ 2 + 1
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 1

        if self.device is not None:
            x = x.to(self.device)

        self.window = self.window.to(x.device)

        S = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window,
                       **self.spec_kwargs)

        if self.normalize_window:
            S /= self.window.pow(2).sum().sqrt()

        # Get magnitude of the complex signal
        S = S.pow(2).sum(-1).sqrt()

        S = torch.max(S, self.eps).log1p()

        if self.normalize:
            S = (S - S.mean()) / (S.std() + self.eps)

        return S.transpose(0, 1)

    def __repr__(self):
        return ('{}(win_len={}, hop_len={}, n_fft={}, normalize={})').format(
            self.__class__.__name__, self.win_len, self.hop_len, self.n_fft,
            self.normalize)


class ToTensor:
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.

    Args:
        sample_rate (int): the expected sample rate
    Returns:
        the torch tensor sound.
    """

    def __init__(self,
                 sample_rate=16000,
                 augment=False,
                 tempo_range=(0.85, 1.15),
                 gain_range=(-6, 8)):
        self.sample_rate = sample_rate
        self.augment = augment
        self.tempo_range = tempo_range
        self.gain_range = gain_range

        self.sox_effects = torchaudio.sox_effects.SoxEffectsChain()

    def __call__(self, filepath):

        data, sample_rate = self._load_with_sox(filepath)
        if self.sample_rate != sample_rate:
            raise ValueError(
                f'Sample rate mismatched. Expected {self.sample_rate}, found {sample_rate}.'
            )

        return data.squeeze()

    def _load_with_sox(self, filepath):
        self.sox_effects.clear_chain()

        self.sox_effects.append_effect_to_chain("rate", [self.sample_rate])
        self.sox_effects.append_effect_to_chain("channels", ["1"])

        if self.augment:
            low_tempo, high_tempo = self.tempo_range
            tempo_value = (high_tempo -
                           low_tempo) * torch.rand(1).item() + low_tempo

            low_gain, high_gain = self.gain_range
            gain_value = (high_gain -
                          low_gain) * torch.rand(1).item() + low_gain

            self.sox_effects.append_effect_to_chain("tempo",
                                                    ["-s", tempo_value])
            self.sox_effects.append_effect_to_chain("gain", [gain_value])

        self.sox_effects.set_input_file(filepath)
        return self.sox_effects.sox_build_flow_effects()

    def __repr__(self):
        return ('{}(sample_rate={})').format(self.__class__.__name__,
                                             self.sample_rate)


class ToLabel:
    """ Parse transcript file or list of utterances into vocab given a dictionary

    Args:
        vocab(str or list): list of vocab or string with the vocab or path to a json containing
            the vocab
        to_lower (bool): if `True`, characters are parsed to lowercase before transforming
        convert_number_to_word (bool): if `True`, converts number to words
        locale (str): locale used to convert number to strings
        remove_accents (bool): if `True`, all accents are removed prior transform
        filter_if_not_in_vocab (bool): if `True` will filter characters not present in the vocab
    """

    GET_NUMBERS_PATTERN = r'-?\d+(?:\.|,)?\d*'

    def __init__(self,
                 alphabet,
                 to_lower=True,
                 convert_number_to_words=False,
                 locale=None,
                 remove_accents=False,
                 filter_if_not_in_alphabet=False,
                 dtype=torch.int):

        self._alphabet = alphabet if isinstance(
            alphabet, Alphabet) else Alphabet.from_file(alphabet)
        self._locale = locale
        self._to_lower = to_lower
        self._dtype = dtype
        self._convert_number_to_words = convert_number_to_words
        self._remove_accents = remove_accents
        self._filter_if_not_in_alphabet = filter_if_not_in_alphabet

    def __call__(self, x):
        """
        Args:
            x (bytes, str, or file path): a file path containing the desired
                transcript to be converted or the string to be converted
        Returns:
            ndarray of size (size,)
        """
        if isinstance(x, str):
            if os.path.isfile(x):
                with open(x, 'r', encoding='utf8') as f:
                    try:
                        transcript = f.readline().strip()
                    except UnicodeDecodeError as e:
                        logger.error(f'Error decoding {x}')
                        raise e
            else:
                transcript = x
        elif isinstance(x, bytes):
            transcript = x.decode('utf8')
        else:
            raise ValueError('input type was not recognized')

        if self._to_lower:
            transcript = transcript.lower()

        if self._convert_number_to_words:
            transcript = self._parse_numbers(transcript)

        if self._remove_accents:
            transcript = unidecode(transcript)

        if self._alphabet is None:
            return transcript

        if self._filter_if_not_in_alphabet:
            transcript = ''.join(
                filter(lambda x: x in self._alphabet, transcript))

        transcript = self._alphabet.str2idx(transcript)

        return torch.as_tensor(transcript, dtype=self._dtype)

    def _parse_numbers(self, transcript):
        return re.sub(
            ToLabel.GET_NUMBERS_PATTERN, lambda x: num2words(
                x.group().replace(',', '.'), lang=self._locale), transcript)

    def __repr__(self):
        repr_ = '{}(path={}, to_lower={}, '.format(self.__class__.__name__,
                                                   self._alphabet,
                                                   self._to_lower)
        repr_ += 'convert_number_to_words={}, locale={}, '.format(
            self._convert_number_to_words, self._locale)
        repr_ += 'remove_accents={}, dtype={})'.format(self._remove_accents,
                                                       self._dtype)

        return repr_
