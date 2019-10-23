import numpy as np
import pytest

from asr.utils import error_rate


def test_wer_1():
    ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
    hyp = 'i GOT IT TO the FULLEST i LOVE TO portable FROM OF STORES last night'
    word_error_rate = error_rate.wer(ref, hyp)

    assert np.allclose(word_error_rate, 0.769230769231)


def test_wer_2():
    ref = 'as any in england i would say said gamewell proudly that is in his day'
    hyp = 'as any in england i would say said came well proudly that is in his day'
    word_error_rate = error_rate.wer(ref, hyp)
    assert np.allclose(word_error_rate, 0.1333333)


def test_wer_3():
    ref = 'the lieutenant governor lilburn w boggs afterward governor was a pronounced mormon hater and throughout the'\
          ' period of the troubles he manifested sympathy with the persecutors'
    hyp = 'the lieutenant governor little bit how bags afterward governor was a pronounced warman hater and '\
          'throughout the period of th troubles he manifests sympathy with the persecutors'
    word_error_rate = error_rate.wer(ref, hyp)
    assert np.allclose(word_error_rate, 0.2692307692)


def test_wer_4():
    ref = 'the wood flamed up splendidly under the large brewing copper and it sighed so deeply'
    hyp = 'the wood flame do splendidly under the large brewing copper and its side so deeply'
    word_error_rate = error_rate.wer(ref, hyp)
    assert np.allclose(word_error_rate, 0.2666666667)


def test_wer_5():
    ref = 'all the morning they trudged up the mountain path and at noon unc and ojo sat on a fallen tree trunk and ' \
          'ate the last of the bread which the old munchkin had placed in his pocket'
    hyp = 'all the morning they trudged up the mountain path and at noon unc in ojo sat on a fallen tree trunk and ate'\
          ' the last of the bread which the old munchkin had placed in his pocket'
    word_error_rate = error_rate.wer(ref, hyp)
    assert np.allclose(word_error_rate, 0.027027027)


def test_wer_6():
    ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
    word_error_rate = error_rate.wer(ref, ref)
    assert word_error_rate == 0.0


def test_wer_7():
    ref = ' '
    hyp = 'Hypothesis sentence'
    with pytest.raises(ValueError):
        error_rate.wer(ref, hyp)


def test_cer_1():
    ref = 'werewolf'
    hyp = 'weae  wolf'
    char_error_rate = error_rate.cer(ref, hyp)
    assert np.allclose(char_error_rate, 0.25)


def test_cer_2():
    ref = 'werewolf'
    hyp = 'weae  wolf'
    char_error_rate = error_rate.cer(ref, hyp, remove_space=True)
    assert np.allclose(char_error_rate, 0.125)


def test_cer_3():
    ref = 'were wolf'
    hyp = 'were  wolf'
    char_error_rate = error_rate.cer(ref, hyp)
    assert np.allclose(char_error_rate, 0.0)


def test_cer_4():
    ref = 'werewolf'
    char_error_rate = error_rate.cer(ref, ref)
    assert char_error_rate == 0.0


def test_cer_5():
    ref = ''
    hyp = 'Hypothesis'
    with pytest.raises(ValueError):
        error_rate.cer(ref, hyp)
