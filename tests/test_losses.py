import pytest
from asr import losses
import torch
import logging

try:
    has_warpctc = True
    from warpctc_pytorch import CTCLoss as warpCTCLoss
except ImportError:
    has_warpctc = False


@pytest.mark.parametrize('name, klass', [("ctc", losses.CTCLoss)])
def test_by_name(name, klass):
    k = losses.by_name(name)
    assert k == klass


def test_invalid_by_name():
    with pytest.raises(ValueError) as excinfo:
        losses.by_name('invalid-name')

    assert 'Loss name invalid-name not found' in str(excinfo.value)

    assert losses.by_name('ctc')


def test_ctc_loss():

    params = 'ctc'

    loss = losses.from_params(params)
    assert isinstance(loss, losses.CTCLoss)

    params = {'type': 'ctc'}
    loss = losses.from_params(params)
    assert isinstance(loss, losses.CTCLoss)

    params = {'type': 'ctc', 'backend': 'pytorch'}
    loss = losses.from_params(params)
    assert isinstance(loss, losses.CTCLoss)
    assert loss.backend == 'pytorch'

    assert isinstance(loss.criterion, torch.nn.CTCLoss)
    assert loss.criterion.reduction == 'none'

    params = {'type': 'ctc', 'backend': 'pytorch', 'blank': 1}
    loss = losses.from_params(params)
    assert loss.blank == 1
    assert loss.criterion.blank == 1

    params = {'type': 'ctc', 'backend': 'pytorch'}
    loss = losses.from_params(params, blank=1)
    assert loss.blank == 1
    assert loss.criterion.blank == 1


@pytest.mark.skipif(not has_warpctc, reason="Requires warpctc_pytorch package")
def test_ctc_baidu_backend():
    params = 'ctc'

    loss = losses.from_params(params)
    assert isinstance(loss, losses.CTCLoss)

    params = {'type': 'ctc'}
    loss = losses.from_params(params)
    assert isinstance(loss, losses.CTCLoss)

    assert loss.backend == 'baidu'
    assert isinstance(loss.criterion, warpCTCLoss)
    assert loss.criterion.size_average == False
    assert loss.criterion.length_average == False

    params = {'type': 'ctc', 'backend': 'baidu'}
    loss = losses.from_params(params)

    assert loss.backend == 'baidu'
    assert isinstance(loss.criterion, warpCTCLoss)

    params = {'type': 'ctc', 'backend': 'baidu', 'blank': 1}
    with pytest.raises(ValueError) as excinfo:
        loss = losses.from_params(params)

    assert 'blank index must be 0' in str(excinfo.value)


def test_logging(caplog):
    params = {'type': 'ctc', 'backend': 'pytorch'}

    loss = losses.from_params(params)
    assert "Instantiating class `asr.losses.CTCLoss` with params {'backend': 'pytorch'}" in caplog.record_tuples[
        0][2]
    assert logging.INFO == caplog.record_tuples[0][1]
