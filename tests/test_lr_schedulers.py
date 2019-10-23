import pytest
from asr import lr_schedulers
import torch
import logging


@pytest.mark.parametrize(
    'name, klass',
    [("step", torch.optim.lr_scheduler.StepLR),
     ("multi_step", torch.optim.lr_scheduler.MultiStepLR),
     ("exponential", torch.optim.lr_scheduler.ExponentialLR),
     ("reduce_on_plateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
     ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR),
     ("cyclic", torch.optim.lr_scheduler.CyclicLR)])
def test_by_name(name, klass):
    k = lr_schedulers.by_name(name)
    assert k == klass


def test_by_name():
    with pytest.raises(ValueError):
        lr_schedulers.by_name('invalid-name')


def test_from_params():
    params = [torch.rand(1).requires_grad_()]
    optimizer = torch.optim.SGD(params, 0.1)

    params = 'step'
    with pytest.raises(TypeError) as excinfo:
        lr_scheduler = lr_schedulers.from_params(params)
        isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)

    assert "missing 1 required positional argument: 'optimizer'" in str(
        excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        lr_scheduler = lr_schedulers.from_params(params, optimizer)

    assert "missing 1 required positional argument: 'step_size'" in str(
        excinfo.value)

    lr_scheduler = lr_schedulers.from_params(params, optimizer, step_size=1)
    isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)

    params = {'type': 'step'}
    lr_scheduler = lr_schedulers.from_params(params, optimizer, step_size=1)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)

    params = {'type': 'step', 'step_size': 1}
    lr_scheduler = lr_schedulers.from_params(params, optimizer)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert lr_scheduler.step_size == 1

    params = {'type': 'step', 'step_size': 1}
    lr_scheduler = lr_schedulers.from_params(params, optimizer, step_size=2)

    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)
    assert lr_scheduler.step_size == 2


def test_logging(caplog):
    params = {'type': 'step', 'step_size': 1}
    optimizer = torch.optim.SGD([torch.rand(1).requires_grad_()], 0.1)

    lr_scheduler = lr_schedulers.from_params(params, optimizer, step_size=2)
    assert "Instantiating class `torch.optim.lr_scheduler.StepLR` with params {'step_size': 2}" in caplog.record_tuples[
        0][2]
    assert logging.INFO == caplog.record_tuples[0][1]
