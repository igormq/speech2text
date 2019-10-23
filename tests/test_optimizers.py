import pytest
from asr import optimizers
import torch
import logging
from asr.common import Params

params_content = """
optimizer:
    type: sgd
    lr: .1
    param_groups:
        - [[^param*], {'lr': 0.01}]
"""


@pytest.fixture(scope='session')
def param_file(tmpdir_factory):
    f = tmpdir_factory.mktemp('temp').join('params.yml')
    f.write(params_content)
    return f


@pytest.mark.parametrize('name, klass',
                         [("adam", torch.optim.Adam),
                          ("sparse_adam", torch.optim.SparseAdam),
                          ("adagrad", torch.optim.Adagrad),
                          ("adadelta", torch.optim.Adadelta),
                          ("sgd", torch.optim.SGD),
                          ("rmsprop", torch.optim.RMSprop),
                          ("rprop", torch.optim.Rprop),
                          ("adamax", torch.optim.Adamax),
                          ("averaged_sgd", torch.optim.ASGD)])
def test_by_name(name, klass):
    k = optimizers.by_name(name)
    assert k == klass


def test_invalid_by_name():
    with pytest.raises(ValueError) as excinfo:
        optimizers.by_name('invalid-name')

    assert 'Optimizer name invalid-name not found' in str(excinfo.value)


def test_from_params():
    model_params = [
        ('param1', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('param2', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('diffparam', torch.nn.Parameter(torch.rand(1).requires_grad_()))
    ]

    params = 'sgd'
    with pytest.raises(TypeError) as excinfo:
        optimizer = optimizers.from_params(params)

    assert "missing 1 required positional argument: 'model_params'" in str(
        excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        optimizer = optimizers.from_params(params, model_params)

    assert "value of required optimization parameter lr" in str(excinfo.value)

    optimizer = optimizers.from_params(params, model_params, lr=.1)
    isinstance(optimizer, torch.optim.SGD)

    params = {'type': 'sgd'}
    optimizer = optimizers.from_params(params, model_params, lr=.1)
    assert isinstance(optimizer, torch.optim.SGD)

    params = {'type': 'sgd', 'lr': .1}
    optimizer = optimizers.from_params(params, model_params)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == .1

    params = {'type': 'sgd', 'lr': .1}
    optimizer = optimizers.from_params(params, model_params, lr=.2)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == .2

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['params'] == [
        param for name, param in model_params
    ]
    assert optimizer.param_groups[0]['lr'] == .2
    assert optimizer.param_groups[0]['momentum'] == 0
    assert optimizer.param_groups[0]['dampening'] == 0
    assert optimizer.param_groups[0]['weight_decay'] == 0
    assert optimizer.param_groups[0]['nesterov'] == False


def test_param_groups(param_file):
    model_params = [
        ('param1', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('param2', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('diffparam', torch.nn.Parameter(torch.rand(1).requires_grad_()))
    ]

    params = Params.load(param_file)
    optimizer = optimizers.from_params(params.pop('optimizer'), model_params)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == .1

    assert optimizer.param_groups[0]['lr'] == 0.01
    assert torch.allclose(optimizer.param_groups[0]['params'][0],
                          model_params[0][1])
    assert torch.allclose(optimizer.param_groups[0]['params'][1],
                          model_params[1][1])

    assert optimizer.param_groups[1]['lr'] == 0.1
    assert torch.allclose(optimizer.param_groups[1]['params'][0],
                          model_params[2][1])


def test_logging(caplog):
    model_params = [
        ('param1', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('param2', torch.nn.Parameter(torch.rand(1).requires_grad_())),
        ('diffparam', torch.nn.Parameter(torch.rand(1).requires_grad_()))
    ]

    params = {'type': 'sgd', 'lr': .1}

    optimizer = optimizers.from_params(params, model_params)

    assert "Number of trainable parameters: 3" in caplog.record_tuples[0][2]
    assert logging.INFO == caplog.record_tuples[0][1]

    assert "Instantiating class `torch.optim.sgd.SGD` with params {'lr': 0.1}" in caplog.record_tuples[
        1][2]
    assert logging.INFO == caplog.record_tuples[1][1]
