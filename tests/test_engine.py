import os
import tempfile

import pytest
import torch

from asr.common.params import Params
from asr.engine import Trainer
from asr.nn import SequenceWise
from asr import losses
from asr.data import Alphabet
from asr.exceptions import ConfigurationError
from asr.metrics import Metrics
from asr.engine import hvd
import glob
import copy
import logging
import time

from unittest.mock import patch, MagicMock, Mock, call

import horovod.torch as hvd


@pytest.fixture(scope="session", autouse=True)
def hvd_wrapper(request):
    hvd.init()
    yield
    hvd.shutdown()


@pytest.fixture()
def params():
    yaml_str = """
    lr_scheduler:
        type: exponential
        gamma: 0.9

    batch_size: 16

    # allow list of regex
    # no_grad:

    optimizer:
        type: sgd
        lr: 0.0003
        momentum: 0.90
        nesterov: True
        weight_decay: 0.00001

    lr_scheduler:
        type: exponential
        gamma: 0.9090909090909

    clip_grad_norm: 400
    clip_grad_value: null

    metrics: [cer, wer]
    """

    with tempfile.NamedTemporaryFile('w', delete=False) as temp:
        temp.write(yaml_str)

    params = Params.load(temp.name)
    os.unlink(temp.name)

    return params


@pytest.fixture()
def model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = SequenceWise(torch.nn.Sequential(torch.nn.Linear(161,
                                                                       5)))

        def foward(self, x, x_length):
            return self.fc(x), x_length

    return MockModel()


@pytest.fixture()
def trainer(tmpdir_factory, params, model):
    serialization_dir = tmpdir_factory.mktemp('serialization_dir')
    loss = losses.CTCLoss(backend='pytorch')
    alphabet = Alphabet('-abc ', blank_index=0)

    trainer = Trainer(str(serialization_dir),
                      params,
                      model,
                      loss,
                      alphabet,
                      device='cpu')

    return trainer


@pytest.fixture()
def loader_factory():
    class Loader():
        def __init__(self):
            self.dataset = list([
                torch.rand((2, 4)),
                torch.tensor([0, 1, 1, 2]),
                torch.tensor([2, 2]),
                torch.tensor([2, 2])
            ] for _ in range(3))

            self.batch_sampler = MagicMock()
            self.sampler = None

        def __iter__(self):
            return iter(self.dataset)

        def __len__(self):
            return len(self.dataset)

    return Loader


def test_constructor(tmpdir, model, params):
    serialization_dir = tmpdir / 'serialization_dir'
    loss = losses.CTCLoss(backend='pytorch')
    alphabet = Alphabet('-abc ', blank_index=0)

    missing_params = copy.deepcopy(params)
    del missing_params['optimizer']
    with pytest.raises(ConfigurationError) as excinfo:
        Trainer(serialization_dir,
                copy.deepcopy(missing_params),
                model,
                loss,
                alphabet,
                device='cpu')

    assert "key 'optimizer' is required" in str(excinfo.value)

    allowed_missing_params = copy.deepcopy(params)
    del allowed_missing_params['lr_scheduler']
    trainer = Trainer(serialization_dir,
                      copy.deepcopy(allowed_missing_params),
                      model,
                      loss,
                      alphabet,
                      device='cpu')
    assert trainer.lr_scheduler == None

    trainer = Trainer(serialization_dir,
                      copy.deepcopy(params),
                      model,
                      loss,
                      alphabet,
                      device='cpu')

    for phase in ['train', 'val']:
        assert isinstance(trainer.metrics[phase], Metrics)

    assert trainer.monitor == 'loss'
    assert trainer.clip_grad_norm == 400
    assert trainer.clip_grad_value == None
    assert trainer.start_epoch == 0
    assert trainer.start_iteration == 0
    assert trainer.iterations_per_epoch == None
    assert trainer.start_time == 0

    params['monitor'] = 'cer'
    trainer = Trainer(serialization_dir,
                      copy.deepcopy(params),
                      model,
                      loss,
                      alphabet,
                      device='cpu')

    assert trainer.monitor == 'cer'


def test_save_checkpoint(tmpdir, model, params):
    serialization_dir = tmpdir / 'serialization_dir'
    loss = losses.CTCLoss(backend='pytorch')
    alphabet = Alphabet('-abc ', blank_index=0)

    trainer = Trainer(serialization_dir,
                      params,
                      model,
                      loss,
                      alphabet,
                      device='cpu')

    # Mocking variables
    trainer.iterations_per_epoch = 10
    trainer.epoch = 0
    trainer.start_time = time.time()

    # should do nothing
    trainer.save_checkpoint(iteration=5, is_train=True)

    assert not (serialization_dir / 'models').exists()

    # should save, end of epoch
    trainer.model.state_dict = Mock(return_value='model dict')
    trainer.optimizer.state_dict = Mock(return_value='optimizer dict')
    trainer.score = Mock(return_value=float('inf'))

    trainer.save_checkpoint(iteration=9, is_train=True)

    assert (serialization_dir / 'models').exists()

    assert len(glob.glob(str(serialization_dir / 'models' / '*'))) == 1

    assert (serialization_dir / 'models' / 'model-10.pth').exists()

    ckpt_dict = torch.load(str(serialization_dir / 'models' / 'model-10.pth'))

    empty_metric_state_dict = {
        'val': 0,
        'avg': 0,
        'count': 0,
        'sum': 0,
        'history': []
    }

    empty_metrics_state_dict = [{
        'type': 'asr.metrics.Loss',
        'state_dict': empty_metric_state_dict
    }, {
        'type': 'asr.metrics.CER',
        'state_dict': empty_metric_state_dict
    }, {
        'type': 'asr.metrics.WER',
        'state_dict': empty_metric_state_dict
    }]

    expected_ckpt_dict = {
        'model': 'model dict',
        'epoch': 1,
        'epoch_iterations': 0,
        'iterations_per_epoch': 10,
        'best_monitor': float('inf'),
        'metrics': {
            'train': empty_metrics_state_dict,
            'val': empty_metrics_state_dict
        },
        'optimizer': 'optimizer dict'
    }

    assert ckpt_dict == expected_ckpt_dict

    # save best
    trainer.epoch = 1
    trainer.score = Mock(return_value=2.0)

    trainer.save_checkpoint(is_train=False)

    assert len(glob.glob(str(serialization_dir / 'models' / '*'))) == 3

    assert (serialization_dir / 'models' / 'model-20.pth').exists()
    assert (serialization_dir / 'models' / 'best-model.pth').exists()

    ckpt_dict = torch.load(str(serialization_dir / 'models' / 'model-20.pth'))

    best_ckpt_dict = torch.load(
        str(serialization_dir / 'models' / 'best-model.pth'))

    assert ckpt_dict == best_ckpt_dict

    expected_ckpt_dict['best_monitor'] = 2.0
    expected_ckpt_dict['epoch'] = 2
    assert best_ckpt_dict == expected_ckpt_dict

    # save by time
    trainer.start_time = time.time() - 60 * 10
    trainer.epoch = 2

    trainer.save_checkpoint(iteration=8, is_train=True)

    assert len(glob.glob(str(serialization_dir / 'models' / '*'))) == 4

    assert (serialization_dir / 'models' / 'model-29.pth').exists()

    ckpt_dict = torch.load(str(serialization_dir / 'models' / 'model-29.pth'))

    expected_ckpt_dict['best_monitor'] = 2.0
    expected_ckpt_dict['epoch'] = 2
    expected_ckpt_dict['epoch_iterations'] = 9
    assert ckpt_dict == expected_ckpt_dict


def test_load_checkpoint(tmpdir, caplog, model, params):
    serialization_dir = (tmpdir / 'serialization_dir').mkdir()
    loss = losses.CTCLoss(backend='pytorch')
    alphabet = Alphabet('-abc ', blank_index=0)

    trainer = Trainer(serialization_dir,
                      params,
                      model,
                      loss,
                      alphabet,
                      device='cpu')

    # no checkpoint
    trainer.load_checkpoint()
    assert not ('Last model checkpoint found' in caplog.record_tuples[-1][2])

    (serialization_dir / 'models').mkdir()

    # no checkpoint
    trainer.load_checkpoint()
    assert not ('Last model checkpoint found' in caplog.record_tuples[-1][2])

    # mocking calls

    ckpt_dict = {
        'model': 'model-mock',
        'optimizer': 'optimizer-mock',
        'best_monitor': 2.0,
        'metrics': {
            'train': 'train-metric-mock',
            'val': 'val-metric-mock'
        },
        'epoch': 1,
        'epoch_iterations': 8,
        'iterations_per_epoch': 10
    }

    trainer.model.load_state_dict = Mock()
    trainer.optimizer.load_state_dict = Mock()
    for split in ['train', 'val']:
        trainer.metrics[split].load_state_dict = Mock()
    torch.load = Mock(return_value=ckpt_dict)

    # find biggest iterations
    f1 = (serialization_dir / 'models').join('model-20.pth')
    f1.write('')
    f2 = (serialization_dir / 'models').join('model-25.pth')
    f2.write('')

    trainer.load_checkpoint()

    assert 'Last model checkpoint found' in caplog.record_tuples[-1][2]

    torch.load.assert_called_with(str(f2), map_location='cpu')
    trainer.model.load_state_dict.assert_called_with('model-mock')
    trainer.optimizer.load_state_dict.assert_called_with('optimizer-mock')

    for split in ['train', 'val']:
        trainer.metrics[split].load_state_dict.assert_called_with(
            f'{split}-metric-mock')

    assert trainer.best_monitor == 2.0
    assert trainer.start_epoch == 1
    assert trainer.start_iteration == 8
    assert trainer.iterations_per_epoch == 10


def test_restore(trainer, caplog):

    trainer.start_epoch = 1
    trainer.start_iteration = 8

    hvd.broadcast = MagicMock(side_effect=lambda x, **kwargs: x + 1)
    hvd.broadcast_parameters = MagicMock()
    hvd.broadcast_optimizer_state = MagicMock()
    trainer.model.state_dict = MagicMock(return_value='mock')
    trainer.load_checkpoint = MagicMock()
    trainer.lr_scheduler.step = MagicMock()

    trainer.restore()

    trainer.load_checkpoint.assert_called_once()

    calls = [
        call(torch.tensor(1), root_rank=0, name='start_epoch'),
        call(torch.tensor(8), root_rank=0, name='start_iteration')
    ]

    assert trainer.start_epoch == 2
    assert trainer.start_iteration == 9

    hvd.broadcast.assert_has_calls(calls, any_order=True)

    hvd.broadcast_parameters.assert_called_once()
    hvd.broadcast_parameters.assert_called_with('mock', root_rank=0)

    hvd.broadcast_optimizer_state.assert_called_once()
    hvd.broadcast_optimizer_state.assert_called_with(trainer.optimizer,
                                                     root_rank=0)

    trainer.lr_scheduler.step.assert_called_once()
    trainer.lr_scheduler.step.assert_called_with(1)

    assert 'Restoring from epoch 2 and it. 9.' == caplog.record_tuples[-1][2]


def test_score(trainer, model, params):
    metrics = trainer.metrics['train']
    metrics.update([
        torch.tensor([0.0, 1.0]),
        torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0]]]).log(),
        torch.tensor([1, 2, 2]),
        torch.tensor([3]),
        torch.tensor([3])
    ])

    expected_cer = torch.tensor(1 / 3).float()

    trainer.monitor = 'cer'
    assert trainer.score(metrics) == expected_cer

    trainer.monitor = 'wer'
    assert trainer.score(metrics) == 1.0

    trainer.monitor = '-cer'
    assert trainer.score(metrics) == -expected_cer

    trainer.monitor = 'invalid'
    with pytest.raises(KeyError):
        trainer.score(metrics)


def test_run(trainer, loader_factory, caplog):
    train_loader = loader_factory()

    def run_epoch(loader, metrics, is_train):
        time.sleep(0.01)
        if is_train:
            return 100.0
        return 200.0

    trainer._run_epoch = Mock(side_effect=run_epoch)
    trainer.lr_scheduler.step = Mock()
    trainer.save_checkpoint = Mock()

    trainer.run(train_loader, num_epochs=2)

    assert trainer.start_epoch == 0
    assert trainer.epoch == 1
    assert trainer.num_epochs == 2
    assert trainer.iterations_per_epoch == len(train_loader)

    trainer.save_checkpoint.call_count == 2

    trainer._run_epoch.call_count == 2
    train_call = call(train_loader,
                      metrics=trainer.metrics['train'],
                      is_train=True)
    trainer._run_epoch.assert_has_calls([train_call])

    trainer.lr_scheduler.step.call_count == 2

    assert caplog.record_tuples[-1][1] == logging.INFO
    assert 'Train:' in caplog.record_tuples[-1][2]
    assert 'Data Time: 100.00' in caplog.record_tuples[-1][2]

    assert caplog.record_tuples[-2][1] == logging.INFO
    assert 'Train:' in caplog.record_tuples[-2][2]
    assert 'Data Time: 100.00' in caplog.record_tuples[-2][2]

    # using also validation loader

    val_loader = loader_factory()

    trainer._run_epoch.reset_mock()
    trainer.lr_scheduler.step.reset_mock()
    trainer.save_checkpoint.reset_mock()
    trainer.start_epoch = 0

    trainer.run(train_loader, val_loader=val_loader, num_epochs=2)

    assert trainer.epoch == 1
    assert trainer.num_epochs == 2
    assert trainer.iterations_per_epoch == len(train_loader)

    trainer.save_checkpoint.call_count == 2

    trainer._run_epoch.call_count == 4
    train_call = call(train_loader,
                      metrics=trainer.metrics['train'],
                      is_train=True)
    val_call = call(val_loader, metrics=trainer.metrics['val'], is_train=False)

    trainer._run_epoch.assert_has_calls(
        [train_call, val_call, train_call, val_call])

    trainer.lr_scheduler.step.call_count == 2

    assert caplog.record_tuples[-2][1] == logging.INFO
    assert 'Train:' in caplog.record_tuples[-2][2]
    assert 'Data Time: 100.00' in caplog.record_tuples[-2][2]

    assert caplog.record_tuples[-1][1] == logging.INFO
    assert 'Val:' in caplog.record_tuples[-1][2]
    assert 'Data Time: 200.00' in caplog.record_tuples[-1][2]


def test_run_and_restore(trainer, loader_factory):
    pass


def test_run_epoch(trainer, loader_factory):

    old_train = trainer.model.train
    trainer.model.train = MagicMock(side_effect=lambda x: old_train(x))

    metric = trainer.metrics['train'][0]

    metric_reset = metric.reset
    metric.reset = MagicMock(side_effect=lambda: metric_reset())

    # Verify if model is set to train or eval
    loader = loader_factory()
    loader.dataset = []

    data_time = trainer._run_epoch(loader, metric, is_train=True)

    trainer.model.train.assert_called_once()
    trainer.model.train.assert_called_with(True)

    loader.batch_sampler.set_epoch.assert_called_once()
    loader.batch_sampler.set_epoch.assert_called_with(0)

    metric.reset.assert_not_called()

    assert data_time == 0

    loader.batch_sampler.reset_mock()
    trainer.model.train.reset_mock()

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    trainer.model.train.assert_called_once()
    trainer.model.train.assert_called_with(False)

    loader.batch_sampler.assert_not_called()
    metric.reset.assert_not_called()

    assert data_time == 0

    trainer.epoch = 1

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    assert trainer.model.train.call_count == 2

    loader.batch_sampler.assert_not_called()
    metric.reset.assert_called_once()

    metric.reset.reset_mock()

    trainer.epoch = 1
    trainer.start_iteration = 1

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    metric.reset.assert_not_called()

    # normal loader
    loader = loader_factory()
    trainer.num_epochs = 2

    # mocking for eval
    def preprocess_batch(batch, device):
        return batch

    trainer.preprocess_batch = MagicMock(side_effect=preprocess_batch)
    trainer.model = MagicMock(side_effect=lambda *args: args)
    trainer.loss = MagicMock(
        return_value=torch.tensor([1.0, 2.0], requires_grad=True))

    metric_update = metric.update
    metric.update = MagicMock(side_effect=lambda *args: metric_update(*args))

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    assert trainer.preprocess_batch.call_count == 3
    assert trainer.model.call_count == 3
    assert trainer.loss.call_count == 3

    loss_value = torch.tensor([1.0, 2.0], requires_grad=True)

    # print(metric.update.call_args_list)
    for d, c in zip(loader.dataset, metric.update.mock_calls):
        _, args, kwargs = c
        assert len(args) == 1
        assert len(kwargs) == 0
        assert torch.all(args[0][0] == loss_value)
        assert torch.all(args[0][1] == d[0])
        assert torch.all(args[0][2] == d[1])
        assert torch.all(args[0][3] == d[2])
        assert torch.all(args[0][4] == d[3])

    # invalid loss
    loss_value = torch.tensor([.0, .0], requires_grad=True)
    metric.update.reset_mock()
    trainer.loss = MagicMock(
        return_value=torch.tensor([float('inf'), 2.0], requires_grad=True))

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    for d, c in zip(loader.dataset, metric.update.mock_calls):
        _, args, kwargs = c
        assert torch.all(args[0][0] == loss_value)

    metric.update.reset_mock()
    trainer.loss = MagicMock(
        return_value=torch.tensor([float('nan'), 2.0], requires_grad=True))

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    for d, c in zip(loader.dataset, metric.update.mock_calls):
        _, args, kwargs = c
        assert torch.all(args[0][0] == loss_value)

    metric.update.reset_mock()
    trainer.loss = MagicMock(
        return_value=torch.tensor([float('-inf'), 2.0], requires_grad=True))

    data_time = trainer._run_epoch(loader, metric, is_train=False)

    for d, c in zip(loader.dataset, metric.update.mock_calls):
        _, args, kwargs = c
        assert torch.all(args[0][0] == loss_value)

    # TODO: train = True


# TODO: test with deterministic setting with load from different checkpoints generate the same result