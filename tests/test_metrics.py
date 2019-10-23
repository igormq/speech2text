import pytest
from asr import metrics
from asr.metrics import Metric, Metrics, Loss, WER, CER
import torch
from asr.data import Alphabet
import numpy as np
import logging
import horovod.torch as hvd


@pytest.fixture(scope="session", autouse=True)
def hvd_wrapper(request):
    hvd.init()
    yield
    hvd.shutdown()


@pytest.mark.parametrize('name, klass', [
    ('wer', metrics.WER),
    ('loss', Loss),
    ('cer', CER),
])
def test_by_name(name, klass):
    k = metrics.by_name(name)
    assert k == klass


def test_invalid_by_name():
    with pytest.raises(ValueError) as excinfo:
        metrics.by_name('invalid-name')

    assert 'Metric name invalid-name not found' in str(excinfo.value)

    assert metrics.by_name('wer')


def test_from_params():
    params = ['cer', 'wer']
    alphabet = Alphabet('-abc ', blank_index=0)
    m = metrics.from_params(params, alphabet=alphabet)

    assert isinstance(m, Metrics)
    assert isinstance(m[0], CER)
    assert isinstance(m[1], WER)

    params = [{'type': 'cer'}, 'wer']
    m = metrics.from_params(params, alphabet=alphabet)

    assert isinstance(m, Metrics)
    assert isinstance(m[0], CER)
    assert isinstance(m[1], WER)


def test_metric(caplog):
    metric = Metric('metric')

    assert metric.name == 'metric'
    assert metric.sum == 0
    assert metric.count == 0

    # Reset without any update
    metric.reset()
    assert 'Metric.reset() called but count = 0' in caplog.record_tuples[-1][2]
    assert caplog.record_tuples[-1][1] == logging.WARNING

    values = torch.rand(3)
    metric.update(values)

    assert metric.val == values.mean(0)
    assert metric.sum == values.sum(0)
    assert metric.count == 3

    assert repr(metric) == 'metric'
    assert str(metric) == f'metric {values.mean(0):.6f} ({values.mean(0):.6f})'
    assert metric.as_dict() == {
        'metric': f'{values.mean(0):.6f} ({values.mean(0):.6f})'
    }

    metric.reset()

    assert metric.val == 0
    assert metric.sum == 0
    assert metric.count == 0
    assert metric.avg == 0


def test_metrics():
    invalid_metrics = 'invalid'
    with pytest.raises(ValueError) as excinfo:
        Metrics(invalid_metrics)
    assert 'Metrics must be a list of Metric' in str(excinfo.value)

    invalid_metrics = ['invalid', Metric('valid')]
    with pytest.raises(ValueError) as excinfo:
        Metrics(invalid_metrics)
    assert "invalid is not a Metric" in str(excinfo.value)

    metrics = Metrics([
        Metric('metric1', output_transform=lambda x: x[0]),
        Metric('metric2', output_transform=lambda x: x[1])
    ])

    values0 = torch.rand((3, 3))
    values1 = torch.rand((3, 3))

    for i, (v0, v1) in enumerate(zip(values0, values1)):
        metrics.update([v0, v1])

        assert np.allclose(metrics[0].val, v0.mean(0).item())
        assert metrics[0].count == (i + 1) * 3
        assert np.allclose(metrics[0].sum, values0[:(i + 1)].sum().item())
        assert np.allclose(metrics[0].avg,
                           values0[:(i + 1)].sum() / ((i + 1) * 3))

        assert np.allclose(metrics[1].val, v1.mean(0).item())
        assert metrics[1].count == (i + 1) * 3
        assert np.allclose(metrics[1].sum, values1[:(i + 1)].sum())
        assert np.allclose(metrics[1].avg,
                           values1[:(i + 1)].sum().item() / ((i + 1) * 3))

    assert repr(metrics) == 'Metrics([metric1, metric2])'
    assert str(metrics) == ', '.join(str(metric) for metric in metrics)
    assert metrics.as_dict() == dict(**metrics[0].as_dict(),
                                     **metrics[1].as_dict())

    history = [m.avg for m in metrics]
    metrics.reset()
    for h, metric in zip(history, metrics):
        assert metric.val == 0
        assert metric.sum == 0
        assert metric.count == 0
        assert metric.avg == 0
        assert metric.history[0] == h


def test_metric_state_dict():
    m = Metric('metric1', output_transform=lambda x: torch.tensor(x))

    m.update(1.0)
    m.update(2.0)
    m.reset()
    m.update(3.0)
    m.update(4.0)
    m.update(5.0)

    expected_state_dict = {
        'history': [1.5],
        'val': 5.0,
        'sum': 12.0,
        'count': 3,
        'avg': 12.0 / 3
    }

    assert m.state_dict() == expected_state_dict

    m = Metric('metric1', output_transform=lambda x: torch.tensor(x))

    m.state_dict() == {'history': 0, 'val': 0, 'sum': 0, 'count': 0, 'avg': 0}

    m.load_state_dict(expected_state_dict)

    assert m.state_dict() == expected_state_dict


def test_metrics_state_dict():
    m1 = Metric('metric1', output_transform=lambda x: torch.tensor(x[0]))
    m2 = Metric('metric2', output_transform=lambda x: torch.tensor(x[1]))
    metrics = Metrics([m1, m2])

    metrics.update([1.0, 1.5])
    metrics.update([2.0, 2.5])
    metrics.reset()
    metrics.update([3.0, 3.5])
    metrics.update([4.0, 4.5])
    metrics.update([5.0, 5.5])

    empty_state_dict = {'history': 0, 'val': 0, 'sum': 0, 'count': 0, 'avg': 0}

    expected_state_dict = [{
        'type': 'asr.metrics.Metric',
        'state_dict': {
            'history': [1.5],
            'val': 5.0,
            'sum': 12.0,
            'count': 3,
            'avg': 12.0 / 3
        }
    }, {
        'type': 'asr.metrics.Metric',
        'state_dict': {
            'history': [2],
            'val': 5.5,
            'sum': 13.5,
            'count': 3,
            'avg': 13.5 / 3
        }
    }]

    assert metrics.state_dict() == expected_state_dict

    m1 = Metric('metric1', output_transform=lambda x: torch.tensor(x[0]))
    m2 = Metric('metric2', output_transform=lambda x: torch.tensor(x[1]))
    metrics = Metrics([m1, m2])

    metrics.state_dict() == [{
        'type': 'asr.metrics.Metric',
        'state_dict': empty_state_dict
    }, {
        'type': 'asr.metrics.Metric',
        'state_dict': empty_state_dict
    }]

    metrics.load_state_dict(expected_state_dict)

    assert metrics.state_dict() == expected_state_dict

    expected_state_dict[0]['type'] = 'invalid'

    with pytest.raises(ValueError) as excinfo:
        metrics.load_state_dict(expected_state_dict)

    assert 'Invalid type in metrics[0]' in str(excinfo.value)


def test_loss():
    loss_values = [torch.rand(3), torch.rand(3)]
    not_loss_values = ['not loss', 'also not a loss']
    values = list(zip(loss_values, not_loss_values))

    metric = Loss('loss')

    for v in values:
        metric.update(v)

    assert metric.val == loss_values[-1].mean(0)
    assert metric.sum == loss_values[0].sum() + loss_values[1].sum()
    assert metric.count == 6
    assert np.allclose(
        metric.avg, (loss_values[0].sum() + loss_values[1].sum()).item() / 6.)


def test_wer():
    alphabet = Alphabet('-abc ', blank_index=0)

    with pytest.raises(ValueError) as excinfo:
        metric = WER()

    assert '`alphabet` is required' in str(excinfo.value)

    targets = ['abcc abc', 'abc']
    targets = [torch.tensor(alphabet.str2idx(t)) for t in targets]
    targets = torch.cat(targets)
    targets_size = torch.tensor([8, 3])

    outputs_str = ['aa-b-c-bc   abb-c--a', 'a-bbbb-cc---------aa']
    outputs_ints = [torch.tensor(alphabet.str2idx(o)) for o in outputs_str]

    outputs = torch.zeros(2, 20, 5)
    for i, o in enumerate(outputs_ints):
        outputs[i, ...].scatter_(1, torch.as_tensor([[i] for i in o]), 1)

    outputs_size = torch.tensor([17, 9])

    wers = torch.tensor([1 / 2, 0])

    metric = WER(alphabet=alphabet)

    assert repr(metric) == 'WER'

    metric.update(['loss', outputs, targets, outputs_size, targets_size])

    assert metric.val == wers.mean()
    assert metric.count == 2
    assert str(metric) == f'WER {wers.mean(0):.02%} ({wers.mean(0):.02%})'

    outputs_str = ['aa-b-c-cc   abb-c---', 'a-bbbb-cc-----------']
    outputs_ints = [torch.tensor(alphabet.str2idx(o)) for o in outputs_str]

    outputs = torch.zeros(2, 20, 5)
    for i, o in enumerate(outputs_ints):
        outputs[i, ...].scatter_(1, torch.as_tensor([[i] for i in o]), 1)

    metric.update(['loss', outputs, targets, outputs_size, targets_size])
    assert metric.val == 0
    assert metric.avg == (1 / 2) / 4
    assert metric.count == 4


def test_cer():
    alphabet = Alphabet('-abc ', blank_index=0)

    with pytest.raises(ValueError) as excinfo:
        metric = CER()

    assert '`alphabet` is required' in str(excinfo.value)

    targets = ['abcc abc', 'abc']
    targets = [torch.tensor(alphabet.str2idx(t)) for t in targets]
    targets = torch.cat(targets)
    targets_size = torch.tensor([8, 3])

    outputs_str = ['aa-b-c-bc   abb-c---', 'a-bbbb-cc-----------']
    outputs_ints = [torch.tensor(alphabet.str2idx(o)) for o in outputs_str]

    outputs = torch.zeros(2, 20, 5)
    for i, o in enumerate(outputs_ints):
        outputs[i, ...].scatter_(1, torch.as_tensor([[i] for i in o]), 1)

    outputs_size = torch.tensor([17, 9])

    cers = torch.tensor([1 / 8, 0])

    metric = CER(alphabet=alphabet)

    metric.update(['loss', outputs, targets, outputs_size, targets_size])

    assert metric.val == cers.mean().item()
    assert metric.count == 2
    assert str(metric) == f'CER {cers.mean(0):.02%} ({cers.mean(0):.02%})'

    outputs_str = ['aa-b-c-cc   abb-c---', 'a-bbbb-cc-----------']
    outputs_ints = [torch.tensor(alphabet.str2idx(o)) for o in outputs_str]

    outputs = torch.zeros(2, 20, 5)
    for i, o in enumerate(outputs_ints):
        outputs[i, ...].scatter_(1, torch.as_tensor([[i] for i in o]), 1)

    metric.update(['loss', outputs, targets, outputs_size, targets_size])
    assert metric.val == 0
    assert metric.avg == (1 / 8) / 4
    assert metric.count == 4
