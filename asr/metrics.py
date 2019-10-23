import torch

from asr.decoders import GreedyCTCDecoder
from collections.abc import Sequence

from asr.utils import klass_fullname
from asr.common import Params
import copy
import logging

import torch.distributed as dist
import os

logger = logging.getLogger(__name__)


class Metrics(Sequence):
    def __init__(self, metrics):
        if not isinstance(metrics, (list, set)):
            raise ValueError('Metrics must be a list of Metric')

        for metric in metrics:
            if not isinstance(metric, Metric):
                raise ValueError(f'{metric} is not a Metric.')

        self.metrics = metrics
        self.metrics_by_name = {
            m.name.lower(): i
            for i, m in enumerate(metrics)
        }

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.metrics[self.metrics_by_name[index]]
        return self.metrics[index]

    def __len__(self):
        return len(self.metrics)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, output):
        for metric in self.metrics:
            metric.update(output)

    @property
    def count(self):
        counts = [metric.count for metric in self.metrics]
        if not all(c == counts[0] for c in counts):
            raise Exception('Not all counts are equal')

        return counts[0]

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return f'Metrics([{", ".join([repr(m) for m in self.metrics])}])'

    def as_dict(self):
        metrics_dict = {}
        for m in self.metrics:
            metrics_dict.update(m.as_dict())

        return metrics_dict

    def to_str(self, val=True, avg=True):
        return ', '.join(m.to_str(val=val, avg=avg) for m in self.metrics)

    def state_dict(self):
        return [
            dict([('state_dict', m.state_dict()), ('type', klass_fullname(m))])
            for m in self.metrics
        ]

    def load_state_dict(self, state_dict):
        # for s, m in
        for i, (d, m) in enumerate(zip(state_dict, self.metrics)):
            if d['type'] != klass_fullname(m):
                raise ValueError(f'Invalid type in metrics[{i}]. Expected '
                                 f'{d["type"]} found {klass_fullname(m)}.')

            m.load_state_dict(d['state_dict'])


class Metric:
    def __init__(self, name, output_transform=lambda x: x, fmt='.6f',
                 **kwargs):
        self.name = name
        self.output_transform = output_transform
        self.fmt = fmt
        self.is_distributed = False

        self.val = 0.
        self.sum = 0.
        self.count = 0
        self.avg = 0.
        self.history = []

    def reset(self):
        self.history.append(self.avg)

        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def state_dict(self):
        return {
            'history': self.history,
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(copy.deepcopy(state_dict))

    @torch.no_grad()
    def update(self, output):
        self._update(self.output_transform(output))

    def _update(self, output):
        output = output.detach()

        val = output.sum(dim=0)
        if output.dim() == 0:
            n = torch.tensor(1, device=output.device)
        else:
            n = torch.as_tensor(output.shape[0], device=output.device)

        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            dist.all_reduce(val)
            dist.all_reduce(n)

        self.sum += val.item()
        self.count += n.item()
        self.val = val.item() / n.float().item()
        self.avg = self.sum / float(self.count)

    def as_dict(self):
        return {
            self.name: ('{val:' + self.fmt + '} ({avg:' + self.fmt +
                        '})').format(**self.__dict__)
        }

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.to_str()

    def to_str(self, val=True, avg=True):
        fmtstr = ['{name}']

        if val:
            fmtstr += ['{val:' + self.fmt + '}']
        if avg:
            avgstr = '{}'
            if val:
                avgstr = '({})'
            avgstr = avgstr.format('{avg:' + self.fmt + '}')

            fmtstr += [avgstr]

        fmtstr = ' '.join(fmtstr)
        return fmtstr.format(**self.__dict__)


class Loss(Metric):
    def __init__(self, name='Loss', output_transform=lambda x: x[0], **kwargs):
        super().__init__(name, output_transform)


class WER(Metric):
    def __init__(self,
                 name='WER',
                 output_transform=lambda x: x[1:],
                 fmt='.02%',
                 **kwargs):
        alphabet = kwargs.pop('alphabet', None)
        super().__init__(name, output_transform, fmt=fmt, **kwargs)

        if alphabet is None:
            raise ValueError('Arg. `alphabet` is required.')

        self.decoder = GreedyCTCDecoder(alphabet)

    def _update(self, output):
        output, target, output_lengths, target_lengths = output

        transcripts = self.decoder.decode(output, output_lengths)
        references = self.decoder.tensor2str(target, target_lengths)

        super()._update(
            torch.tensor([
                self.decoder.wer(t, r) /
                float(len(r.split())) if len(r.split()) else 1
                for t, r in zip(transcripts[0], references)
            ],
                         device=output.device))


class CER(Metric):
    def __init__(self,
                 name='CER',
                 output_transform=lambda x: x[1:],
                 fmt='.02%',
                 **kwargs):
        alphabet = kwargs.pop('alphabet', None)
        super().__init__(name, output_transform, fmt=fmt, **kwargs)

        if alphabet is None:
            raise ValueError('Arg. `alphabet` is required.')

        self.decoder = GreedyCTCDecoder(alphabet)

    def _update(self, output):
        output, target, output_lengths, target_lengths = output

        transcripts = self.decoder.decode(output, output_lengths)

        references = self.decoder.tensor2str(target, target_lengths)

        super()._update(
            torch.tensor([
                self.decoder.cer(t, r) / float(len(r)) if len(r) else 1
                for t, r in zip(transcripts[0], references)
            ],
                         device=output.device))


metric_names = {
    'cer': CER,
    'loss': Loss,
    'wer': WER,
}


def by_name(name):
    try:
        return metric_names[name]
    except KeyError:
        raise ValueError(
            f'Metric name {name} not found. '
            f'Metrics available: [{",".join(metric_names.keys())}]')


def from_params(params, **extra):
    params = copy.copy(params)

    def _from_params(params, **extras):
        klass_name = params.pop('type') if isinstance(params,
                                                      (dict,
                                                       Params)) else params
        if isinstance(params, str):
            params = {}

        klass = by_name(klass_name)
        params = {**params, **extra}

        logger.info(
            f'Instantiating class `{klass_fullname(klass)}` with params {params}'
        )

        return klass(**params)

    metrics = []
    for metric_params in params:
        metrics.append(_from_params(metric_params, **extra))

    return Metrics(metrics)