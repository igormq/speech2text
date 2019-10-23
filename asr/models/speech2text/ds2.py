import torch
import torch.nn.functional as F
import torch.nn as nn

from asr.nn import DeepBatchRNN, LookaheadConvolution, SequenceWise
from asr.nn.conv import MaskLayer

DEFAULT_CONV_LAYERS = [{
    'kernel_size': [41, 11],
    'stride': [2, 2],
    'out_channels': 32,
    'padding': [20, 5]
}, {
    'kernel_size': [21, 11],
    'stride': [2, 1],
    'out_channels': 32,
    'padding': [10, 5]
}]


class DeepSpeech2(torch.nn.Module):
    def __init__(
            self,
            conv_layers=DEFAULT_CONV_LAYERS,
            rnn_type=nn.GRU,
            rnn_hidden_size=800,
            num_rnn_layers=5,
            window_size=320,
            bidirectional=True,
            dropout=0.5,
            num_classes=29,
            hidden_size=1600,
            log_softmax=False,  # Baidu warp-ctc does not require
            context=20):
        super().__init__()

        if isinstance(rnn_type, str):
            rnn_type = getattr(torch.nn, rnn_type.upper())

        self.input_dim = int(window_size / 2 + 1)
        self._rnn_type = rnn_type
        self._rnn_hidden_size = rnn_hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._window_size = window_size
        self._bidirectional = bidirectional
        self._context = context
        self._dropout = dropout
        self._hidden_size = hidden_size
        self._num_classes = num_classes
        self._log_softmax = log_softmax

        convs = []
        in_channels = 1
        for conv_params in conv_layers:
            out_channels = conv_params['out_channels']

            convs.append(nn.Conv2d(in_channels=in_channels, **conv_params))
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        self.convs = MaskLayer(nn.Sequential(*convs))

        self.rnns = DeepBatchRNN(self._rnn_input_size(), self._rnn_hidden_size,
                                 self._rnn_type, self._bidirectional,
                                 self._num_rnn_layers)

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            LookaheadConvolution(self._rnn_hidden_size, context=self._context),
            nn.ReLU(inplace=True)) if not self._bidirectional else None

        self.fc = SequenceWise(
            nn.Sequential(nn.Linear(self._rnn_hidden_size, self._hidden_size),
                          nn.Hardtanh(0, 20, inplace=True),
                          nn.Dropout(self._dropout),
                          nn.Linear(self._hidden_size, self._num_classes)))

    def _rnn_input_size(self):
        # FFT output
        rnn_input_size = self.input_dim

        out_channels = 1
        for m in self.convs.modules():

            if not isinstance(m, torch.nn.modules.conv.Conv2d):
                continue

            rnn_input_size = int(
                ((rnn_input_size + 2 * m.padding[0] - m.dilation[0] *
                  (m.kernel_size[0] - 1) - 1) / m.stride[0] + 1))

            out_channels = m.out_channels

        # Channels and height collapse
        return rnn_input_size * out_channels

    def forward(self, x, lengths):
        """
        Args:
            x: tensor of size (B, T, N):
                B: batch size
                T: max seq. length
                N: num_features
        """
        # Add channel dimension. B x T x N -> B x 1 x T x N
        # Transpose T by N. B x 1 x T x N -> B x 1 x N x T
        x = x.unsqueeze(1).transpose_(2, 3)

        x, output_lengths = self.convs(x, lengths)

        # Collapse feature dimension
        B, C, D, T = x.shape

        # B x C x N x T -> B x T x C * N
        x = x.view(B, C * D, T).permute(0, 2, 1)

        x = self.rnns(x, output_lengths)

        # no need for lookahead layer in bidirectional
        if not self._bidirectional:
            x = self.lookahead(x)

        x = self.fc(x.contiguous())

        if self._log_softmax or not self.training:
            x = F.log_softmax(x, dim=-1)

        return x, output_lengths
