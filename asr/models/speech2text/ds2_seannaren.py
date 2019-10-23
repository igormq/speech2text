import torch
import torch.nn.functional as F

from asr.nn import DeepBatchRNN, LookaheadConvolution, SequenceWise
from asr.nn.conv import MaskLayer


class DeepSpeech2SeanNaren(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 rnn_type=torch.nn.GRU,
                 rnn_hidden_size=800,
                 num_rnn_layers=5,
                 window_size=320,
                 bidirectional=True,
                 log_softmax=False,
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
        self._log_softmax = log_softmax

        self.convs = MaskLayer(
            torch.nn.Sequential(
                torch.nn.Conv2d(1,
                                32,
                                kernel_size=(41, 11),
                                stride=(2, 2),
                                padding=(20, 5)), torch.nn.BatchNorm2d(32),
                torch.nn.Hardtanh(0, 20, inplace=True),
                torch.nn.Conv2d(
                    32,
                    32,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                ), torch.nn.BatchNorm2d(32),
                torch.nn.Hardtanh(0, 20, inplace=True)))

        self.rnns = DeepBatchRNN(self._rnn_input_size(),
                                 self._rnn_hidden_size,
                                 rnn_type=self._rnn_type,
                                 bidirectional=self._bidirectional,
                                 num_layers=self._num_rnn_layers,
                                 sum_directions=True,
                                 bias=True)

        self.lookahead = torch.nn.Sequential(
            # consider adding batch norm?
            LookaheadConvolution(self._rnn_hidden_size, context=self._context),
            torch.nn.Hardtanh(
                0, 20, inplace=True)) if not self._bidirectional else None

        fully_connected = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self._rnn_hidden_size),
            torch.nn.Linear(self._rnn_hidden_size, num_classes, bias=False))
        self.fc = torch.nn.Sequential(SequenceWise(fully_connected))

    def _rnn_input_size(self):
        # FFT output
        rnn_input_size = self.input_dim

        out_channels = 1
        for m in self.convs.modules():

            if not isinstance(m, torch.torch.nn.modules.conv.Conv2d):
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
