import torch.nn as nn


class BatchRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type=nn.LSTM,
                 bidirectional=False,
                 batch_norm=True,
                 bias=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            bias=bias)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = x._replace(data=self.batch_norm(x.data))
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x._replace(data=x.data[:, :self.hidden_size] +
                           x.data[:, self.hidden_size:])  # sum bidirectional
        return x


class DeepBatchRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type=nn.LSTM,
                 bidirectional=False,
                 num_layers=1,
                 batch_norm=True,
                 sum_directions=True,
                 bias=False,
                 **kwargs):
        super(DeepBatchRNN, self).__init__()
        self._bidirectional = bidirectional
        rnns = []
        rnn = BatchRNN(input_size=input_size,
                       hidden_size=hidden_size,
                       rnn_type=rnn_type,
                       bidirectional=bidirectional,
                       batch_norm=False,
                       bias=bias)
        rnns.append(rnn)
        for x in range(num_layers - 1):
            rnn = BatchRNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           rnn_type=rnn_type,
                           bidirectional=bidirectional,
                           batch_norm=batch_norm,
                           bias=bias)
            rnns.append(rnn)
        self.rnns = nn.Sequential(*rnns)

        self.sum_directions = sum_directions

    def flatten_parameters(self):
        for x in range(len(self.rnns)):
            self.rnns[x].flatten_parameters()

    def forward(self, x, lengths):
        max_seq_length = x.shape[1]
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x = self.rnns(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=max_seq_length, batch_first=True)
        return x
