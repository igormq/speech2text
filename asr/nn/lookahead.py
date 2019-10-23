import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LookaheadConvolution(nn.Module):
    """ Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks

    Wang et al 2016

    Input shape is a ndarray with shape (sequence, batch, feature)
    """

    def __init__(self, num_features, context):
        super(LookaheadConvolution, self).__init__()

        assert context > 0, "Context should be greater than 0"

        self.num_features = num_features
        self.context = context

        self.weight = nn.Parameter(torch.Tensor(num_features, context + 1))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.uniform_(-stdv, stdv)

    def forward(self, x):
        seq_len = x.shape[0]

        x = F.pad(x, (0, 0, 0, 0, 0, self.context))

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        # TxLxNxH - sequence, context, batch, feature
        x = [x[i:i + self.context + 1] for i in range(seq_len)]
        x = torch.stack(x)

        # TxNxHxL - sequence, batch, feature, context
        x = x.permute(0, 2, 3, 1)

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'num_features=' + str(
            self.num_features) + ', context=' + str(self.context) + ')'
