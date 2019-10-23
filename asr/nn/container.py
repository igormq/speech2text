import torch


class SequenceWise(torch.nn.Module):
    """
        Collapses input of dim (sequences x batch_size x num_features) to
        (sequences * batch_size) x num_features, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.

        Args:
            module: Module to apply input to.
    """

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        T, B = x.shape[:2]

        x = x.view(T * B, -1)
        x = self.module(x)
        x = x.view(T, B, -1)

        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr
