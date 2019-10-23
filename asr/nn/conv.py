import torch
from torch import nn


class MaskLayer(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that
        the results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (B x * x T)

        Args:
            seq_module: The sequential module containing the conv stack.
        """
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        Args:
            x: The input of size Bx*xT
            lengths: The actual length of each sequence in the batch
        Returns
            Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            lengths = self._output_lengths(module, lengths)

            indexes = lengths.view([-1] + [1] * (x.dim() - 1)) - 1
            mask = torch.arange(x.shape[-1],
                                device=x.device).expand_as(x) > indexes
            x = x.masked_fill(mask, 0)

        return x, lengths

    def _output_lengths(self, module, lengths):
        if not isinstance(module, nn.modules.conv.Conv2d):
            return lengths

        return ((lengths + 2 * module.padding[1] - module.dilation[1] *
                 (module.kernel_size[1] - 1) - 1) / module.stride[1] + 1).type(
                     lengths.dtype)
