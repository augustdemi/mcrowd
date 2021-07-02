# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .functional import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, x_mask, memory=None, memory_mask=None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers: # 6 layers
            if memory is not None:
                x = layer(x, x_mask, memory, memory_mask)  # x = bs, 12, 512 (in inference: bs, 1, 512) // memory = bs, 7, 512
            else:
                x = layer(x, x_mask)
        return self.norm(x)
