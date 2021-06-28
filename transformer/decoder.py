# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch.nn as nn

from .layer_norm import LayerNorm
from .functional import clones


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, latents, src_mask, tgt_mask):
        for layer in self.layers: # 6 layers
            x = layer(x, memory, latents, src_mask, tgt_mask) # x = bs, 12, 512 (in inference: bs, 1, 512) // memory = bs, 7, 512
        return self.norm(x)
