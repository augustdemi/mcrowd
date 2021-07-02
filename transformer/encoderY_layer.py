# -*- coding: utf-8 -*-
# date: 2018-11-30 15:27
import torch.nn as nn

from .functional import clones
from .sublayer_connection import SublayerConnection


class EncoderYLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(EncoderYLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, x_mask, memory, memory_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, x_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, memory_mask))
        return self.sublayer[2](x, lambda x: self.feed_forward(x))