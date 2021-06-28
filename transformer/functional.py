# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import math
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
from torch.nn.functional import softmax


def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = numpy.triu(numpy.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # bs, # head, # query, # key = 70,8, 12, 7 ==> neighbors 여러개(k)의 src를 동시에 주면 70, 8,12,7k의 shape 될것. mask는 k=1에 대해서는 mask=True, k>1인 이웃이면 mask=False이되, value를 0.5로 설정
    if mask is not None: # mask = # 70, 1, 1, 7 for encoder, [70, 1, 12, 12] for decoder
        scores = scores.masked_fill_(mask == 0, value=-1e9) # mask 1인 src의 mask에서는 모두 False로 전달될것. 따라서 score가 그대로 유지됨.  mask=True가 있는 trg mask에서는 mask==0이 False를 밷어서 score가 그대로. mask=False인 값에서는 score가 매우 적어질것.
    p_attn = softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn #(bs, 8, # query, 64), (bs, 8, # query, # key)
