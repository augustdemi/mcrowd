# -*- coding: utf-8 -*-
# date: 2018-11-30 16:35
import torch.nn as nn

from .functional import clones, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, dist_weight=False):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.dist_weight = dist_weight

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k where h = num of heads: bs, 7, 512 -> bs, 8, 7, 64
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, dist_weight = self.dist_weight) #(bs, 8, # query, 64), (bs, 8, # query, # key)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) #(bs, #query, 512)
        return self.linears[-1](x)
