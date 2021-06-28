# -*- coding: utf-8 -*-
# date: 2018-11-30 16:49
import torch.nn as nn
from torch.nn.functional import relu
import torch


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(relu(self.w_1(x))))



class ConcatPointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model, d_latent, d_ff, dropout=0.1):
        super(ConcatPointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model+d_latent, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, latents):
        x = torch.cat((x, latents.repeat((1, x.shape[1], 1))), dim=2)
        return self.w_2(self.dropout(relu(self.w_1(x))))
