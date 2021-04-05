import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from gmm2d import GMM2D
from torch.distributions.laplace import Laplace
import random

###############################################################################

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# -----------------------------------------------------------------------------#

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


###############################################################################
# -----------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim, drop_out):
        super(Encoder, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(1, 4, 7, stride=3, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 4, 5, stride=2, bias=False)
        self.fc1 = nn.Linear(4 * 7 * 7 + 2, fc_hidden_dim, bias=False)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim, bias=False)

    def forward(self, state, map, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 6)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        x = self.pool(F.relu(self.conv1(map))) # 198 ->64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 4 * 7 * 7)
        x = torch.cat((x, state[:, 2:4]), -1)
        x = F.relu(self.fc1(x))
        obst_feat = self.fc2(x)
        obst_feat = F.dropout(obst_feat,
                            p=self.drop_out,
                            training=train)

        return obst_feat



class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(self, fc_hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim, bias=False)
        self.fc2 = nn.Linear(fc_hidden_dim, 4 * 7 * 7 + 2, bias=False)
        self.upsample1 = nn.Upsample(14)
        self.deconv1 = nn.ConvTranspose2d(4, 4, 6, stride=2, bias=False)
        self.upsample2 = nn.Upsample(64)
        self.deconv2 = nn.ConvTranspose2d(4, 1, 9, stride=3, bias=False)

    def forward(self, obst_feat):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - enc_h_feat: hidden feature from the encoder
        - z: sample from the posterior/prior dist.
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        x= self.fc1(obst_feat)
        x= self.fc2(F.relu(x))
        x = x[:, :-2].view(-1, 4, 7, 7)
        x = self.upsample1(F.relu(x))
        x = self.deconv1(x) # 32, 32
        x = self.upsample2(F.relu(x)) #64
        map = self.deconv2(x) #198
        return F.sigmoid(map)

