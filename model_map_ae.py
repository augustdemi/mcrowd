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
        self.pool = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(1, 4, 7, stride=3, bias=False)  # 198 ->64 -> 32
        # self.conv2 = nn.Conv2d(4, 4, 5, stride=2, bias=False)
        # self.fc1 = nn.Linear(4 * 7 * 7 + 2, fc_hidden_dim, bias=False)
        # self.fc2 = nn.Linear(fc_hidden_dim, output_dim, bias=False)

        # self.conv1 = nn.Conv2d(1, 4, 7, stride=1, bias=False) #192->96
        # self.conv2 = nn.Conv2d(4, 4, 5, stride=1, bias=False) #92->46
        # self.conv3 = nn.Conv2d(4, 4, 3, stride=1, bias=False) #44->22
        # self.conv4 = nn.Conv2d(4, 4, 3, stride=1, bias=False) #20 ->10

        self.conv1 = nn.Conv2d(1, 4, 7, stride=2, bias=False) #96 -> 48
        self.conv2 = nn.Conv2d(4, 4, 5, stride=1, bias=False) #44 ->22
        self.conv3 = nn.Conv2d(4, 4, 3, stride=1, bias=False) #20->10
        self.conv4 = nn.Conv2d(4, 4, 3, stride=1, bias=False) # 8 -> 4
        # self.fc1 = nn.Linear(4 * 4 * 4 + 2, fc_hidden_dim, bias=False)
        # self.fc2 = nn.Linear(fc_hidden_dim, output_dim, bias=False)

    def forward(self, state, map, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 6)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        x = self.pool(F.relu(self.conv1(map)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # x = x.view(-1, 4 * 4 * 4)
        # x = torch.cat((x, state[:, 2:4]), -1)
        # x = F.relu(self.fc1(x))
        # obst_feat = self.fc1(x)
        # obst_feat = F.dropout(obst_feat,
        #                     p=self.drop_out,
        #                     training=train)

        return x



class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(self, fc_hidden_dim, input_dim):
        super(Decoder, self).__init__()
        # self.fc1 = nn.Linear(input_dim, fc_hidden_dim, bias=False)
        # self.fc2 = nn.Linear(fc_hidden_dim, 4 * 4 * 4 + 2, bias=False)
        self.deconv = nn.Sequential(
            nn.Upsample(8),
            nn.ConvTranspose2d(4, 4, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Upsample(20),
            nn.ConvTranspose2d(4, 4, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Upsample(44),
            nn.ConvTranspose2d(4, 4, 5, stride=1, bias=False),
            nn.ReLU(),
            nn.Upsample(96),
            nn.ConvTranspose2d(4, 1, 8, stride=2, bias=False),
        )


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
        # x= self.fc1(obst_feat)
        # x= self.fc2(obst_feat)
        # x = x[:, :-2].view(-1, 4, 4, 4)
        map = self.deconv(obst_feat)
        return F.sigmoid(map)

