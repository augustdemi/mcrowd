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
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, bias=False)
        self.fc1 = nn.Linear(32 * 4 * 4 + 2, fc_hidden_dim, bias=False)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim, bias=False)

    def forward(self, vel, map, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 6)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        x = F.relu(self.conv1(map)) # 14
        x = F.relu(self.conv2(x))  # 12
        if self.drop_out > 0:
            x = F.dropout(x,
                          p=self.drop_out,
                          training=train)

        x = F.relu(self.conv3(x))  # 10
        x = self.pool(F.relu(self.conv4(x))) # 8->4
        x = x.view(-1, 32 * 4 * 4)
        x = torch.cat((x, vel), -1)
        x = F.relu(self.fc1(x))
        obst_feat = self.fc2(x)
        if self.drop_out > 0:
            obst_feat = F.dropout(obst_feat,
                                p=self.drop_out,
                                training=train)
        return obst_feat



class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(self, fc_hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_hidden_dim, bias=False)
        self.fc2 = nn.Linear(fc_hidden_dim, 32 * 4 * 4 + 2, bias=False)
        self.upsample1 = nn.Upsample(22)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(16, 4, 3, stride=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(4, 1, 3, stride=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(1, 1, 3, stride=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 3, stride=1, bias=False)

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
        v = x[:,-2:]
        x = x[:,:-2].view(-1, 32, 4, 4)
        y = nn.Upsample(10)(self.deconv1(F.relu(x))) # 6
        y = self.deconv2(F.relu(y)) #12
        y = self.deconv3(F.relu(y)) #14
        y = self.deconv4(F.relu(y)) #16
        return F.sigmoid(y), v

