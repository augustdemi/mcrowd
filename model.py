import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import imageio
import numpy as np
from torch.distributions.normal import Normal

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage


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


###############################################################################
# -----------------------------------------------------------------------

class EncoderX(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_feat_dim=64, map_mlp_dim=32,
            num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0, device='cpu'
    ):
        super(EncoderX, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.dropout_rnn=dropout_rnn
        self.dropout_mlp=dropout_mlp

        n_state=6

        self.rnn_encoder = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim
        )

        self.sg_rnn_enc = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True)
        self.fc_sg = nn.Linear(4*enc_h_dim, enc_h_dim)


        # self.fc_map = nn.Linear(map_feat_dim + 9, map_mlp_dim)
        self.fc_hidden = nn.Linear(2 * enc_h_dim, mlp_dim)
        self.fc_latent = nn.Linear(mlp_dim, zS_dim*2)


    def forward(self, obs_traj, seq_start_end, sg_state, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """


        # traj rnn enc
        _, (final_encoder_h, _) = self.rnn_encoder(obs_traj) # [8, 656, 16], 두개의 [1, 656, 32]
        obs_feat = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train).view(-1, self.enc_h_dim)


        ### sg encoding
        _, sg_h = self.sg_rnn_enc(sg_state) # [8, 656, 16], 두개의 [1, 656, 32]
        sg_h = torch.cat(sg_h, dim=0).permute(1, 0, 2)

        sg_h = F.dropout(sg_h,
                        p=self.dropout_rnn,
                        training=train)  # [bs, max_time, enc_rnn_dim]

        sg_feat = self.fc_sg(sg_h.reshape(-1, 4 * self.enc_h_dim))

        #traj, map, SG
        hx = torch.cat((obs_feat, sg_feat), dim=-1) # 64(32 without attn) to z dim
        prior_stat = self.fc_hidden(hx)
        prior_stat = F.dropout(F.relu(prior_stat),
                      p=self.dropout_mlp,
                      training=train)

        prior_stat = self.fc_latent(prior_stat)

        return hx, prior_stat[:,:self.zS_dim], prior_stat[:,self.zS_dim:]


class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""

    def __init__(
            self, zS_dim, enc_h_dim=64, mlp_dim=32, hx_dim=128,
            num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0,
            device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim = zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.device = device
        n_state = 6
        n_pred_state = 2
        self.dropout_rnn = dropout_rnn
        self.dropout_mlp = dropout_mlp

        self.initial_h_model = nn.Linear(n_state, enc_h_dim)
        self.initial_c_model = nn.Linear(n_state, enc_h_dim)
        self.rnn_encoder = nn.LSTM(
            input_size=n_pred_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True
        )

        self.fc_hidden = nn.Linear(hx_dim + 4*enc_h_dim, mlp_dim)
        self.fc_latent = nn.Linear(mlp_dim, zS_dim*2)

    def forward(self, last_obs_traj, fut_vel, seq_start_end, hx, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory

        initial_h = self.initial_h_model(last_obs_traj)  # 81, 32
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)  # 2, 81, 32

        initial_c = self.initial_c_model(last_obs_traj)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        state_tuple = (initial_h, initial_c)

        _, state = self.rnn_encoder(fut_vel, state_tuple)

        final_encoder_h = torch.cat(state, dim=0).permute(1, 0, 2)  # 2,81,32두개 -> 4, 81,32 -> 81,4,32
        fut_feat = F.dropout(final_encoder_h,
                                    p=self.dropout_rnn,
                                    training=train)  # [bs, max_time, enc_rnn_dim]


        # final distribution
        fut_feat = fut_feat.reshape(-1, 4 * self.enc_h_dim)

        post_stat = self.fc_hidden(torch.cat([hx, fut_feat], -1))
        post_stat = F.dropout(F.relu(post_stat),
                      p=self.dropout_mlp,
                      training=train)
        post_stat = self.fc_latent(post_stat)
        return post_stat[:,:self.zS_dim], post_stat[:,self.zS_dim:]


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1, hx_dim=128,
        dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        device='cpu', scale=1, dt=0.4
    ):
        super(Decoder, self).__init__()
        n_state=6
        n_pred_state=2
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.dec_h_dim = dec_h_dim
        self.enc_h_dim = enc_h_dim
        self.device=device
        self.num_layers = num_layers
        self.dropout_rnn = dropout_rnn
        self.scale = scale
        self.dt = dt

        self.dec_hidden = nn.Linear(hx_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=hx_dim + z_dim + n_pred_state, hidden_size=dec_h_dim
        )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)

    def forward(self, last_obs_st, hx, z, sg_update_idx, sg_state, fut_vel_st=None):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_obs_st: Tensor of shape (batch, 6)
        - hx: hidden feature from the encoder
        - z: sample from the posterior/prior dist.
        - sg: sg position (batch, # sg, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - fut_vel_st: tensor of shape (seq_len, batch, 2)
        """

        # x_feat+z(=zx) initial state생성(FC)
        zx = torch.cat([hx, z], dim=1) # 493, 89(64+25)
        decoder_h=self.dec_hidden(zx) # 493, 128
        # Infer initial action state for node from current state
        pred_vel = self.to_vel(last_obs_st)


        ### traj decoding
        mus = []
        stds = []
        j=0
        for i in range(self.seq_len):
            decoder_h= self.rnn_decoder(torch.cat([zx, pred_vel], dim=1), decoder_h) #493, 128
            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            mus.append(mu)
            stds.append(std)

            if fut_vel_st is not None:
                pred_vel = fut_vel_st[i]
            else:
                if(i == sg_update_idx[j]):
                    pred_vel = sg_state[j+1,:,2:4]
                    j += 1
                else:
                    pred_vel = Normal(mu, std).rsample()

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        return Normal(mus, stds)


