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
def ConvBlock(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def ConvTransBlock(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 3, stride = 2, padding=1, output_padding = 1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def Maxpool():
    pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
    return pool

def ConvBlock2X(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        ConvBlock(in_dim, out_dim, act_fn),
        ConvBlock(out_dim, out_dim, act_fn),
    )
    return model

class LGEncoder(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, drop_out_conv=0., mlp_dim=32, drop_out_mlp=0.1, device='cpu'
    ):
        super(LGEncoder, self).__init__()
        act_fn = nn.ReLU
        in_dim=9
        # ch_dim = [32,32,64,64,64]
        ch_dim = 32

        self.down_1 = ConvBlock2X(in_dim, ch_dim, act_fn)
        self.pool_1 = Maxpool()
        self.down_2 = ConvBlock2X(ch_dim, ch_dim, act_fn)
        self.pool_2 = Maxpool()
        self.down_3 = ConvBlock2X(ch_dim, ch_dim * 2, act_fn)
        self.pool_3 = Maxpool()
        self.down_4 = ConvBlock2X(ch_dim * 2, ch_dim *2, act_fn)
        self.pool_4 = Maxpool()
        self.down_5 = ConvBlock2X(ch_dim * 2, ch_dim *2, act_fn)
        self.pool_5 = Maxpool()

        self.zS_dim=zS_dim
        self.enc_h_dim = 32 * 5 * 5
        self.fc1 = nn.Linear(self.enc_h_dim + 2, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, zS_dim, bias=False)


    def forward(self, obs_heat_map, last_obs_vel_local, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        down_1 = self.down_1(obs_heat_map)  # concat w/ trans_4
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)  # concat w/ trans_3
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)  # concat w/ trans_2
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)  # concat w/ trans_1
        pool_4 = self.pool_4(down_4)


        x = F.relu(self.conv1(obs_heat_map)) # 14
        x = self.pool(F.relu(self.conv2(x)))  # 12
        if (self.drop_out_conv > 0) and train:
            x = F.dropout(x,
                          p=self.drop_out,
                          training=train)

        x = F.relu(self.conv3(x))  # 10
        x = self.pool(F.relu(self.conv4(x))) # 8->4
        x = x.view(-1, self.enc_h_dim)
        # add last velocity
        x = torch.cat((x, last_obs_vel_local), -1)
        hx = self.fc1(x)
        x = F.relu(hx)
        if (self.drop_out_mlp > 0) and train:
            x = F.dropout(x,
                        p=self.drop_out_mlp,
                        training=train)
        z = self.fc2(x)

        return hx, z





class EncoderX(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_feat_dim=32, map_mlp_dim=32,
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


        self.fc1 = nn.Linear(enc_h_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim + map_feat_dim, zS_dim*2)

        # self.local_map_feat_dim = np.prod([*a])
        self.map_h_dim = 64*10*10

        self.map_fc1 = nn.Linear(self.map_h_dim + 9, map_mlp_dim)
        self.map_fc2 = nn.Linear(map_mlp_dim, map_feat_dim)


    def forward(self, obs_traj, seq_start_end, local_map_feat, local_homo, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """


        # traj rnn enc
        _, (final_encoder_h, _) = self.rnn_encoder(obs_traj) # [8, 656, 16], 두개의 [1, 656, 32]
        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]
        hx = self.fc1(final_encoder_h.view(-1, self.enc_h_dim))
        hx = F.dropout(F.relu(hx),
                      p=self.dropout_mlp,
                      training=train)

        # map enc
        local_map_feat = local_map_feat.view(-1, self.map_h_dim)
        local_homo = local_homo.view(-1, 9)
        map_feat = self.map_fc1(torch.cat((local_map_feat, local_homo), dim=-1))
        map_feat = F.dropout(F.relu(map_feat),
                      p=self.dropout_mlp,
                      training=train)
        map_feat = self.map_fc2(map_feat)

        # map and traj
        stats = self.fc2(torch.cat((hx, map_feat), dim=-1)) # 64(32 without attn) to z dim

        return hx, stats[:,:self.zS_dim], stats[:,self.zS_dim:]

class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""

    def __init__(
            self, zS_dim, enc_h_dim=64, mlp_dim=32,
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

        self.fc1 = nn.Linear(4*enc_h_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, zS_dim*2)

    def forward(self, last_obs_traj, fut_vel, seq_start_end, obs_enc_feat, train=False):
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
        final_encoder_h = F.dropout(final_encoder_h,
                                    p=self.dropout_rnn,
                                    training=train)  # [bs, max_time, enc_rnn_dim]


        # final distribution
        stats = self.fc1(final_encoder_h.reshape(-1, 4 * self.enc_h_dim))
        stats = F.dropout(F.relu(stats),
                      p=self.dropout_mlp,
                      training=train)
        stats = self.fc2(stats)

        return stats[:,:self.zS_dim], stats[:,self.zS_dim:]


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        device='cpu'
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

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + 2*n_pred_state, hidden_size=dec_h_dim
        )

        # self.mlp = make_mlp(
        #     [32 + z_dim, dec_h_dim], #mlp_dim + z_dim = enc_hidden_feat after mlp + z
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout_mlp
        # )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)


    def forward(self, last_obs_state, enc_h_feat, z, sg, fut_traj=None):
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

        # x_feat+z(=zx) initial state생성(FC)
        zx = torch.cat([enc_h_feat, z], dim=1) # 493, 89(64+25)
        decoder_h=self.dec_hidden(zx) # 493, 128
        # Infer initial action state for node from current state
        a = self.to_vel(last_obs_state)
        # a = self.to_vel(torch.cat((last_obs_traj_st, map[0]), dim=-1)) # map[0] = last observed map

        dt = 0.4*12

        rel_to_goal = (sg - last_obs_state[:, :2]) / dt
        # sg_vel_x = []
        # sg_vel_y = []
        # for pos in last_ob_sg:
        #     sg_vel_x.append(torch.gradient(pos[:,0], spacing=dt)[0])
        #     sg_vel_y.append(torch.gradient(pos[:,1], spacing=dt)[0])
        # sg_vel_x = torch.stack(sg_vel_x)
        # sg_vel_y = torch.stack(sg_vel_y)
        # sg_vel = torch.stack([sg_vel_x, sg_vel_y], dim=-1)
        #
        # for i in range(3):
        #     print((last_ob_sg[1, i+1] - last_ob_sg[1, i]) / dt)
        #     print(sg_vel[1,i])
        #     print('--------------')
        # print(sg_vel[1,3])

        mus = []
        stds = []
        j=0
        for i in range(self.seq_len):


            decoder_h= self.rnn_decoder(torch.cat([zx, a, rel_to_goal], dim=1), decoder_h) #493, 128
            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            mus.append(mu)
            stds.append(std)

            if fut_traj is not None:
                a = fut_traj[i,:,2:4]
            else:
                a = Normal(mu, std).rsample()

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        return Normal(mus, stds)


# def integrate_samples(v, p_0, dt=1):
#     """
#     Integrates deterministic samples of velocity.
#
#     :param v: Velocity samples
#     :return: Position samples
#     """
#     v=v.permute(1, 0, 2) #(t, bs, 2) -> (bs,t,2)
#     abs_traj = torch.cumsum(v, dim=1) * dt + p_0.unsqueeze(1)
#     return  abs_traj.permute((1, 0, 2))
