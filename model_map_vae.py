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
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_size=100, emb_dim=16,
            batch_norm=False, num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0, dropout_map=0.5, activation='relu'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.emb_dim=emb_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.dropout_rnn=dropout_rnn
        self.dropout_map=dropout_map
        self.map_size=map_size


        '''
        Trajectron++ map network
        '''
        # self.conv1 = nn.Conv2d(1, 4, 3, bias=False) # torch.Size([1088, 4, 41, 41])
        # self.pool = nn.MaxPool2d(2, 2) # torch.Size([1088, 4, 20, 20])
        # self.conv2 = nn.Conv2d(4, 8, 3, bias=False) # torch.Size([1088, 4, 8, 8])
        # self.map_fc = nn.Linear(4 * 4 * 4, rnn_input_dim, bias=False)

        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 4, 7, stride=3, bias=False), # torch.Size([1088, 4, 41, 41])
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # torch.Size([1088, 4, 20, 20])
            nn.Conv2d(4, 4, 5, stride=2, bias=False), # torch.Size([1088, 4, 8, 8])
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # torch.Size([1088, 4, 4, 4])
        )
        self.map_fc = nn.Linear(4 * 4 * 4, emb_dim, bias=False)

        # pool = nn.MaxUnpool2d(2,2)
        # de1 = nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1, bias=False)
        # de2 = nn.ConvTranspose2d(4, 1, 7, stride=3, bias=False)


        self.rnn_encoder = nn.LSTM(
            input_size=emb_dim, hidden_size=enc_h_dim
        )


        input_dim = enc_h_dim
        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_mlp
        )
        self.fc2 = nn.Linear(mlp_dim, zS_dim)


    def forward(self, obstacles, seq_start_end, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        # batch = rel_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

        # conv_feat = []
        # map_feat = self.conv1(obstacles.contiguous().view(-1, 1, self.map_size, self.map_size)) # torch.Size([179, 4, 4, 4])
        # conv_feat.append(map_feat)
        # map_feat = self.conv2(self.pool(map_feat))
        # conv_feat.append(map_feat)
        map_feat=self.map_encoder(obstacles.contiguous().view(-1, 1, self.map_size, self.map_size))
        map_feat = F.dropout(map_feat,
                            p=self.dropout_map,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        map_feat=map_feat.view(obstacles.shape[0],obstacles.shape[1], -1) #  torch.Size([8, 179, 64])
        map_emb = self.map_fc(map_feat) # torch.Size([8, 179, 16])

        _, (final_encoder_h, _) = self.rnn_encoder(map_emb) # [8, 656, 16], 두개의 [1, 656, 32]
        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]
        dist_fc_input = final_encoder_h.view(-1, self.enc_h_dim)

        # final distribution
        dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input) # 64(32 without attn) to z dim

        return map_feat[-1], dist_fc_input, stats


class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_size=128, emb_dim=16,
            num_layers=1, dropout_rnn=0.0, dropout_map=0.5, device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.device = device
        self.dropout_rnn=dropout_rnn
        self.dropout_map=dropout_map
        self.map_size=map_size

        '''
        Trajectron++ map network
        '''
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 4, 7, stride=3, bias=False), # torch.Size([1088, 4, 41, 41])
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # torch.Size([1088, 4, 20, 20])
            nn.Conv2d(4, 4, 5, stride=2, bias=False), # torch.Size([1088, 4, 8, 8])
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # torch.Size([1088, 4, 4, 4])
        )
        self.map_fc = nn.Linear(4 * 4 * 4, emb_dim, bias=False)


        self.rnn_encoder = nn.LSTM(
            input_size=emb_dim, hidden_size=enc_h_dim, num_layers=1, bidirectional=True)



        input_dim = enc_h_dim*4 + mlp_dim
        self.fc2 = nn.Linear(input_dim, zS_dim)

        self.initial_h_model = nn.Linear(map_size, enc_h_dim)
        self.initial_c_model = nn.Linear(map_size, enc_h_dim)


    def forward(self, last_past_obst, fut_obst, seq_start_end, obs_enc_feat, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode last observed map for initial state
        # initial_h = self.initial_h_model(last_past_obst) # 81, 32
        # initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0) # 2, 81, 32
        #
        # initial_c = self.initial_c_model(last_past_obst)
        # initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        # state_tuple=(initial_h, initial_c)

        # Encode future map
        fut_map_feat = self.map_encoder(fut_obst.contiguous().view(-1, 1, self.map_size, self.map_size))
        fut_map_feat = F.dropout(fut_map_feat,
                            p=self.dropout_map,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        fut_map_feat=fut_map_feat.view(fut_obst.shape[0],fut_obst.shape[1], -1)
        fut_map_emb = self.map_fc(fut_map_feat)

        _, state = self.rnn_encoder(fut_map_emb)

        state = torch.cat(state, dim=0).permute(1, 0, 2)  # 2,81,32두개 -> 4, 81,32 -> 81,4,32
        state_size = state.size()
        final_encoder_h = torch.reshape(state, (-1, state_size[1] * state_size[2]))  # [81, 128]

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        dist_fc_input = final_encoder_h.view(-1, 4 * self.enc_h_dim)


        dist_fc_input = torch.cat([dist_fc_input, obs_enc_feat], dim=1)


        # final distribution
        # dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input)

        return fut_map_feat, dist_fc_input, stats





class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        emb_dim=16, dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        device='cpu'
    ):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.dec_h_dim = dec_h_dim
        self.enc_h_dim = enc_h_dim
        self.device=device
        self.num_layers = num_layers
        map_feat_size=64

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)

        self.map_feat_to_emb = nn.Linear(map_feat_size, emb_dim)
        self.hidden_to_feat = nn.Linear(dec_h_dim, emb_dim)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + emb_dim, hidden_size=dec_h_dim
        )

        self.fc = nn.Linear(emb_dim, map_feat_size)

        self.deconv = nn.Sequential(
            nn.Upsample(8),
            nn.ConvTranspose2d(4, 4, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Upsample(41),
            nn.ConvTranspose2d(4, 1, 8, stride=3, bias=False),
            nn.Sigmoid()
        )
        # de1 = nn.ConvTranspose2d(4, 4, 4, stride=2, bias=False)
        # de2 = nn.ConvTranspose2d(4, 1, 8, stride=3, bias=False)
        # up2 = nn.Upsample(41)
        # up1 = nn.Upsample(8)

    def forward(self, last_past_map_feat, enc_h_feat, z, fut_state=None):
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
        a = self.map_feat_to_emb(last_past_map_feat)

        # dec rnn이 할일:deconv의 12 input만들기 = encY에서 conv의 12 output 만들기. 따라서 최초 a는 encX에서 conv의 8번째 output
        map_feat = []
        for i in range(self.seq_len):
            decoder_h= self.rnn_decoder(torch.cat([zx, a], dim=1), decoder_h) #493, 128
            a= self.hidden_to_feat(decoder_h)
            map_feat.append(a)

        map_feat = torch.stack(map_feat, dim=0)
        map_feat = self.fc(map_feat)

        map_mean = self.deconv(map_feat.view(-1,4,4,4))
        map_mean = map_mean.view(fut_state.shape[0], fut_state.shape[1], -1, map_mean.shape[2], map_mean.shape[3])
        map_dist=Laplace(map_mean, torch.tensor(0.01).to(z.device))
        return map_dist

