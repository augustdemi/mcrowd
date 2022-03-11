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
        self.fc_latent = nn.Linear(mlp_dim, zS_dim*2)


    def forward(self, obs_traj, seq_start_end, train=False):
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
        stats = self.fc_latent(hx)

        mu = stats[:, :self.zS_dim]
        log_var = stats[:, self.zS_dim:]
        mu = torch.clamp(mu, min=-1e8, max=1e8)
        log_var = torch.clamp(log_var, max=8e1)
        return hx, mu, log_var


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
        mu = stats[:, :self.zS_dim]
        log_var = stats[:, self.zS_dim:]
        mu = torch.clamp(mu, min=-1e8, max=1e8)
        log_var = torch.clamp(log_var, max=8e1)

        return mu, log_var


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        device='cpu', scale=1, dt=0.4, dropout_mlp=0.3, context_dim = 32
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
        self.context_dim = context_dim

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + 2*n_pred_state, hidden_size=dec_h_dim
        )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)

        self.sg_rnn_enc = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True)
        self.sg_fc = nn.Linear(4*enc_h_dim, n_pred_state)

        self.pool_net = PoolHiddenNet(
            h_dim=dec_h_dim,
            context_dim=context_dim,
            dropout=dropout_mlp
        )
        # self.mlp_context_enc = nn.Linear(enc_h_dim + dec_h_dim, dec_h_dim)
        self.mlp_context= nn.Linear(dec_h_dim + context_dim, dec_h_dim)

    def forward(self, seq_start_end, last_obs_st, last_pos, enc_h_feat, z, sg, sg_update_idx, fut_vel_st=None, train=False):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_obs_st: Tensor of shape (batch, 6)
        - enc_h_feat: hidden feature from the encoder
        - z: sample from the posterior/prior dist.
        - sg: sg position (batch, # sg, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - fut_vel_st: tensor of shape (seq_len, batch, 2)
        """
        # Infer initial action state for node from current state
        pred_vel = self.to_vel(last_obs_st)
        # pred_vel = last_obs_st[:,2:4] # bs, 2
        zx = torch.cat([enc_h_feat, z], dim=1) # bs, (32+20)
        decoder_h=self.dec_hidden(zx) # 493, 128

        # create context hidden feature
        # context = self.pool_net(enc_h_feat, seq_start_end, last_pos)  # batchsize, 1024
        # decoder_h=self.dec_hidden(torch.cat([enc_h_feat, context, z], dim=1)) # 493, 128


        ### make six states
        dt = self.dt * (12/len(sg_update_idx))
        last_ob_sg = torch.cat([last_pos.unsqueeze(1), sg], dim=1).detach().cpu().numpy()
        last_ob_sg = (last_ob_sg - last_ob_sg[:,:1])/self.scale # bs, 4(last obs + # sg), 2

        sg_state = []
        for pos in last_ob_sg:
            vx = np.gradient(pos[:,0], dt)
            vy = np.gradient(pos[:,1], dt)
            ax = np.gradient(vx, dt)
            ay = np.gradient(vy, dt)
            sg_state.append(np.array([pos[:,0], pos[:,1], vx, vy, ax, ay]))
        sg_state = torch.tensor(np.stack(sg_state)).permute((2,0,1)).float().to(z.device) # bs, 6, 4(last_obs + #sg) --> 4, bs, 6

        ### sg encoding
        _, sg_h = self.sg_rnn_enc(sg_state) # [8, 656, 16], 두개의 [1, 656, 32]
        sg_h = torch.cat(sg_h, dim=0).permute(1, 0, 2)
        sg_h = F.dropout(sg_h,
                        p=self.dropout_rnn,
                        training=train)  # [bs, max_time, enc_rnn_dim]
        sg_feat = self.sg_fc(sg_h.reshape(-1, 4 * self.enc_h_dim))


        ### traj decoding
        mus = []
        stds = []
        j=0
        for i in range(self.seq_len):
            # predict next position
            decoder_h= self.rnn_decoder(torch.cat([zx, pred_vel, sg_feat], dim=1), decoder_h) #493, 128
            mu = self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            # std = torch.sqrt(torch.exp(logVar))

            mu = torch.clamp(mu, min=-1e8, max=1e8)
            logVar = torch.clamp(logVar, max=8e1)
            std = torch.clamp(torch.sqrt(torch.exp(logVar)), min=1e-8)


            if fut_vel_st is not None:
                pred_vel = Normal(mu, std).rsample()
                curr_pos = pred_vel * self.scale * self.dt + last_pos
                context = self.pool_net(decoder_h, seq_start_end, curr_pos)  # batchsize, 1024
                decoder_h = self.mlp_context(torch.cat([decoder_h, context], dim=1))  # mlp : 1152 -> 1024 -> 128
                # refine the prediction
                mu = self.fc_mu(decoder_h)
                logVar = self.fc_std(decoder_h)
                mu = torch.clamp(mu, min=-1e8, max=1e8)
                logVar = torch.clamp(logVar, max=8e1)
                std = torch.clamp(torch.sqrt(torch.exp(logVar)), min=1e-8)
                pred_vel = Normal(mu, std).rsample()
                curr_pos = pred_vel * self.scale * self.dt + last_pos
                last_pos = curr_pos
            else:
                if (i == sg_update_idx[j]):
                    pred_vel = sg_state[j + 1, :, 2:4]
                    j += 1
                else:
                    pred_vel = Normal(mu, std).rsample()
                    curr_pos = pred_vel * self.scale * self.dt + last_pos
                    context = self.pool_net(decoder_h, seq_start_end, curr_pos)  # batchsize, 1024
                    decoder_h = self.mlp_context(
                        torch.cat([decoder_h, context], dim=1))  # mlp : 1152 -> 1024 -> 128
                    # refine the prediction
                    mu = self.fc_mu(decoder_h)
                    logVar = self.fc_std(decoder_h)
                    mu = torch.clamp(mu, min=-1e8, max=1e8)
                    logVar = torch.clamp(logVar, max=8e1)
                    std = torch.clamp(torch.sqrt(torch.exp(logVar)), min=1e-8)
                    pred_vel = Normal(mu, std).rsample()
                    curr_pos = pred_vel * self.scale * self.dt + last_pos
                    last_pos = curr_pos

            mus.append(mu)
            stds.append(std)

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        return Normal(mus, stds)

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
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

class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, h_dim=64, context_dim=32,
        activation='relu', batch_norm=False, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.h_dim = h_dim
        self.context_dim = context_dim
        # self.embedding_dim = embedding_dim

        mlp_pre_dim = 2 + 2*h_dim # 2+128*2
        mlp_pre_pool_dims = [mlp_pre_dim, 512, context_dim]

        # self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            num_ped = end - start
            curr_hidden = h_states[start:end] # (num_layer, batchsize, hidden_size) -> (num_layer*batchsize, hidden_size)
            curr_end_pos = end_pos[start:end]
            # hidden feature
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1) # Repeat -> H1, H2, H1, H2
            curr_hidden_2 = self.repeat(curr_hidden, num_ped) # Repeat -> H1, H2, H1, H2
            # position distance & embedding
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1) # Repeat position -> P1, P2, P1, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped) # Repeat position -> P1, P1, P2, P2
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2 # 다른 agent와의 relative거리 (a1-a1, a2-a1, a2-a1, a1-a2, a2-a2, a3-a2, a1-a3, a2-a3, a3-a3))이런식으로 상대거리
            # curr_rel_embedding = self.spatial_embedding(curr_rel_pos) # 다른 agent와의 relative거리의 embedding: (repeated data, 64)
            # mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1) #(repeated data, 64+128)
            mlp_h_input = torch.cat([curr_rel_pos, curr_hidden_1, curr_hidden_2], dim=1) #(repeated data, 64+128)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input) # 64+128 -> 512 -> (repeated data, bottleneck_dim)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0] # (sqrt(repeated data), sqrt(repeated data), 1024) 로 바꾼후, 각 agent별로 상대와의 거리가 가장 큰걸 골라냄. (argmax말로 value를)
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h
