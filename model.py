import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from gmm2d import GMM2D
from torch.distributions.normal import Normal
import random
from data.trajectories import crop

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



class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, pool_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.pool_dim = pool_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, pool_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
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
        - pool_h: Tensor of shape (batch, pool_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end] # (num_layer, batchsize, hidden_size) -> (num_layer*batchsize, hidden_size) -> (num_ped, hidden_size)
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1) #(num_ped*num_ped, hidden_size)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped) #(num_ped*num_ped, hidden_size)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2 # 다른 agent와의 relative거리 (a1-a1, a2-a1, a3-a1, a1-a2, a2-a2, a3-a2, a1-a3, a2-a3, a3-a3))이런식으로 상대거리

            curr_rel_embedding = self.spatial_embedding(curr_rel_pos) # 다른 agent와의 relative거리의 embedding: (repeated data, 64)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1) #(repeated data, 64+128)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input) # 64+128 -> 512 -> (repeated data, bottleneck_dim)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0] # (sqrt(repeated data), sqrt(repeated data), 1024) 로 바꾼후, 각 agent별로 상대와의 거리가 가장 큰걸 골라냄. (argmax말로 value를)
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



class AttentionHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, enc_h_dim=64
    ):
        super(AttentionHiddenNet, self).__init__()
        self.h_dim = enc_h_dim

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

    def forward(self, h_states, seq_start_end):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - context_mat: Tensor of shape (batch, pool_dim)
        """
        context_mat = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end] #num_ped, latent
            score=torch.matmul(curr_hidden, curr_hidden.transpose(1,0)) #num_ped, num_ped
            attn_dist = torch.softmax(score, dim=1) #(num_ped, num_ped)
            curr_context_mat= [] # 현재 start-end 프레임 안의 num ped만큼의 hidden feat들이 서로간 이루는 attn_dist(score)값을 반영한 context vec를 모음.
            for i in range(num_ped):
                curr_attn = attn_dist[i].repeat(curr_hidden.size(1), 1).transpose(1, 0)
                context_vec = torch.sum(curr_attn * curr_hidden, dim=0)
                curr_context_mat.append(context_vec)
            # for i in range(num_ped):
            #     context_vec = torch.zeros(curr_hidden.size(1)).to(self.device)
            #     for j in range(num_ped):
            #         context_vec += attn_dist[i][j] * curr_hidden[j] #(latent)
            #     curr_context_mat.append(context_vec)
            context_mat.append(torch.stack(curr_context_mat))

        context_mat = torch.cat(context_mat, dim=0)
        return context_mat


class CNNMapEncoder(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim):
        super(CNNMapEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 4, stride=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 4, 3, stride=1, bias=False)
        self.conv3 = nn.Conv2d(4, 4, 3, stride=1, bias=False)
        self.fc1 = nn.Linear(4 * 6 * 6 + 2, fc_hidden_dim, bias=False)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim, bias=False)

    def forward(self, x, v):
        x = self.pool(F.relu(self.conv1(x)))  # 64->30
        x = self.pool(F.relu(self.conv2(x)))  # 30 ->14
        x = self.pool(F.relu(self.conv3(x)))  # 14 ->6
        x = x.view(-1, 4 * 6 * 6)
        x = torch.cat((x, v), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class Encoder(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, attention=False, map_size=64,
            batch_norm=False, num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0,  activation='relu'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.dropout_rnn=dropout_rnn
        self.attention=attention
        self.map_size=map_size

        n_state=6
        map_out_dim = 8

        self.rnn_encoder = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim
        )

        self.attn_net = AttentionHiddenNet(
            enc_h_dim=enc_h_dim,
        )

        input_dim = enc_h_dim
        if attention:
            input_dim += enc_h_dim

        self.map_net = CNNMapEncoder(
            fc_hidden_dim=mlp_dim,
            output_dim=map_out_dim
        )
        input_dim += map_out_dim


        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_mlp
        )
        self.fc2 = nn.Linear(mlp_dim, zS_dim)


    def forward(self, obs_state, seq_start_end, past_obstacle, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        # batch = rel_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

        _, (final_encoder_h, _) = self.rnn_encoder(obs_state) # [8, 656, 16], 두개의 [1, 656, 32]

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        # attention
        if self.attention:
            pool_h = self.attn_net(final_encoder_h, seq_start_end) # 656, 32
            # Construct input hidden states for decoder
            dist_fc_input = torch.cat([final_encoder_h.squeeze(0), pool_h], dim=1) # [656, 64]
        else:
            dist_fc_input = final_encoder_h.view(-1, self.enc_h_dim)


        # map encoding
        if self.map_size:
            obst_feat = self.map_net(past_obstacle[-1], obs_state[-1,:,2:4]) # obstacle map + velocity
            obst_feat = F.dropout(obst_feat,
                                     p=0.5,
                                     training=train)
            dist_fc_input = torch.cat([dist_fc_input, obst_feat], dim=1)

        # final distribution
        dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input) # 64(32 without attn) to z dim

        return dist_fc_input, stats


class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, attention=False,
            batch_norm=False, num_layers=1,  dropout_mlp=0.0, dropout_rnn=0.0, activation='relu', device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.device = device
        self.attention=attention
        n_state=6
        n_pred_state=2
        self.dropout_rnn=dropout_rnn

        self.initial_h_model = nn.Linear(n_state, enc_h_dim)
        self.initial_c_model = nn.Linear(n_state, enc_h_dim)
        self.rnn_encoder = nn.LSTM(
            input_size=n_pred_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True
        )

        enc_h_dim = enc_h_dim*4
        self.attn_net = AttentionHiddenNet(
            enc_h_dim=enc_h_dim,
        )
        input_dim = enc_h_dim + mlp_dim
        if attention:
            input_dim +=enc_h_dim


        # self.fc1 = make_mlp(
        #     [input_dim, mlp_dim],
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout_mlp
        # )
        self.fc2 = nn.Linear(input_dim, zS_dim)




    def forward(self, last_obs_traj, fut_rel_traj, seq_start_end, obs_enc_feat, fut_obst, train=False):
        """
        Inputs:
        - last_obs_rel_traj: Tensor of shape (bs, 6)
        Output:
        - fut_rel_traj: Tensor of shape [12, bs, 2]
        """
        # Encode observed Trajectory

        initial_h = self.initial_h_model(last_obs_traj) # bs, 32
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0) # 2, bs, 32

        initial_c = self.initial_c_model(last_obs_traj)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        state_tuple=(initial_h, initial_c)

        _, state = self.rnn_encoder(fut_rel_traj, state_tuple)

        state = torch.cat(state, dim=0).permute(1, 0, 2)  # 2,bs,32두개 -> 4, bs,32 -> bs,4,32
        state_size = state.size()
        final_encoder_h = torch.reshape(state, (-1, state_size[1] * state_size[2]))  # [bs, 128]

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        # attention
        if self.attention:
            pool_h = self.attn_net(final_encoder_h, seq_start_end) # 656, 32
            # Construct input hidden states for decoder
            dist_fc_input = torch.cat([final_encoder_h.view(-1, 4*self.enc_h_dim), pool_h], dim=1) # [656, 64]
        else:
            dist_fc_input = final_encoder_h.view(-1, 4*self.enc_h_dim)

        dist_fc_input = torch.cat([dist_fc_input, obs_enc_feat], dim=1)


        # final distribution
        # dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input)

        return dist_fc_input, stats





class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout_mlp=0.0, dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        activation='relu', batch_norm=False, device='cpu', attention=False
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
        self.attention = attention

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + n_pred_state, hidden_size=dec_h_dim
        )


        if attention:
            self.attn_net = AttentionHiddenNet(
                enc_h_dim=dec_h_dim,
            )
            self.fc_attn = nn.Linear(2*dec_h_dim, dec_h_dim)

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)

    def forward(self, last_state, enc_h_feat, z, seq_start_end, fut_obst, fut_state=None):
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
        # if fut_state:
        #     train=True
        # else:
        #     train=False

        # x_feat+z(=zx) initial state생성(FC)
        zx = torch.cat([enc_h_feat, z], dim=1) # 493, 89(64+25)
        decoder_h=self.dec_hidden(zx) # 493, 128
        a = self.to_vel(last_state)

        mus = []
        stds = []
        for i in range(self.seq_len):
            decoder_h= self.rnn_decoder(torch.cat([zx, a], dim=1), decoder_h) #493, 128
            if self.attention:
                pool_h = self.attn_net(decoder_h, seq_start_end)  # 656, 32
                # Construct input hidden states for decoder
                decoder_h = torch.cat([decoder_h, pool_h], dim=1)  # [656, 64]
                decoder_h = self.fc_attn(decoder_h)
            # if self.map_size:
            #     if train:
            #         fut_cropped_map = fut_obst[i]
            #     else:
            #
            #         fut_cropped_map =  crop(map, target_pos1, inv_h_t, mean_pos, context_size=198)
            #     obst_feat = self.map_net(fut_cropped_map, fut_state[i])  # obstacle map + velocity
            #     obst_feat = F.dropout(obst_feat,
            #                           p=0.5,
            #                           training=train)
            #     decoder_h = torch.cat([decoder_h, obst_feat], dim=1)  # [656, 64]

            # if self.attention or self.map_size:
            #     decoder_h = self.fc_attn(decoder_h)

            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            if fut_state is not None:
                a = fut_state[i]
            else:
                a = Normal(mu, std).rsample()
            mus.append(mu)
            stds.append(std)

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        rel_pos_dist =  Normal(mus, stds)
        return rel_pos_dist
