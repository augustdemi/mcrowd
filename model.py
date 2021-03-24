import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from utils import sample_gaussian
from torch.distributions.normal import Normal

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
        self, embedding_dim=64, h_dim=64, pool_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(AttentionHiddenNet, self).__init__()

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
            curr_context_mat= []
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



class Encoder(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, embedding_dim=64, enc_h_dim=64, mlp_dim=32, pool_dim=32,
            batch_norm=False, num_layers=1, dropout=0.0, activation='relu', pooling_type='pool'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.h_dim = enc_h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type

        self.spatial_embedding = nn.Linear(2, embedding_dim)

        self.rnn_encoder = nn.LSTM(
            input_size=2, hidden_size=32
        )

        if pooling_type=='pool':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )
        elif pooling_type=='attn':
            self.pool_net = AttentionHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )

        input_dim = enc_h_dim + pool_dim

        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.fc2 = nn.Linear(input_dim, zS_dim)


    def forward(self, rel_traj, seq_start_end):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        # batch = rel_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

        _, (final_encoder_h, _) = self.rnn_encoder(rel_traj) # [8, 656, 16], 두개의 [1, 656, 32]

        # pooling
        if self.pooling_type:
            end_pos = rel_traj[-1, :, :] # 656, 2
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos) # 656, 32
            # Construct input hidden states for decoder
            dist_fc_input = torch.cat([final_encoder_h.squeeze(0), pool_h], dim=1) # [656, 64]
        else:
            dist_fc_input = final_encoder_h.view(-1, self.h_dim)


        # state = F.dropout(state,
        #                   p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                   training=(mode == ModeKeys.TRAIN))


        # final distribution
        # dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input) # 64(32 without attn) to z dim

        return dist_fc_input, stats


class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, embedding_dim=64, enc_h_dim=64, mlp_dim=32, pool_dim=32,
            batch_norm=False, num_layers=1, dropout=0.0, activation='relu', pooling_type='pool', device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim=zS_dim
        self.h_dim = enc_h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.device = device

        self.spatial_embedding = nn.Linear(2, embedding_dim)

        self.rnn_encoder = nn.LSTM(
            input_size=2, hidden_size=32, num_layers=1, bidirectional=True
        )

        if pooling_type=='pool':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )
        elif pooling_type=='attn':
            self.pool_net = AttentionHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )

        input_dim = 128+64


        # self.fc1 = make_mlp(
        #     [input_dim, mlp_dim],
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout
        # )
        self.fc2 = nn.Linear(input_dim, zS_dim)

        self.initial_h_model = nn.Linear(2, 32)
        self.initial_c_model = nn.Linear(2, 32)


    def forward(self, last_obs_rel_traj, fut_rel_traj, seq_start_end, obs_enc_feat):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory

        initial_h = self.initial_h_model(last_obs_rel_traj) # 81, 32
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0) # 2, 81, 32

        initial_c = self.initial_c_model(last_obs_rel_traj)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        state_tuple=(initial_h, initial_c)

        _, state = self.rnn_encoder(fut_rel_traj, state_tuple)

        state = torch.cat(state, dim=0).permute(1, 0, 2)  # 2,81,32두개 -> 4, 81,32 -> 81,4,32
        state_size = state.size()
        final_encoder_h = torch.reshape(state, (-1, state_size[1] * state_size[2]))  # [81, 128]


        # pooling
        # if self.pooling_type:
        #     end_pos = fut_rel_traj[-1, :, :]
        #     pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos) # 702, pool_dim(default=1024)
        #     # Construct input hidden states for decoder
        #     dist_fc_input = torch.cat([final_encoder_h.squeeze(0), pool_h], dim=1)
        # else:
        #     dist_fc_input = final_encoder_h.view(-1, self.h_dim)

        dist_fc_input = torch.cat([final_encoder_h, obs_enc_feat], dim=1)

        # state = F.dropout(state,
        #                   p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                   training=(mode == ModeKeys.TRAIN))


        # final distribution
        # dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input)

        return dist_fc_input, stats



class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout=0.0, pool_dim=1024, enc_h_dim=32, z_dim=32,
        activation='relu', batch_norm=False, device='cpu'
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.dec_h_dim = dec_h_dim
        self.enc_h_dim = enc_h_dim
        self.embedding_dim = 2
        # self.dec_inp_dim = embedding_dim
        self.dec_inp_dim = 64 + z_dim + 2
        self.device=device
        self.num_layers = num_layers

        self.rnn_decoder = nn.GRUCell(
            input_size=self.dec_inp_dim, hidden_size=dec_h_dim
        )

        self.mlp = make_mlp(
            [64 + z_dim, dec_h_dim], #mlp_dim + z_dim = enc_hidden_feat after mlp + z
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.spatial_embedding = nn.Linear(2, self.embedding_dim)
        self.fc_mu = nn.Linear(dec_h_dim, 2)
        self.fc_std = nn.Linear(dec_h_dim, 2)

    def forward(self, last_pos_rel, enc_h_feat, z):
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
        decoder_h=self.mlp(zx) # 493, 128

        mus = []
        stds = []
        for i in range(self.seq_len):
            decoder_h= self.rnn_decoder(torch.cat([zx, last_pos_rel], dim=1), decoder_h) #493, 128
            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            # normal = Normal(mu, std)
            # rel_pos = normal.rsample()
            mus.append(mu)
            stds.append(std)

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        rel_pos_dist =  Normal(mus, stds)
        return rel_pos_dist

# -----------------------------------------------------------------

