import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from gmm2d import GMM2D
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
        self, zS_dim, enc_h_dim=64, mlp_dim=32, pool_dim=32,
            batch_norm=False, num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0,  activation='relu', pooling_type='pool'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout_rnn=dropout_rnn
        n_state=6

        self.rnn_encoder = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim
        )

        # if pooling_type=='pool':
        #     self.pool_net = PoolHiddenNet(
        #         embedding_dim=self.embedding_dim,
        #         h_dim=enc_h_dim,
        #         pool_dim=pool_dim,
        #         batch_norm=batch_norm
        #     )
        # elif pooling_type=='attn':
        #     self.pool_net = AttentionHiddenNet(
        #         embedding_dim=self.embedding_dim,
        #         h_dim=enc_h_dim,
        #         pool_dim=pool_dim,
        #         batch_norm=batch_norm
        #     )

        input_dim = enc_h_dim + pool_dim

        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_mlp
        )
        self.fc2 = nn.Linear(mlp_dim, zS_dim)


    def forward(self, rel_traj, seq_start_end, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        # batch = rel_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

        _, (final_encoder_h, _) = self.rnn_encoder(rel_traj) # [8, 656, 16], 두개의 [1, 656, 32]

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        # pooling
        if self.pooling_type:
            end_pos = rel_traj[-1, :, :] # 656, 2
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos) # 656, 32
            # Construct input hidden states for decoder
            dist_fc_input = torch.cat([final_encoder_h.squeeze(0), pool_h], dim=1) # [656, 64]
        else:
            dist_fc_input = final_encoder_h.view(-1, self.enc_h_dim)


        # final distribution
        dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input) # 64(32 without attn) to z dim

        return dist_fc_input, stats


class EncoderY(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, pool_dim=32,
            batch_norm=False, num_layers=1,  dropout_mlp=0.0, dropout_rnn=0.0, activation='relu', pooling_type='pool', device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.device = device
        n_state=6
        n_pred_state=2
        self.dropout_rnn=dropout_rnn

        self.rnn_encoder = nn.LSTM(
            input_size=n_pred_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True
        )

        if pooling_type=='pool':
            self.pool_net = PoolHiddenNet(
                embedding_dim=n_pred_state,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )
        elif pooling_type=='attn':
            self.pool_net = AttentionHiddenNet(
                embedding_dim=n_pred_state,
                h_dim=enc_h_dim,
                pool_dim=pool_dim,
                batch_norm=batch_norm
            )

        input_dim = enc_h_dim*4 + mlp_dim


        # self.fc1 = make_mlp(
        #     [input_dim, mlp_dim],
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout_mlp
        # )
        self.fc2 = nn.Linear(input_dim, zS_dim)

        self.initial_h_model = nn.Linear(n_state, enc_h_dim)
        self.initial_c_model = nn.Linear(n_state, enc_h_dim)


    def forward(self, last_obs_rel_traj, fut_rel_traj, seq_start_end, obs_enc_feat, train=False):
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

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

        dist_fc_input = torch.cat([final_encoder_h, obs_enc_feat], dim=1)


        # final distribution
        # dist_fc_input = self.fc1(dist_fc_input)
        stats = self.fc2(dist_fc_input)

        return dist_fc_input, stats





class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout_mlp=0.0, dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        activation='relu', batch_norm=False, device='cpu'
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
        self.dropout_rnn=dropout_rnn
        self.z_dim=z_dim

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + n_pred_state, hidden_size=dec_h_dim
        )

        # self.mlp = make_mlp(
        #     [32 + z_dim, dec_h_dim], #mlp_dim + z_dim = enc_hidden_feat after mlp + z
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout_mlp
        # )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)

        self.fc_log_pis = nn.Linear(dec_h_dim, 1)
        self.fc_corrs = nn.Sequential(
            nn.Linear(dec_h_dim, 1),
            nn.Tanh()
        )


    def forward(self, last_state, enc_h_feat, z, fut_traj=None, train=False, num_samples=1):
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
        num_components=z.shape[0]
        # zx = torch.cat([enc_h_feat, z], dim=1) # 493, 89(64+25)
        z = torch.reshape(z, (-1, self.z_dim))
        zx = torch.cat([enc_h_feat.repeat(num_samples * num_components, 1), z], dim=1)
        state=self.dec_hidden(zx) # 493, 128
        a_0 = self.to_vel(last_state)

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)  # 6400, 99(97+2)
        # input_ = torch.cat([zx, a_0], dim=1)  # 6400, 99(97+2)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        for i in range(self.seq_len):
            h_state = self.rnn_decoder(input_, state) # 6400, 128 or 256,128 (test time: 20,128)

            log_pi_t = self.fc_log_pis(h_state) # 577, 1
            mu_t = self.fc_mu(h_state) # 577, 2
            log_sigma_t = self.fc_std(h_state) # 577, 2
            corr_t = self.fc_corrs(h_state) # 577, 1

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]
            if train:
                a_t = fut_traj[i,:,2:4].repeat(num_samples * num_components, 1)
            else:
                a_t = gmm.rsample() #577, 2 (test time:20,2)

            log_pis.append(
                torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
            )

            # mu_t = 6400(256*25),2 -> reshape: 256,50
            mus.append(mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            log_sigmas.append(log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            input_ = torch.cat([zx, a_t], dim=1) # 6400, 99(97+2)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1) # [256, 12, 1, 25*1]
        mus = torch.stack(mus, dim=1) # [256, 50] 12개 쌓아서 [256, 12, 25*2]
        log_sigmas = torch.stack(log_sigmas, dim=1) # 256, 12, 50
        corrs = torch.stack(corrs, dim=1) # 256, 12, 25

        rel_pos_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, self.seq_len, num_components]),
                       torch.reshape(mus, [num_samples, -1, self.seq_len, num_components*2]),
                       torch.reshape(log_sigmas, [num_samples, -1, self.seq_len, num_components*2]),
                       torch.reshape(corrs, [num_samples, -1, self.seq_len, num_components])) # 256,12,25

        # rel_pos_dist = GMM2D(log_pis.unsqueeze(0), mus.unsqueeze(0), log_sigmas.unsqueeze(0), corrs.unsqueeze(0))
        #rel_pos_dist.mus.shape = [1, 577, 12, 1, 2]

        return rel_pos_dist
