import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import imageio
from utils import crop

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



def load_map_encoder(device):
    map_encoder_path = 'ckpts/nmap_map_size_180_drop_out0.0_run_25/iter_6500_encoder.pt'
    if device == 'cuda':
        map_encoder = torch.load(map_encoder_path)
    else:
        map_encoder = torch.load(map_encoder_path, map_location='cpu')
    for p in map_encoder.parameters():
        p.requires_grad = False
    return map_encoder

###############################################################################
# -----------------------------------------------------------------------

class Encoder(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, pool_dim=32,
            batch_norm=False, num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0,  activation='relu', pooling_type='pool', device='cpu'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout_rnn=dropout_rnn
        n_state=6+8

        self.rnn_encoder = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim
        )

        input_dim = enc_h_dim

        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_mlp
        )
        self.fc2 = nn.Linear(mlp_dim, zS_dim)
        self.map_encoder = load_map_encoder(device)


    def forward(self, obs_traj_st, seq_start_end, map, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        map_feat = self.map_encoder(map.reshape(-1, map.shape[2], map.shape[3], map.shape[4]), train=False)
        map_feat = map_feat.reshape((8, -1, map_feat.shape[-1]))

        rnn_input = torch.cat((obs_traj_st, map_feat), dim=-1)
        _, (final_encoder_h, _) = self.rnn_encoder(rnn_input) # [8, 656, 16], 두개의 [1, 656, 32]

        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)  # [bs, max_time, enc_rnn_dim]

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
        n_pred_state=2+8
        self.dropout_rnn=dropout_rnn

        self.rnn_encoder = nn.LSTM(
            input_size=n_pred_state, hidden_size=enc_h_dim, num_layers=1, bidirectional=True
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
        self.map_encoder = load_map_encoder(device)


    def forward(self, last_obs_traj_st, fut_vel_st, seq_start_end, obs_enc_feat, map, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory

        initial_h = self.initial_h_model(last_obs_traj_st) # 81, 32
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0) # 2, 81, 32

        initial_c = self.initial_c_model(last_obs_traj_st)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        state_tuple=(initial_h, initial_c)

        map_feat = self.map_encoder(map.reshape(-1, map.shape[2], map.shape[3], map.shape[4]), train=False)
        map_feat = map_feat.reshape((12, -1, map_feat.shape[-1]))

        rnn_input = torch.cat((fut_vel_st, map_feat), dim=-1)
        _, state = self.rnn_encoder(rnn_input, state_tuple)

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
        activation='relu', batch_norm=False, device='cpu', map_size=180
    ):
        super(Decoder, self).__init__()
        n_state=6
        n_pred_state=2
        d_map_feat=8
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.dec_h_dim = dec_h_dim
        self.enc_h_dim = enc_h_dim
        self.device=device
        self.num_layers = num_layers
        self.map_size = map_size

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + n_pred_state + d_map_feat, hidden_size=dec_h_dim
        )

        # self.mlp = make_mlp(
        #     [32 + z_dim, dec_h_dim], #mlp_dim + z_dim = enc_hidden_feat after mlp + z
        #     activation=activation,
        #     batch_norm=batch_norm,
        #     dropout=dropout_mlp
        # )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)
        self.map_encoder = load_map_encoder(device)


    def forward(self, last_obs_traj_st, enc_h_feat, z, last_obs_and_fut_map, fut_traj=None, map_info=None):
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
        a = self.to_vel(last_obs_traj_st)
        # a = self.to_vel(torch.cat((last_obs_traj_st, map[0]), dim=-1)) # map[0] = last observed map

        if fut_traj is None:
            seq_start_end, map_path, inv_h_t, integrate_fn = map_info[0], map_info[1], map_info[2], map_info[3]

        mus = []
        stds = []
        map = last_obs_and_fut_map[0]
        for i in range(self.seq_len):
            map_feat = self.map_encoder(map, train=False)

            decoder_h= self.rnn_decoder(torch.cat([zx, a, map_feat], dim=1), decoder_h) #493, 128
            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            mus.append(mu)
            stds.append(std)

            if fut_traj is not None:
                a = fut_traj[i,:,2:4]
                map = last_obs_and_fut_map[i+1]
            else:
                a = Normal(mu, std).rsample()
                ####
                pred_fut_traj = integrate_fn(a.unsqueeze(0)).squeeze(0)
                map = []
                for j, (s, e) in enumerate(seq_start_end):
                    if map_path[j] is None:
                        seq_cropped_map = torch.zeros((e - s), 1, 64, 64)
                        seq_cropped_map[:, 0, 31, 31] = 0.0144
                        seq_cropped_map[:, 0, 31, 32] = 0.0336
                        seq_cropped_map[:, 0, 32, 31] = 0.0336
                        seq_cropped_map[:, 0, 32, 32] = 0.0784
                    else:
                        seq_map = imageio.imread(map_path[j])  # seq = 한 씬에서 모든 neighbors니까. 같은 데이터셋.
                        seq_cropped_map = crop(seq_map, pred_fut_traj[s:e], inv_h_t[j],
                                               context_size=self.map_size)  # (e-s), 1, 64, 64
                    map.append(seq_cropped_map)
                map = torch.cat(map)
                ####

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        rel_pos_dist =  Normal(mus, stds)
        return rel_pos_dist

