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



def load_map_encoder(device, map_feat_dim):
    # map_encoder_path = 'ckpts/nmap_map_size_180_drop_out0.0_run_25/iter_6500_encoder.pt'
    map_encoder_path = 'ckpts/A2E_map_size_16_drop_out0.1_hidden_d256_latent_d' + str(map_feat_dim)+'_run_4/iter_20000_encoder.pt'
    if device == 'cuda':
        map_encoder = torch.load(map_encoder_path)
    else:
        map_encoder = torch.load(map_encoder_path, map_location='cpu')
    for p in map_encoder.parameters():
        p.requires_grad = False
    return map_encoder


def crop(map, target_pos, inv_h_t, context_size=198):
    # context_size=32
    expanded_obs_img = np.full((map.shape[0] + context_size, map.shape[1] + context_size), False, dtype=np.float32)
    expanded_obs_img[context_size//2:-context_size//2, context_size//2:-context_size//2] = map.astype(np.float32) # 99~-99


    target_pixel = np.matmul(np.concatenate([target_pos.detach().cpu().numpy(), np.ones((len(target_pos), 1))], axis=1), inv_h_t)
    target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    target_pixel = target_pixel[:,:2]

    # plt.imshow(map)
    # for i in range(len(target_pixel)):
    #     plt.scatter(target_pixel[i][0], target_pixel[i][1], c='r', s=1)
    # plt.show()

    img_pts = context_size//2 + np.round(target_pixel).astype(int)

    # if (img_pts[i][0] < context_size // 2)

    nearby_area = context_size//2
    cropped_img = []
    for i in range(target_pos.shape[0]):
        im = expanded_obs_img[img_pts[i, 1] - nearby_area: img_pts[i, 1] + nearby_area,
        img_pts[i, 0] - nearby_area: img_pts[i, 0] + nearby_area]
        if np.prod(im.shape) != context_size**2:
            im = np.ones((context_size, context_size))
        cropped_img.append(transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor()
        ])(Image.fromarray(im)))
    cropped_img = torch.stack(cropped_img, axis=0)

    cropped_img[:,: , nearby_area, nearby_area] = 0

    # plt.imshow(cropped_img[0,0])
    # plt.show()

    return cropped_img

###############################################################################
# -----------------------------------------------------------------------

class Encoder(nn.Module):
    """Encoder:spatial emb -> lstm -> pooling -> fc for posterior / conditional prior"""
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_feat_dim=32,
            batch_norm=False, num_layers=1, dropout_mlp=0.0, dropout_rnn=0.0,  activation='relu', pooling_type='pool', device='cpu'
    ):
        super(Encoder, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout_rnn=dropout_rnn
        n_state=6

        self.rnn_encoder = nn.LSTM(
            input_size=n_state + map_feat_dim, hidden_size=enc_h_dim
        )

        input_dim = enc_h_dim

        self.fc1 = make_mlp(
            [input_dim, mlp_dim],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout_mlp
        )
        self.fc2 = nn.Linear(mlp_dim, zS_dim)
        self.map_encoder = load_map_encoder(device, map_feat_dim)


    def forward(self, obs_traj_st, seq_start_end, map, obs_vel, train=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        map_feat = self.map_encoder(obs_vel.reshape(-1, 2), map.reshape(-1, map.shape[2], map.shape[3], map.shape[4]), train=False)
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
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_feat_dim=32,
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
            input_size=n_pred_state + map_feat_dim, hidden_size=enc_h_dim, num_layers=1, bidirectional=True
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
        self.map_encoder = load_map_encoder(device, map_feat_dim)


    def forward(self, last_obs_traj_st, fut_vel_st, seq_start_end, obs_enc_feat, map, fut_vel, train=False):
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

        map_feat = self.map_encoder(fut_vel.reshape(-1, 2), map.reshape(-1, map.shape[2], map.shape[3], map.shape[4]), train=False)
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
        map_feat_dim=32, dropout_rnn=0.0, enc_h_dim=32, z_dim=32,
        activation='relu', batch_norm=False, device='cpu', map_size=180
    ):
        super(Decoder, self).__init__()
        n_state=6
        n_pred_state=2
        d_map_feat=map_feat_dim
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
        self.map_encoder = load_map_encoder(device, map_feat_dim)


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
            map_feat = self.map_encoder(a, map, train=False)

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
                    seq_map = imageio.imread(map_path[j])  # seq = 한 씬에서 모든 neighbors니까. 같은 데이터셋.
                    seq_cropped_map = crop(seq_map, pred_fut_traj[s:e], inv_h_t[j],
                                           context_size=self.map_size)  # (e-s), 1, 64, 64
                    map.append(seq_cropped_map)
                map = torch.cat(map).to(self.device)
                ####

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        rel_pos_dist =  Normal(mus, stds)
        return rel_pos_dist

