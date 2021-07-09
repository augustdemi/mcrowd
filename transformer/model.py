import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import numpy as np

from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward, ConcatPointerwiseFeedforward
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from torch.distributions.normal import Normal


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EncoderX(nn.Module):
    def __init__(self, enc_inp_size, d_latent, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(EncoderX, self).__init__()

        self.d_model = d_model
        self.embed_fn = nn.Sequential(
            LinearEmbedding(enc_inp_size,d_model),
            PositionalEncoding(d_model, dropout)
        )
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model, dropout), PointerwiseFeedforward(d_model, d_ff, dropout), dropout), N)
        # layer = EncoderLayer(d_model, MultiHeadAttention(h, d_model), PointerwiseFeedforward(d_model, d_ff, dropout), dropout)
        # self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)
        self.fc = nn.Linear(d_model, d_latent)
        # self.fc2 = nn.Linear(d_model, d_latent)

        self.init_weights(self.encoder.parameters())
        self.init_weights(self.fc.parameters())

    def init_weights(self, params):
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, src, src_mask):
        logit_token = Variable(torch.FloatTensor(np.random.rand(src.shape[0], 1, self.d_model))).to(src.device)
        src_emb = torch.cat((logit_token, self.embed_fn(src)), dim=1)
        enc_out = self.encoder(src_emb, src_mask) # bs, 1+8, 512
        logit = self.fc(enc_out[:,0])

        return enc_out[:,1:], logit



class EncoderY(nn.Module):
    def __init__(self, enc_inp_size, d_latent, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(EncoderY, self).__init__()
        self.d_model = d_model
        self.embed_fn = nn.Sequential(
            LinearEmbedding(enc_inp_size,d_model),
            PositionalEncoding(d_model, dropout)
        )
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model, dropout), PointerwiseFeedforward(d_model, d_ff, dropout), dropout), N)
        self.fc = nn.Linear(d_model, d_latent)

        self.init_weights(self.encoder.parameters())
        self.init_weights(self.fc.parameters())

    def init_weights(self, params):
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src_trg, src_trg_mask):
        logit_token = Variable(torch.FloatTensor(np.random.rand(src_trg.shape[0], 1, self.d_model))).to(src_trg.device)
        src_trg_emb = torch.cat((logit_token, self.embed_fn(src_trg)), dim=1)
        enc_out = self.encoder(src_trg_emb, src_trg_mask) # bs, 1+8, 512
        logit = self.fc(enc_out[:,0])

        return enc_out[:,1:], logit



class DecoderY(nn.Module):
    def __init__(self, dec_inp_size, dec_out_size, d_latent, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(DecoderY, self).__init__()

        self.dec_out_size = dec_out_size
        self.d_model = d_model

        self.trg_embed = nn.Sequential(
            LinearEmbedding(dec_inp_size,d_model),
            PositionalEncoding(d_model, dropout)
        )
        self.decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(h, d_model, dropout), MultiHeadAttention(h, d_model, dropout),
                                            ConcatPointerwiseFeedforward(d_model, 2*d_latent, d_ff, dropout), dropout), N)
        self.neighbor_attn = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model, dropout), PointerwiseFeedforward(d_model, d_ff, dropout), dropout), N)
        self.neighbor_fc = nn.Linear(d_model, d_latent)
        self.fc = nn.Linear(d_model, dec_out_size * 2)

        self.init_weights(self.decoder.parameters())
        self.init_weights(self.fc.parameters())
        self.init_weights(self.neighbor_attn.parameters())

    def init_weights(self, params):
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, enc_out, latents, trg, src_mask, trg_mask, seq_start_end, src_traj):
        bs = enc_out.shape[0]
        last_obs_enc_feat = enc_out[:,-1]
        max_n_agents = (seq_start_end[:,1] - seq_start_end[:,0]).max()
        neighbor_mask = torch.zeros((bs, max_n_agents))
        last_obs_neighbors = torch.zeros((bs, max_n_agents, self.d_model))

        i = 0
        for seq in seq_start_end:
            num_ped = seq[1] - seq[0]
            curr_seq_all_agents_feat = last_obs_enc_feat[seq[0]:seq[1]]
            for a in range(num_ped):
                last_obs_neighbors[i, : num_ped] = curr_seq_all_agents_feat
                neighbor_mask[i, :num_ped] = 1
                i+=1

        neighbor_feat = self.neighbor_attn(last_obs_neighbors.to(enc_out.device), neighbor_mask.unsqueeze(1).to(enc_out.device)) # bs, max_n_agents, 512
        neighbor_feat = self.neighbor_fc(neighbor_feat).mean(1) # avg pooling

        dec_out =  self.decoder(self.trg_embed(trg), enc_out, torch.cat([latents, neighbor_feat], dim=1).unsqueeze(1), src_mask, trg_mask) # bs, 12, 512

        stats = self.fc(dec_out) # bs, 12, out*2
        mu = stats[:,:,:self.dec_out_size]
        logvar = stats[:,:,self.dec_out_size:]
        return mu, logvar
