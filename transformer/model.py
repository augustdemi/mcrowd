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
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model), PointerwiseFeedforward(d_model, d_ff, dropout), dropout), N)
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
        # logit_token = Variable(torch.FloatTensor(np.random.rand(src.shape[0], 1, self.d_model))).to(src.device)
        # src_emb = torch.cat((logit_token, self.embed_fn(src)), dim=1)
        src_emb = self.embed_fn(src)
        enc_out = self.encoder(src_emb, src_mask) # bs, 1+8, 512
        logit = self.fc(enc_out).max(1)[0] # pooling the latent dist logit throughout the "time step" dim

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
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model), PointerwiseFeedforward(d_model, d_ff, dropout), dropout), N)
        self.fc = nn.Linear(d_model, d_latent)

        self.init_weights(self.encoder.parameters())
        self.init_weights(self.fc.parameters())

    def init_weights(self, params):
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src_trg, src_trg_mask):
        # logit_token = Variable(torch.FloatTensor(np.random.rand(src_trg.shape[0], 1, self.d_model))).to(src_trg.device)
        # src_trg_emb = torch.cat((logit_token, self.embed_fn(src_trg)), dim=1)
        src_trg_emb = self.embed_fn(src_trg)
        enc_out = self.encoder(src_trg_emb, src_trg_mask) # bs, 1+8, 512
        logit = self.fc(enc_out).max(1)[0] # pooling the latent dist logit throughout the "time step" dim

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
        self.decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(h, d_model), MultiHeadAttention(h, d_model),
                                            ConcatPointerwiseFeedforward(d_model, d_latent, d_ff, dropout), dropout), N)
        self.fc = nn.Linear(d_model, dec_out_size * 2)

        self.init_weights(self.decoder.parameters())
        self.init_weights(self.fc.parameters())

    def init_weights(self, params):
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, enc_out, latents, trg, src_mask, trg_mask):
        dec_out =  self.decoder(self.trg_embed(trg), enc_out, latents.unsqueeze(1), src_mask, trg_mask) # bs, 12, 512
        stats = self.fc(dec_out) # bs, 12, out*2
        mu = stats[:,:,:self.dec_out_size]
        logvar = stats[:,:,self.dec_out_size:]
        return mu, logvar
