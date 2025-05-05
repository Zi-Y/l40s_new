import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from torch.distributions import Normal

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        self.configs = configs
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # Output projection layer(s)
        if configs.probabilistic:
            # 修改: 在概率模式下使用两个输出层（均值和标准差）
            self.projection_mu = nn.Linear(configs.d_model, configs.pred_len, bias=True)       # project to prediction length for mean
            self.projection_log_sigma = nn.Linear(configs.d_model, configs.pred_len, bias=True)    # project to prediction length for std (or log-std)
        else:
            # 原有确定性输出层
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)      # project to prediction length for deterministic output


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.configs.probabilistic:
            mu = self.projection_mu(enc_out).permute(0, 2, 1)[:, :, :N]
            log_sigma = self.projection_log_sigma(enc_out).permute(0, 2, 1)[:, :, :N]
            # 采用 log尺度归一化：sigma = exp(log_sigma)
            sigma = torch.exp(log_sigma)
            # 反归一化：将预测均值和标准差恢复到原始尺度

            if self.use_norm:
                mu = mu * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)  # 反归一化
                sigma = sigma * stdev[:, 0, :].unsqueeze(1)  # 仅乘 stdev 反归一化
            dec_out = [mu, sigma]
        else:
            # B N E -> B N S -> B S N
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

            if self.use_norm:
                # De-Normalization from Non-stationary Transformer
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if self.output_attention:
            return dec_out, attns

        else:
            return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            attns = dec_out[1]
            dec_out = dec_out[0]
        if self.configs.probabilistic:
            mu_sigma = [dec_out[0][:, -self.pred_len:, :],
                        dec_out[1][:, -self.pred_len:, :]]
            # return mu_sigma,
            if self.output_attention:
                return mu_sigma, attns
            else:
                return mu_sigma

        else:
            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]