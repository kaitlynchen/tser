from typing import Optional, Any
import math

import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from functools import lru_cache
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from models.ClimaX.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        if config['model'] == 'climax':
            return TSTEncoder(config['d_model'], config['d_model'], config['num_heads'], 
                              d_ff=config['dim_feedforward'], dropout=config['dropout'],
                              activation=config['activation'], n_layers=config['num_layers'])
    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'climax':
            return ClimaX(list([feat_dim]), img_size=list(data.feature_df.shape), max_seq_len=max_seq_len, patch_size=config['patch_length'],
                          stride=config['stride'], embed_dim=config['d_model'], depth=config['num_layers'], decoder_depth=config['num_decoder_layers'],
                          num_heads=config['num_heads'], num_classes=num_labels)
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))

def _get_activation_fn(activation):
      if activation == "relu":
          return F.relu
      elif activation == "gelu":
          return F.gelu
      raise ValueError(
          "activation should be relu/gelu, not {}".format(activation))
    
class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        max_seq_len=1024,
        patch_size=2,
        stride=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        num_classes=0,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        activation='gelu',
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.default_vars = default_vars
        self.max_len = max_seq_len
        # variable tokenization: separate embedding layer for each input variable

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed = self.create_var_embedding(embed_dim)
        self.embed_layer = nn.Linear(patch_size, embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        # TODO: consider relative positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(int((max_seq_len - patch_size) / stride + 1), embed_dim), requires_grad=True)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * patch_size - 1), num_heads)) 

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, embed_dim // 2))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(p=drop_rate)

        # final linear layer
        self.output_layer = nn.Linear(embed_dim // 2 * int((max_seq_len - patch_size) / stride + 1), num_classes)

        self.linear_before_pool = nn.Linear(embed_dim, embed_dim)
        final_emb_dim = embed_dim * max_seq_len
        self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, final_emb_dim),
                nn.ReLU(),
                nn.Linear(final_emb_dim, num_heads)
        )
        self.fc = nn.Linear(embed_dim * num_heads, num_classes)

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1],
            np.arange(int((self.max_len - self.patch_size) / self.stride + 1))
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        # number of variables x embedding dim
        var_embed = nn.Parameter(torch.zeros(self.img_size[1], dim), requires_grad=True)
        return var_embed

    # def unpatchify(self, x: torch.Tensor, h=None, w=None):
    #     """
    #     x: (B, L, V * patch_size**2)
    #     return imgs: (B, V, H, W)
    #     """
    #     p = self.patch_size
    #     c = len(self.default_vars)
    #     h = self.img_size[0] // p if h is None else h // p
    #     w = self.img_size[1] // p if w is None else w // p
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum("nhwpqc->nchpwq", x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    #     return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, T, V]` shape.

        # tokenize each variable separately
        x = x.permute(0, 2, 1) # B, V, T
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # B, V, L, P
        x = self.embed_layer(x)

        # add variable embedding
        var_embed = self.var_embed # V, D
        var_embed = var_embed.unsqueeze(0).unsqueeze(2) # 1, V, 1, D
        x = x + var_embed  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_pooling(self, x, plot_dir=None):
        x = self.linear_before_pool(x)
        attn_weights = self.attention_pool(x)  # [batch, time, n_heads]
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch, time, n_heads]
        attn_weights = attn_weights.permute((0, 2, 1))  # [batch, n_heads, time]
        aggregated_x = torch.matmul(attn_weights, x)  # [batch, n_heads, channel]
        aggregated_x = aggregated_x.reshape((aggregated_x.shape[0], -1))  # [batch, n_heads*channel]
        x = self.fc(aggregated_x)

        return x

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[batch_size, seq_length, feat_dim]` shape. 
        Returns:
            preds (torch.Tensor): `[B]` shape. Predicted output.
        """
        preds = self.forward_encoder(x)
        preds = self.act(preds)
        preds = self.dropout1(preds)
        preds = self.forward_pooling(preds)    

        return preds