from typing import Optional, Any
import math

import torch
import torch.nn as nn
import numpy as np
import copy
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from functools import lru_cache
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
import matplotlib.pyplot as plt
import os

from models.ClimaX.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from utils import visualization_utils

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
        if config['model'] == 'climax_smooth':
            return TSTEncoder(config['d_model'], config['d_model'], config['num_heads'],
                              d_ff=config['dim_feedforward'], dropout=config['dropout'],
                              activation=config['activation'], n_layers=config['num_layers'])
    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'climax_smooth':
            return ClimaX(list([feat_dim]), device=config['device'], img_size=list(data.feature_df.shape), max_seq_len=max_seq_len, patch_size=config['patch_length'],
                          stride=config['stride'], embed_dim=config['d_model'], depth=config['num_layers'], decoder_depth=config['num_decoder_layers'],
                          num_heads=config['num_heads'], feedforward_dim=config['dim_feedforward'],
                          drop_rate=config['dropout'],
                          activation=config['activation'],
                          norm=config['normalization_layer'],
                          num_classes=num_labels, freeze=config['freeze'],
                          pos_encoding=config['pos_encoding'],
                          relative_pos_encoding=config['relative_pos_encoding'],
                          agg_vars=config['agg_vars'],
                          conv_transformer=config['conv_transformer'],
                          where_to_add_relpos=config['where_to_add_relpos'],
                          conv_projection=config['conv_projection'],
                          local_mask=config['local_mask'])
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
        norm: BatchNorm or LayerNorm
        activation: gelu or relu
        pos_encoding: ABSOLUTE positional encoding method (fixed, learnable, learnable_sin_init, none)
        relative_pos_encoding: RELATIVE positional encoding method (erpe, alibi, none)
        agg_vars: whether to use cross-variable attention (if False, lumps all variables into one token)
        local_mask: if set to a positive number, only allow attention between tokens that are at most this distance apart. If set to -1, no restriction.
    """

    def __init__(
        self,
        default_vars,
        device,
        img_size=[32, 64],
        max_seq_len=1024,
        patch_size=2,
        stride=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        feedforward_dim=256,
        num_classes=0,
        freeze=False,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        norm='BatchNorm',
        activation='gelu',
        pos_encoding='learnable_init_sin',
        relative_pos_encoding='none',
        agg_vars=False,
        conv_transformer=False,
        where_to_add_relpos=False,
        conv_projection=False,
        local_mask=-1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.default_vars = default_vars
        self.max_len = max_seq_len
        self.pos_encoding = pos_encoding
        self.relative_pos_encoding = relative_pos_encoding
        self.agg_vars = agg_vars
        self.conv_transformer = conv_transformer
        self.device = device
        self.local_mask = local_mask
        self.where_to_add_relpos = where_to_add_relpos
        self.conv_projection = conv_projection

        # variable tokenization: separate embedding layer for each input variable

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables

        if self.agg_vars:
            self.embed_layer = nn.Linear(patch_size, embed_dim)

            # variable aggregation: a learnable query and a single-layer cross attention
            self.var_embed = self.create_var_embedding(embed_dim)
            self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            seq_len = int((max_seq_len - patch_size) / stride + 1)

        elif self.conv_transformer:
            # Convolutional encoder
            self.embed_layer = ConvEmbed(patch_size, img_size[1], embed_dim,
                                         stride, padding=int(np.ceil((patch_size-stride)/2)),  # Ensures that num_patches is num_timesteps/stride
                                         norm_layer=nn.BatchNorm1d)
            seq_len = int(max_seq_len // stride)
        else:
            self.embed_layer = nn.Linear(patch_size*img_size[1], embed_dim)  # each patch has patch_size*num_variables elements

            # Number of tokens (patches) in the time dimension
            seq_len = int((max_seq_len - patch_size) / stride + 1)
        self.seq_len = seq_len

        # positional embedding
        self.setup_posenc(pos_encoding, relative_pos_encoding, seq_len, embed_dim, num_heads)

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        # THIS IS NOT USED CURRENTLY
        # if norm == 'BatchNorm':
        #     norm_layer = nn.BatchNorm1d
        # elif norm == 'LayerNorm':
        #     norm_layer = nn.LayerNorm
        # else:
        #     raise ValueError("Unsupported norm layer")
        # activation = _get_activation_fn(activation)
        # self.blocks = nn.ModuleList(
        #     [
        #         Block(
        #             embed_dim,
        #             num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             drop_path=dpr[i],
        #             norm_layer=norm_layer,  # @joshuafan customized to allow different normalization layers
        #             attn_drop=drop_rate,  # @joshuafan changed: newest timm version does not have 'drop' parameter, only 'attn_drop' and 'proj_drop'
        #             proj_drop=drop_rate
        #         )
        #         for i in range(depth)
        #     ]
        # )
        # self.norm = nn.LayerNorm(embed_dim // 2)

        # --------------------------------------------------------------------------

        # # prediction head
        # self.head = nn.ModuleList()
        # for _ in range(decoder_depth):
        #     self.head.append(nn.Linear(embed_dim, embed_dim))
        #     self.head.append(nn.GELU())
        # self.head.append(nn.Linear(embed_dim, embed_dim // 2))
        # self.head = nn.Sequential(*self.head)

        if self.conv_transformer:
            encoder_layer = ConvTransformerBlock(embed_dim, num_heads, patch_size,
                                                 feedforward_dim, drop_rate * (1.0 - freeze))
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                embed_dim, num_heads, feedforward_dim, drop_rate * (1.0 - freeze), where_to_add_relpos=where_to_add_relpos, conv_projection=conv_projection)
        self.transformer_encoder = TransformerEncoder(encoder_layer, depth)
        # self.head_linear = nn.Linear(embed_dim, embed_dim // 2)

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(p=drop_rate)

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # final linear layer
        # self.output_layer = nn.Linear(embed_dim // 2 * int((max_seq_len - patch_size) / stride + 1), num_classes)
        self.output_layer = nn.Linear(embed_dim * seq_len, num_classes)

        # Local mask. Restrict which pairs of timesteps can pay attention to each other
        if self.local_mask >= 0:

            # Note that if relative positional encoding is being used, this is redundant with "relative_coords"
            indices = torch.arange(0, seq_len, device=device)
            distance_matrix = torch.abs(indices.reshape((1, -1)) - indices.reshape((-1, 1)))  # [seq_len, seq_len]
            self.invalid_mask = torch.zeros((len(indices), len(indices))).bool()  # [seq_len, seq_len]
            self.invalid_mask[distance_matrix > self.local_mask] = True
            print(self.invalid_mask)
        else:
            self.invalid_mask = None


    def setup_posenc(self, pos_encoding, relative_pos_encoding, seq_len, emb_dim, num_heads):

        # ABSOLUTE POSITION ENCODING: vector for each timestep, or matrix of size [seq_len, emb_dim]
        if pos_encoding == "learnable":
            # Simple learnable vector for each position
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, emb_dim), requires_grad=True)
            nn.init.uniform_(self.pos_embed, -0.02, 0.02)
        elif pos_encoding == "learnable_sin_init":
            # Simple learnable vector for each position, initialized with sinusoidal features
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, emb_dim), requires_grad=True)
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1],
                np.arange(seq_len)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        elif pos_encoding == "learnable_tape_init":
            # Simple learnable vector for each position, initialized with sinusoidal features (using TAPE trick to determine frequencies)
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, emb_dim), requires_grad=True)
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1],
                np.arange(seq_len),
                max_len = seq_len
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        elif pos_encoding == "fixed":
            # FIXED sinusoidal vector for each position
            # Code from Zerveas TST repo, https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py#L65
            scale_factor = 1.0  # Hardcode default value
            pe = torch.zeros(seq_len, emb_dim)  # positional encoding
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pos_embed = scale_factor * pe  #.unsqueeze(0).transpose(0, 1) do not need to create extra dimension
            self.register_buffer('pos_embed', pos_embed)  # this stores the variable in the state_dict (used for non-trainable variables)
        elif pos_encoding == "none":
            self.pos_embed = None
        else:
            raise ValueError("Invalid value of absolute_pos_embed (must be: learnable, learnable_sin_init, fixed, none)")

        # Helper function for ALIBI
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        # RELATIVE POSITION ENCODING: adjustment to the attention matrix that depends
        # only on the relative offset between two timesteps. This can be added to the
        # attention matrix before softmax or after softmax (see `where_to_add_relpos`)
        if relative_pos_encoding == "erpe":
            # eRPE: learnable bias for each relative offset between timesteps
            # Define a parameter table of relative position bias
            self.relative_bias_table = nn.Parameter(torch.zeros(2*seq_len-1, num_heads), requires_grad=True)  # Relative offsets range from (seq_len-1) to -(seq_len-1), inclusive

            # The attention matrix will have shape [time, time].
            # For entry (i, j), we want to look up the appropriate index in relative_bias_table,
            # which will be (i - j) + seq_len - 1. "relative_coords" does this lookup.
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [seq_len, seq_len]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0. Each entry (i, j) contains (i - j) + seq_len - 1
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "erpe_symmetric":
            # Same as eRPE, but offsets x and -x will now share a common offset.
            # In other words, the offset only depends on the absolute distance, not the sign.
            self.relative_bias_table = nn.Parameter(torch.zeros(seq_len, num_heads), requires_grad=True)  # Relative offsets range from (0) to (seq_len-1), inclusive

            # The attention matrix will have shape [time, time].
            # For entry (i, j), we want to look up the appropriate index in relative_bias_table,
            # which will be |i - j|. "relative_coords" does this lookup.
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = torch.abs(coords_t[:, None] - coords_t[None, :])  # [seq_len, seq_len]. Each entry (i, j) contains |i - j|
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "erpe_alibi_init":
            # Calculate initial bias table using ALIBI linear functions for each head.
            # Note that the linear function is multiplying "slope" with absolute |distance|.
            slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [num_heads]
            bias_table_init = torch.zeros(2*seq_len-1, num_heads)
            bias_table_init[0:seq_len-1] = torch.arange(start=seq_len-1, end=0, step=-1, device=self.device).unsqueeze(1) * slopes
            bias_table_init[seq_len-1:] = torch.arange(start=0, end=seq_len, device=self.device).unsqueeze(1) * slopes
            self.relative_bias_table = nn.Parameter(bias_table_init, requires_grad=True)  # Relative offsets range from (seq_len-1) to -(seq_len-1), inclusive

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [seq_len, seq_len]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)
        elif relative_pos_encoding == "custom_rpe":
            # Custom relative position encoding. Some heads will be initialized to behave like ConViT,
            # with attention focused on a specific offset and decaying from there. Here, the offset,
            # slope, and intercept are all learnable. Other heads will be randomly initialized and
            # fully learnable (like ERPE).
            convit_heads = num_heads // 2
            normal_heads = num_heads - convit_heads

            # Normal heads have purely learnable relative positional embeddings
            self.normal_biases = nn.Parameter(torch.zeros(2*seq_len-1, normal_heads), requires_grad=True)  # Relative offsets range from (seq_len-1) to -(seq_len-1), inclusive

            # Convit heads are initialized to focus attention around `convit_offsets`, with peak
            # intensity `convit_intercepts` and decay `convit_slopes`
            self.convit_slopes = nn.Parameter(torch.tensor([0.5 for i in range(convit_heads)], device=self.device), requires_grad=True)
            self.convit_intercepts = nn.Parameter(torch.zeros((convit_heads), device=self.device), requires_grad=True)
            self.convit_offsets = nn.Parameter(torch.tensor([-1 * (2.0 ** i) for i in range(convit_heads//2)] +
                                                            [2.0 ** i for i in range(convit_heads//2)], device=self.device), requires_grad=True)

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [seq_len, seq_len]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "alibi":
            # FIXED bias for relative offsets. Each head has a different function.
            # Code from https://github.com/ofirpress/attention_with_linear_biases/issues/5

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = torch.abs(coords_t[:, None] - coords_t[None, :])  # [seq_len, seq_len]. Each entry (i, j) contains ABS|i-j|. Note that this is different from the above eRPE approach.
            self.register_buffer("relative_coords", relative_coords)

            self.slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [num_heads]
            self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * self.relative_coords  # Broadcasting: [num_heads, 1, 1] * [seq_len, seq_len] -> [num_heads, seq_len, seq_len]


    def posenc_smoothness_loss(self, logger, plot_dir=None, epoch_num=None):
        file_prefix = "epoch{}".format(epoch_num) if epoch_num is not None else ""

        # Smoothness of absolute position encoding
        smoothness_loss = 0.
        if "learnable" in self.pos_encoding:
            # self.pos_embed has shape [seq_len, embed_dim]
            smoothness_loss += (torch.norm(self.pos_embed[1:, :] - self.pos_embed[:-1, :], dim=1)).mean()

            if plot_dir is not None:
                logger.info("Abs pos encoding smoothness: {}".format(smoothness_loss.item()))

                # Plot positional encoding
                im = plt.imshow(self.pos_embed.detach().cpu().numpy())
                plt.xlabel("Embedding index")
                plt.ylabel("Timestep")
                plt.colorbar(im)
                plt.title("Absolute positional embeddings")
                plt.savefig(os.path.join(plot_dir, f'{file_prefix}_absolute_pos_encoding.png'))
                plt.close()

                # Compute pairwise distance between each pair of positions
                pairwise_distances = torch.cdist(self.pos_embed.unsqueeze(0), self.pos_embed.unsqueeze(0)).squeeze(0)
                im = plt.imshow(pairwise_distances.detach().cpu().numpy())
                plt.colorbar(im)
                plt.title("Pairwise distances between absolute pos encodings")
                plt.savefig(os.path.join(plot_dir, f'{file_prefix}_absolute_pos_encoding_distances.png'))
                plt.close()

        if "erpe" in self.relative_pos_encoding or self.relative_pos_encoding == "custom_rpe":
            # self.relative_bias_table has shape [2*seq_len-1, num_heads]
            rel_smoothness = ((self.relative_bias_table[1:, :] - self.relative_bias_table[:-1, :]) ** 2).mean()
            smoothness_loss += rel_smoothness

            if plot_dir is not None:
                logger.info("Rel pos encoding smoothness: {}".format(rel_smoothness.item()))
                im = plt.imshow(self.relative_bias_table.detach().cpu().numpy(), aspect=0.2, interpolation='none')  # stretch each column horizontally 5x
                plt.xlabel("Head number")
                plt.ylabel("Relative offset (middle is 0)")
                plt.colorbar(im)
                plt.title("Relative attention biases")
                plt.savefig(os.path.join(plot_dir, f'{file_prefix}_relative_pos_offsets.png'))
                plt.close()

        return smoothness_loss



    def initialize_weights(self):
        # NOTE: pos_embed initialization is moved to setup_pos_embed

        # var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        # self.var_embed.data.copy_(torch.from_numpy(var_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        # number of variables x embedding dim
        var_embed = nn.Parameter(torch.zeros(self.img_size[1], dim), requires_grad=True)
        return var_embed

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, D
        # query, key, value

        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D

        return x

    def forward_encoder(self, x: torch.Tensor, plot_dir: str = None):
        # x: `[B, T, V]` shape.

        if plot_dir is not None:
            # Plot an example input and distances between timesteps (just for comparison with later)
            x_detached = x.detach().cpu()
            visualization_utils.plot_time_series(x_detached[0, :, :].numpy(), os.path.join(plot_dir, 'example_x0.png'))
            n_rows = 5  # Examples to plot
            n_cols = 1

            # Plot Euclidean distance between timestep feature vectors
            feature_distances = torch.linalg.norm(x_detached.unsqueeze(1) - x_detached.unsqueeze(2), dim=3)  # [batch, time, time]
            # feature_distances_manual = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=x.device)
            # for i in range(x.shape[0]):
            #     for j in range(x.shape[1]):
            #         for k in range(x.shape[1]):
            #             feature_distances_manual[i, j, k] = torch.linalg.norm(x[i, j, :] - x[i, k, :])
            # assert torch.allclose(feature_distances, feature_distances_manual)

            min_value, max_value = torch.min(feature_distances), torch.max(feature_distances)
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
            for r in range(n_rows):
                im = axeslist[r].imshow(feature_distances[r, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)
                if r == 0:
                    axeslist[r].set_title(f"Input")
            plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
            plt.colorbar(im)
            plt.suptitle("Distance between timestep INPUTS")
            plt.savefig(os.path.join(plot_dir, 'timestep_input_distances.png'))
            plt.close()

        if self.agg_vars:
            # tokenize each variable separately
            x = x.permute(0, 2, 1) # B, V, T
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # B, V, L, D
            x = self.embed_layer(x)

            # add variable embedding
            var_embed = self.var_embed # V, D
            var_embed = var_embed.unsqueeze(0).unsqueeze(2) # 1, V, 1, D
            x = x + var_embed  # B, V, L, D

            # variable aggregation
            x = self.aggregate_variables(x)  # B, L, D

        elif self.conv_transformer:
            x = x.permute(0, 2, 1) # B, V, T  [batch, channel, time (num_patches)]
            x = self.embed_layer(x)  # [batch, embed_dim, num_patches]
            x = x.permute(0, 2, 1)  # [batch, num_patches (seq_len), embed_dim]
        else:
            x = x.permute(0, 2, 1) # B, V, T
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # B, V, num_patches, patch_size
            x = x.permute(0, 2, 1, 3)  # B, num_patches, V, patch_size
            x = x.reshape((x.shape[0], x.shape[1], -1))  # B, num_patches, V*patch_size
            x = self.embed_layer(x)

        # Add ABSOLUTE pos embedding if using.
        # At this point, X should be [batch, seq_len, embed_dim], and pos_embed should be [seq_len, embed_dim]. (seq_len = number of patches along time dimension)
        if self.pos_embed is not None:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # apply Transformer blocks. NOT USED ANYMORE
        # for blk in self.blocks:
        #     x = blk(x)

        # Construct mask for relative positional encoding.
        offset_mask = None
        if self.relative_pos_encoding in ["erpe", "erpe_symmetric"]:  # erpe_symmetric can use the same code since we set "relative_coords" to use absolute distances
            # self.relative_bias_table: [2*seq_len-1, num_heads]
            # self.relative_coords: [seq_len, seq_len] - ID of offset between timesteps
            # To construct relative embedding matrix, flatten the "offset matrix" (relative_coords),
            # and use these as indices into the relative bias table.
            # Then reshape to construct the real bias matrix (same shape as attention matrix)
            num_heads = self.relative_bias_table.shape[1]
            flattened_indices = self.relative_coords.flatten()  # [seq_len*seq_len]
            offset_mask = self.relative_bias_table.index_select(dim=0, index=flattened_indices).reshape(self.seq_len, self.seq_len, num_heads)  # [seq_len, seq_len, heads]
            offset_mask = offset_mask.permute(2, 0, 1).repeat((x.shape[0], 1, 1))  # [batch*num_heads, seq_len, seq_len]
        elif self.relative_pos_encoding == "custom_rpe":
            # Set up bias table
            convit_biases = -1.0 * self.convit_slopes * torch.abs(torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(1) - (self.seq_len-1+self.convit_offsets)) + self.convit_intercepts # Distance to "focus pixel", [2*seq_len-1, num_convit_heads]
            bias_table = torch.cat([self.normal_biases, convit_biases], dim=1)
            self.relative_bias_table = bias_table

            # Compute the actual offset matrix
            num_heads = self.relative_bias_table.shape[1]
            flattened_indices = self.relative_coords.flatten()  # [seq_len*seq_len]
            offset_mask = self.relative_bias_table.index_select(dim=0, index=flattened_indices).reshape(self.seq_len, self.seq_len, num_heads)  # [seq_len, seq_len, heads]
            offset_mask = offset_mask.permute(2, 0, 1).repeat((x.shape[0], 1, 1))  # [batch*num_heads, seq_len, seq_len]

        elif self.relative_pos_encoding == "alibi":
            offset_mask = self.alibi.repeat((x.shape[0], 1, 1))  # Repeat along the batch dimension, as PyTorch expects mask to be [batch*num_heads, seq_len, seq_len]

        # If some positions are not allowed to attend, either use the Boolean mask, or if combining with
        # relative position encoding, set those mask entries to -inf
        if self.invalid_mask is not None:
            if offset_mask is None:
                offset_mask = self.invalid_mask  # True at positions that are NOT ALLOWED to attend (too far)
            else:
                offset_mask[self.invalid_mask] = float("-inf")

        x, attn_weights = self.transformer_encoder(x, mask=offset_mask, plot_dir=plot_dir)  # after encoder. x: [batch, seq_len, embed_dim]. attn_weights: [batch, n_layer*n_head, seq_len, seq_len]
        # x = self.head_linear(x)
        # x = self.norm(x)

        return x, attn_weights

    def forward(self, x, plot_dir=None):
        """Forward pass through the model.

        Args:
            x: `[batch_size, seq_length, feat_dim]` shape.
            plot_dir: if provided, plot attention matrices and distances between timestep feature vectors at each layer.
        Returns:
            preds (torch.Tensor): `[B]` shape. Predicted output.
        """
        preds, attn_weights = self.forward_encoder(x, plot_dir=plot_dir)
        preds = self.act(preds)
        preds = self.dropout1(preds)
        preds = preds.reshape(preds.shape[0], -1)
        preds = self.output_layer(preds)
        return preds, attn_weights

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.modules.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, plot_dir: str = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required). Must be of shape [BATCH, TIME, EMBED_DIM]
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            plot_dir: if provided, plot attention matrices and distances between timestep feature vectors

        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        attn_weights_layers = None

        # FOR VISUALIZATIONS: Compute Euclidean distance between each timestep's feature vectors.
        # "output" is assumed to be shape [batch, seq_len, embed_dim].
        # Via broadcasting, we convert this to [batch, seq_len, seq_len, embed_dim], then take the norm over embed_dim.
        # Result is [batch, seq_len, seq_len]
        feat_detached = output.detach().cpu()
        feature_distances_layers = [torch.linalg.norm(feat_detached.unsqueeze(1) - feat_detached.unsqueeze(2), dim=3)]

        # FOR VISUALIZATIONS: Compute cosine similarity between each timestep's feature vectors.
        # "output" is assumed to be shape [batch, seq_len, embed_dim].
        # Via broadcasting, we convert this to [batch, seq_len, seq_len, embed_dim], then compute similarity over embed_dim.
        # Result is [batch, seq_len, seq_len]
        # See https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
        similarity_matrix_layers = [F.cosine_similarity(feat_detached.unsqueeze(1), feat_detached.unsqueeze(2), dim=3)]

        # Compute forward pass
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers, plot_dir=plot_dir)  # output: [batch, seq_len, embed_dim], attn_weights: [batch, num_heads, seq_len, seq_len]
            if attn_weights_layers is None:
              attn_weights_layers = attn_weights
            else:
              attn_weights_layers = torch.cat((attn_weights_layers, attn_weights), dim=1)  # attn_weights: [batch, n_layers*num_heads, seq_len, seq_len]

            # DEBUGGING: Compute distances between timestep feature vectors
            feat_detached = output.detach().cpu()
            feature_distances_layers.append(torch.linalg.norm(feat_detached.unsqueeze(1) - feat_detached.unsqueeze(2), dim=3))
            similarity_matrix_layers.append(F.cosine_similarity(feat_detached.unsqueeze(1), feat_detached.unsqueeze(2), dim=3))

        # VISUALIZATIONS
        if plot_dir is not None:
            n_rows = 5  # Examples to plot

            # Plot Euclidean distance between timestep feature vectors
            n_cols = len(feature_distances_layers)
            feature_distances_layers = torch.stack(feature_distances_layers, dim=1)  # Convert this to similar format as attn_weights_layers [batch, n_matrices, seq_len, seq_len]
            min_value, max_value = torch.min(feature_distances_layers), torch.max(feature_distances_layers)
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
            for r in range(n_rows):
                for c in range(n_cols):
                    im = axeslist[r, c].imshow(feature_distances_layers[r, c, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)
                    if r == 0:
                        if c < n_cols-1:
                            axeslist[r, c].set_title(f"Before layer {c+1}")
                        else:
                            axeslist[r, c].set_title(f"Final")
            plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
            plt.colorbar(im)
            plt.suptitle("Distance between timestep feature vectors")
            plt.savefig(os.path.join(plot_dir, 'timestep_distances.png'))
            plt.close()

            # Plot cosine similarity between timestep feature vectors
            n_cols = len(similarity_matrix_layers)
            similarity_matrix_layers = torch.stack(similarity_matrix_layers, dim=1)  # Convert this to similar format as attn_weights_layers [batch, n_matrices, seq_len, seq_len]
            min_value, max_value = torch.min(similarity_matrix_layers), torch.max(similarity_matrix_layers)
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
            for r in range(n_rows):
                for c in range(n_cols):
                    im = axeslist[r, c].imshow(similarity_matrix_layers[r, c, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)
                    if r == 0:
                        if c < n_cols-1:
                            axeslist[r, c].set_title(f"Before layer {c+1}")
                        else:
                            axeslist[r, c].set_title(f"Final")
            plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
            plt.colorbar(im)
            plt.suptitle("Cos similarity between timestep feature vectors")
            plt.savefig(os.path.join(plot_dir, 'timestep_similarities.png'))
            plt.close()

            # Plot attention matrices
            n_matrices = attn_weights_layers.shape[1]  # Total number of attention maps per example (n_layers*n_heads)
            n_cols = n_matrices//4
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))

            for r in range(n_rows):
                for c in range(n_cols):
                    im = axeslist[r, c].imshow(attn_weights_layers[r, c*(n_matrices//n_cols), :, :].detach().cpu().numpy(), vmin=0, vmax=3/attn_weights_layers.shape[2])  #0/attn_weights_layers.shape[1])
            plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
            plt.colorbar(im)
            plt.suptitle("Example attention matrices")
            plt.savefig(os.path.join(plot_dir, 'attention_matrices.png'))
            plt.close()

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights_layers

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, where_to_add_relpos=False, conv_projection=False):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        if where_to_add_relpos == "before":
            # Note: we could also use Attention_Rel_Scl here. TODO - check that they behave the same way
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            assert conv_projection == False, "conv_projection is only supported for custom attention (Attention_Rel_Scl)"
        else:
            # Custom attention if we want relative position offset to be applied after softmax
            self.self_attn = Attention_Rel_Scl(d_model, nhead, dropout=dropout, conv_projection=conv_projection, where_to_add_relpos=where_to_add_relpos)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # normalizes each feature across batch samples and time steps
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, plot_dir = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required). Shape: [batch, seq_len, embed_dim]
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if type(self.self_attn) == Attention_Rel_Scl:
            # Attention_Rel_Scl allows plot_dir
            src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask, average_attn_weights=False, plot_dir=plot_dir)  # src2: [batch, seq_len, d_model], attn_output_weights: [batch, num_heads, seq_len, seq_len]
        else:
            src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask, average_attn_weights=False)  # src2: [batch, seq_len, d_model], attn_output_weights: [batch, num_heads, seq_len, seq_len]

        src = src + self.dropout1(src2)  # (batch, seq_len, d_model)  TODO temporarily removed
        src = src.permute(0, 2, 1)  # (batch, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(0, 2, 1)  # restore (batch, seq_len, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (batch, seq_len, d_model)
        src = src.permute(0, 2, 1)  # (batch, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(0, 2, 1)  # restore (batch, seq_len, d_model)
        return src, attn_output_weights




# ========================================================================================
# Code from ConvTran: https://github.com/Navidfoumani/ConvTran/blob/main/Models/Attention.py
# except that the relative bias table isn't stored here, we pass it as a mask instead.
# Note that the attention bias is added after softmax, and we further use a gating param to weight them.
# ========================================================================================
class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, conv_projection, where_to_add_relpos, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.conv_projection = conv_projection
        self.where_to_add_relpos = where_to_add_relpos
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        if conv_projection:
            self.key = nn.Sequential(OrderedDict([
                ('rearrange_to_conv', Rearrange('b t c -> b c t')),
                ('conv', nn.Conv1d(emb_size, emb_size, kernel_size=5, padding=2,stride=1, bias=False, groups=emb_size)),
                ('rearrange_from_conv', Rearrange('b c t -> b t c')),
                ('relu', nn.ReLU()),
                ('linear', nn.Linear(emb_size, emb_size, bias=False))
            ]))
            self.value = nn.Sequential(OrderedDict([
                ('rearrange_to_conv', Rearrange('b t c -> b c t')),
                ('conv', nn.Conv1d(emb_size, emb_size, kernel_size=5, padding=2,stride=1, bias=False, groups=emb_size)),
                ('rearrange_from_conv', Rearrange('b c t -> b t c')),
                ('relu', nn.ReLU()),
                ('linear', nn.Linear(emb_size, emb_size, bias=False))
            ]))
            self.query = nn.Sequential(OrderedDict([
                ('rearrange_to_conv', Rearrange('b t c -> b c t')),
                ('conv', nn.Conv1d(emb_size, emb_size, kernel_size=5, padding=2,stride=1, bias=False, groups=emb_size)),
                ('rearrange_from_conv', Rearrange('b c t -> b t c')),
                ('relu', nn.ReLU()),
                ('linear', nn.Linear(emb_size, emb_size, bias=False))
            ]))
        else:
            self.key = nn.Linear(emb_size, emb_size, bias=False)
            self.value = nn.Linear(emb_size, emb_size, bias=False)
            self.query = nn.Linear(emb_size, emb_size, bias=False)
            self.key.weight.data.copy_(torch.eye(emb_size))
            self.value.weight.data.copy_(torch.eye(emb_size))
            self.query.weight.data.copy_(torch.eye(emb_size))

            # torch.nn.init.xavier_uniform_(self.key.weight)
            # torch.nn.init.xavier_uniform_(self.value.weight)
            # torch.nn.init.xavier_uniform_(self.query.weight)

        self.dropout = nn.Dropout(dropout)
        self.gating_param = nn.Parameter(torch.zeros(num_heads), requires_grad=True)  # nn.Parameter(torch.cat([-1*torch.ones(num_heads//2), torch.ones(num_heads//2)]))
        # self.to_out = nn.LayerNorm(emb_size)

    def forward(self, query, key, value, attn_mask, plot_dir=None, **kwargs):
        """
        Input (query/key/value) should be [batch, seq_len, embed_dim]. They can be identical
        as the linear projections happen inside the method.
        Mask should be [batch_size*num_heads, seq_len, seq_len]

        Output is [batch, seq_len, embed_dim], and attn matrix [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        # return torch.zeros_like(query), torch.eye((seq_len)).unsqueeze(0).unsqueeze(0).repeat((batch_size, self.num_heads, 1, 1))

        k = self.key(key).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(value).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(query).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k shape = (batch_size, num_heads, d_head, seq_len)
        # v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale  # attn shape (batch_size, num_heads, seq_len, seq_len)

        # Add mask (relative position encoding) before softmax if specified
        if self.where_to_add_relpos == 'before' and attn_mask is not None:
            # Reshape attn_mask to (batch_size, num_heads, seq_len, seq_len)
            attn_mask = rearrange(attn_mask, '(b h) l t -> b h l t', h=self.num_heads)
            attn += attn_mask

        # Perform softmax
        attn = nn.functional.softmax(attn, dim=-1)
        # print("Init attn", attn[0, 0, 0:10, 0:10])
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, '(b h) l t -> b h l t', h=self.num_heads)
            # print("Attn mask", attn_mask[0, 0:8, 0:8])
            content_attn = attn
            if self.where_to_add_relpos == 'after':
                attn = content_attn + attn_mask
            elif self.where_to_add_relpos == "after_gating":
                # print("Gating (Pr position)", torch.sigmoid(self.gating_param))
                gating = self.gating_param.view(1,-1,1,1)
                attn = (1.-torch.sigmoid(gating))*content_attn + torch.sigmoid(gating)*F.softmax(attn_mask, dim=-1)  # First term is original content attention, second term is position attention
                attn /= attn.sum(dim=-1).unsqueeze(-1)

            if plot_dir is not None:
                # PLOTTING ONLY
                # Plot attention breakdown (content/position) for a single example, 'n_rows' heads
                n_rows = 4
                n_cols = 3
                fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))

                for r in range(n_rows):
                    head_num = r * (attn.shape[1] // n_rows)
                    max_value = 0.1 #/attn.shape[2]
                    content_attn_head = content_attn[0, head_num, :, :]
                    pos_attn_head = F.softmax(attn_mask, dim=-1)[0, head_num, :, :]
                    total_attn_head = attn[0, head_num, :, :]
                    axeslist[r, 0].imshow(content_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    axeslist[r, 1].imshow(pos_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    im = axeslist[r, 2].imshow(total_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    if r == 0:
                        axeslist[r, 0].set_title("Content attn")
                        axeslist[r, 1].set_title("Position attn")
                        axeslist[r, 2].set_title("Combined attn")
                plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
                plt.colorbar(im)
                plt.suptitle("Attn breakdown, single example (each row is one head)")
                plt.savefig(os.path.join(plot_dir, 'attention_breakdown.png'))
                plt.close()

        # print("Final attn", attn[0, 0, 0:10, 0:10])
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        # out = self.to_out(out)
        # print("Out", out[0, 0:10, :])
        return out, attn


# ========================================================================
# Below code is from CvT: https://github.com/leoxiaobin/CvT/blob/main/lib/models/cls_cvt.py
# ========================================================================
from functools import partial
from itertools import repeat
# from torch._six import container_abcs

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='linear',  # 'dw_bn', TODO @joshuafan changed to make this similar to original to full transformer
                 kernel_size=3,
                 stride=1,
                 padding="same"
                 ):
        super().__init__()
        self.stride = stride
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5

        # Decide how much to use positional vs content attention
        init_gating = torch.ones(self.num_heads)*2 # Second half of heads prefer position attention
        init_gating[0:self.num_heads//2] = -2  # First half of heads prefer content attentio
        self.gating_param = nn.Parameter(init_gating, requires_grad=True)

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding,
            stride, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding,
            stride, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding,
            stride, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm1d(dim_in)),
                ('rearrage', Rearrange('b c t -> b t c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool1d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c t -> b t c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x):
        """
        Input/output are assumed to be [batch, seq_len, embed_dim] or [b, t, c]
        """
        x = rearrange(x, 'b t c -> b c t')

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c t -> b t c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c t -> b t c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c t -> b t c')

        return q, k, v

    def forward(self, x, src_mask=None):
        """Input/output assumed to be [batch, seq_len, embed_dim]
        mask should be [batch*num_heads, seq_len, seq_len]"""
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x)  # [batch, seq_len, embed_dim]
        else:
            q, k, v = x, x, x

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        # if src_mask is not None:
        #     attn_score = attn_score + rearrange(src_mask, '(b h) l t -> b h l t', h=self.num_heads)

        # Add mask (relative position encoding)
        attn = F.softmax(attn_score, dim=-1)

        if src_mask is not None:
            print("Gating (Pr position)", torch.sigmoid(self.gating_param))
            gating = self.gating_param.view(1,-1,1,1)
            src_mask = rearrange(src_mask, '(b h) l t -> b h l t', h=self.num_heads)
            attn = (1.-torch.sigmoid(gating))*attn + torch.sigmoid(gating)*F.softmax(src_mask, dim=-1)  # First term is original content attention, second term is position attention
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')  # [batch, seq_len, embed_dim]

        x = self.proj(x)
        x = self.proj_drop(x)  # [batch, seq_len, embed_dim]
        return x, attn


class ConvTransformerBlock(nn.Module):
    """
    Should be exact replacement for TransformerEncoderBatchNormLayer.
    """
    def __init__(self,
                 d_model,
                 n_head,
                 kernel_size,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.attn = Attention(
            d_model, d_model, n_head, attn_drop=dropout, proj_drop=dropout,
            kernel_size=kernel_size  # stride=stride, padding=padding,
        )

        self.drop_path = DropPath(dropout) \
            if dropout > 0. else nn.Identity()
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            drop=dropout
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Input/output: [batch, seq_len (TIME), embed_dim] or [B, T, D]
        """
        res = x

        # Change shapes just for BatchNorm1d, then back
        x = x.permute((0, 2, 1))  # [batch, embed_dim, seq_len]
        x = self.norm1(x)
        x = x.permute((0, 2, 1))  # [batch, seq_len, embed_dim]

        x, attn = self.attn(x, src_mask=src_mask)  # [batch, seq_len, embed_dim]
        x = res + self.drop_path(x)

        # Change shapes just for BatchNorm1d, then back
        x = x.permute((0, 2, 1))  # [batch, embed_dim, seq_len]
        x = self.norm2(x)
        x = x.permute((0, 2, 1))  # [batch, seq_len, embed_dim]

        x = x + self.drop_path(self.mlp(x))
        return x, attn


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv1d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        """
        Input [batch, in_chans, time] or [B, D, T]
        Output [batch, embed_dim, time] or [B, D, T]
        """
        x = self.proj(x)

        if self.norm:
            x = self.norm(x)
        return x
