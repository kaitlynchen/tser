"""
Code for the main models we are testing. Includes original MVTS Transformer, along with:
- Patching
- Variable aggregation
- Relative positional embeddings (offset biases)
- Absolute positional embeddings
- SeqPool + positional embeddings there

We use the following letters to annotate shapes:
B: batch (examples)
T_orig: timesteps in original data
T: timesteps after initial patching (i.e. number of patches)
P: patch size (timesteps in a single patch)
V: number of variables ("channels") in original data
D: 'channels' or embedding dimension for each timestep
H: number of heads
L: number of layers

"""


from typing import Optional, Any
import math

from models.ts_climax_rope import AttentionWithRoPE
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

torch.cuda.empty_cache()

from models.ClimaX.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from utils import visualization_utils, utils

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
                          where_to_add_abspos=config['where_to_add_abspos'],
                          relative_pos_encoding=config['relative_pos_encoding'],
                          where_to_add_relpos=config['where_to_add_relpos'],
                          agg_vars=config['agg_vars'],
                          conv_transformer=config['conv_transformer'],
                          conv_projection=config['conv_projection'],
                          local_mask=config['local_mask'],
                          pool=config['pool'])
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
        pos_encoding='learnable_random_init',
        where_to_add_abspos='start_add',
        relative_pos_encoding='none',
        where_to_add_relpos='before',
        agg_vars=False,
        conv_transformer=False,
        conv_projection=False,
        local_mask=-1,
        pool="linear",
    ):
        super().__init__()

        self.img_size = img_size  # Should be [n_examples*T_orig, V]
        self.patch_size = patch_size
        self.stride = stride
        self.default_vars = default_vars
        self.max_len = max_seq_len
        self.num_layers = depth
        self.num_heads = num_heads
        self.pos_encoding = pos_encoding
        self.where_to_add_abspos = where_to_add_abspos
        self.relative_pos_encoding = relative_pos_encoding
        self.where_to_add_relpos = where_to_add_relpos
        self.agg_vars = agg_vars
        self.conv_transformer = conv_transformer
        self.device = device
        self.local_mask = local_mask
        self.conv_projection = conv_projection
        self.pool = pool

        if self.agg_vars:
            # Variable tokenization: create tokens for each variable, of size "patch_size"
            # Here we use the same embedding layer for all variables.
            # TODO: In Climax code, I think they use separate embedding layers for each input?
            # https://github.com/microsoft/ClimaX/blob/main/src/climax/arch.py 
            self.embed_layer = nn.Linear(patch_size, embed_dim)

            # Variable embedding to denote which variable each token belongs to
            # helps in aggregating variables
            self.var_embed = self.create_var_embedding(embed_dim)  # [V, D]

            # variable aggregation: a learnable query and a single-layer cross attention
            self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # [1, 1, D]
            self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            seq_len = int((max_seq_len - patch_size) / stride + 1)

        elif self.conv_transformer:
            # Convolutional encoder
            self.embed_layer = ConvEmbed(patch_size, img_size[1], embed_dim,
                                         stride, padding=int(np.ceil((patch_size-stride)/2)),  # Ensures that num_patches (T) is num_timesteps/stride
                                         norm_layer=nn.BatchNorm1d)
            seq_len = int(max_seq_len // stride)
        else:
            self.embed_layer = nn.Linear(patch_size*img_size[1], embed_dim)  # Each patch has patch_size*num_variables (P*V) elements. Map to embed_dim (D).

            # Number of tokens (patches) in the time dimension (T)
            seq_len = int((max_seq_len - patch_size) / stride + 1)
        self.seq_len = seq_len

        # Positional embedding
        # Note: if absolute positional embedding is added inside the pooling attention,
        # the embedding size is equal to the number of heads. Otherwise it's the normal embedding dim.
        if where_to_add_abspos in ["pooling_before_softmax", "pooling_gating"]:
            if self.pool == "seqpool":
                absolute_emb_dim = 1
            else:
                absolute_emb_dim = num_heads
        else:
            absolute_emb_dim = embed_dim
        utils.setup_absolute_posenc(self, pos_encoding, seq_len, absolute_emb_dim)
        self.setup_relative_posenc(relative_pos_encoding, seq_len, self.num_layers, num_heads)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Define single encoder layer
        if self.conv_transformer:
            encoder_layer = ConvTransformerBlock(embed_dim, num_heads, patch_size,
                                                 feedforward_dim, drop_rate * (1.0 - freeze))
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                embed_dim, num_heads, feedforward_dim, drop_rate * (1.0 - freeze), where_to_add_relpos=where_to_add_relpos, conv_projection=conv_projection)
        
        # Create TransformerEncoder with multiple layers
        self.transformer_encoder = TransformerEncoder(encoder_layer, depth)

        # Activation/dropout
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(p=drop_rate)

        # TODO: Try not initializing weights and stick with default (Kaiming uniform)?
        self.initialize_weights()

        # Initialize pooling
        utils.setup_pooling(self, embed_dim, num_heads, seq_len, num_classes)

        # Local mask. Restrict which pairs of timesteps can pay attention to each other
        if self.local_mask >= 0:
            # Note that if relative positional encoding is being used, this is redundant with "relative_coords"
            indices = torch.arange(0, seq_len, device=device)  # [T]
            distance_matrix = torch.abs(indices.reshape((1, -1)) - indices.reshape((-1, 1)))  # [T, T]
            self.invalid_mask = torch.zeros((len(indices), len(indices)), device=self.device).bool()  # [T, T]
            self.invalid_mask[distance_matrix > self.local_mask] = True
        else:
            self.invalid_mask = None


    def setup_relative_posenc(self, relative_pos_encoding, seq_len, num_layers, num_heads):
        """
        Setup relative positional encoding. Typically, for each layer we have
        a matrix of biases, with a value for each of the 2T-1 relative offsets
        between timesteps (and for each layer/head). Shape: [L (layers), 2T-1, H (heads)].

        The entire matrix is usually a learnable parameter (unless relative_pos_encoding
        is 'convit_half' or 'convit', for which we only allow 'linear decays from some peak', 
        with the slope/peak being learnable).
        """
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
        if "erpe" in relative_pos_encoding:
            # Bias table for each relative offset.
            # Relative offsets range from (T-1) to -(T-1), inclusive.
            # Thus, the table has shape [L, 2T-1, H] (since we have a bias for each layer and head).
            # Consider different initializations.
            if relative_pos_encoding == "erpe":
                bias_table_init = torch.zeros(num_layers, 2*seq_len-1, num_heads)
            elif relative_pos_encoding == "erpe_uniform_init":
                bias_table_init = torch.zeros(num_layers, 2*seq_len-1, num_heads)
                nn.init.uniform_(bias_table_init, -0.02, 0.02)
            elif relative_pos_encoding == "erpe_alibi_init":
                # Calculate initial bias table using ALIBI linear functions for each head.
                # Note that the linear function is multiplying "slope" with absolute |distance|.
                slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [H]
                bias_table_init = torch.zeros(2*seq_len-1, num_heads)
                bias_table_init[0:seq_len-1] = torch.arange(start=seq_len-1, end=0, step=-1, device=self.device).unsqueeze(1) * slopes
                bias_table_init[seq_len-1:] = torch.arange(start=0, end=seq_len, device=self.device).unsqueeze(1) * slopes

                # Duplicate for each layer
                bias_table_init = bias_table_init.repeat(num_layers, 1, 1)  # [L, 2T-1, H]
            elif relative_pos_encoding == "erpe_convit_init":
                # Calculate initial bias with CONVIT linear decays for half of heads (remaining are zero init).
                bias_table_init = torch.zeros(2*seq_len-1, num_heads)
                self.convit_heads = num_heads  # (num_heads // 2) + 1
                self.convit_slopes = torch.tensor([1.0 for i in range(self.convit_heads)], device=self.device)
                self.convit_intercepts = torch.zeros((self.convit_heads), device=self.device)
                self.convit_offsets = torch.tensor([0] + [-1 * (2.0 ** i) for i in range(self.convit_heads//2)] +
                                          [2.0 ** i for i in range(self.convit_heads//2 - 1)], device=self.device)
                # self.convit_offsets = torch.tensor([0] + [-1 * (3.0 ** i) for i in range(self.convit_heads//2)] +
                #                                 [3.0 ** i for i in range(self.convit_heads//2)], device=self.device)
                print("Convit offsets", self.convit_offsets, self.convit_slopes, self.convit_intercepts)
                convit_biases = -1.0 * self.convit_slopes * torch.abs(torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(1) - (self.seq_len-1+self.convit_offsets)) + self.convit_intercepts  # Distance to "focus pixel", [2T-1, H]
                bias_table_init[:, :self.convit_heads] = convit_biases
                # bias_table_init = torch.clamp(bias_table_init, min=-5)

                # Duplicate for each layer
                bias_table_init = bias_table_init.repeat(num_layers, 1, 1)  # [L, 2T-1, H]
            
            elif relative_pos_encoding == "erpe_convalibi_init":
                # Convit heads
                self.convit_heads = num_heads // 2
                self.convit_slopes = torch.tensor([0.5 for i in range(self.convit_heads)], device=self.device)
                self.convit_intercepts = torch.zeros((self.convit_heads), device=self.device)
                self.convit_offsets = torch.tensor([-1 * (2.0 ** i) for i in range(self.convit_heads//2)] +
                                                   [2.0 ** i for i in range(self.convit_heads//2)], device=self.device)
                convit_biases = -1.0 * self.convit_slopes * torch.abs(torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(1) - (self.seq_len-1+self.convit_offsets)) + self.convit_intercepts  # Distance to "focus pixel", [2T-1, H]

                # Alibi heads
                self.alibi_heads = num_heads // 2
                log_slopes = torch.linspace(-1, -np.log2(seq_len/4), steps=self.alibi_heads, device=self.device)
                self.alibi_slopes = 2 ** log_slopes
                self.alibi_intercepts = torch.zeros((self.alibi_heads), device=self.device)  # Always 0 for now
                self.alibi_offsets = torch.zeros((self.alibi_heads), device=self.device)
                alibi_biases = -1.0 * self.alibi_slopes * torch.abs(torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(1) - (self.seq_len-1+self.alibi_offsets)) + self.alibi_intercepts  # Distance to "zero", [2T-1, H]

                # Combine
                bias_table_init = torch.cat([convit_biases, alibi_biases], dim=1)  # [2T-1, H]
                # bias_table_init = torch.clamp(bias_table_init, min=-5)

                # Duplicate for each layer
                bias_table_init = bias_table_init.repeat(num_layers, 1, 1)  # [L, 2T-1, H]

            # Define a parameter table of relative position bias
            self.relative_bias_table = nn.Parameter(bias_table_init, requires_grad=True)  # Relative offsets range from (T-1) to -(T-1), inclusive. Shape: [L, 2T-1, H]
            self.relpos_temp = nn.Parameter(torch.ones((num_layers, 1, num_heads)))  # Temperature for relative positional softmax (divide pre-softmax by this value). [L, 1, H] so relative_bias_table can be divided by this.

            # The attention matrix will have shape [T, T].
            # For entry (i, j), we want to look up the appropriate index in relative_bias_table,
            # which will be (i - j) + seq_len - 1. "relative_coords" does this lookup.
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [T, T]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0. Each entry (i, j) contains (i - j) + T - 1
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "convit":
            # ConViT-style relative position encoding. Because time-series are 1D (instead of 2D images),
            # we can simplify so that we just learn offset, intercept, and slope of linearly decaying functions
            # centered around a "focus point". These are used to construct the relative position bias table.
            assert num_heads % 2 == 0, "Assuming an even number of heads in convit_half relative position encoding"
            self.convit_heads = num_heads

            # Convit heads are initialized to focus attention around `convit_offsets`, with peak
            # intensity `convit_intercepts` and decay `convit_slopes`
            convit_slopes = torch.tensor([0.5 for i in range(self.convit_heads)], device=self.device)
            convit_intercepts = torch.zeros((self.convit_heads), device=self.device)
            convit_offsets = torch.tensor([-1 * ((2.0 ** i) - 0.5) for i in range(self.convit_heads//2)] +
                                          [(2.0 ** i) - 0.5 for i in range(self.convit_heads//2)], device=self.device)

            # Repeat for each layer. These have shape [L, H]
            self.convit_slopes = nn.Parameter(convit_slopes.repeat(num_layers, 1), requires_grad=True)
            self.convit_intercepts = nn.Parameter(convit_intercepts.repeat(num_layers, 1), requires_grad=True)
            self.convit_offsets = nn.Parameter(convit_offsets.repeat(num_layers, 1), requires_grad=True)

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [T, T]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "convit_half":
            # Half-ConViT relative position encoding. Some heads will be initialized to behave like ConViT,
            # with attention focused on a specific offset and decaying from there. Here, the offset,
            # slope, and intercept are all learnable. Other heads will be randomly initialized and
            # fully learnable (like ERPE).
            assert num_heads % 2 == 0, "Assuming an even number of heads in convit_half relative position encoding"
            self.convit_heads = (num_heads // 2) + 1
            self.normal_heads = num_heads - self.convit_heads

            # Normal heads have purely learnable relative positional embeddings
            self.normal_biases = nn.Parameter(torch.zeros(num_layers, 2*seq_len-1, self.normal_heads), requires_grad=True)  # Relative offsets range from (T-1) to -(T-1), inclusive

            # Convit heads are initialized to focus attention around `convit_offsets`, with peak
            # intensity `convit_intercepts` and decay `convit_slopes`
            convit_slopes = torch.tensor([1.0 for i in range(self.convit_heads)], device=self.device)
            convit_intercepts = torch.zeros((self.convit_heads), device=self.device)
            convit_offsets = torch.tensor([0] + [-1 * (3.0 ** i) for i in range(self.convit_heads//2)] +
                                          [3.0 ** i for i in range(self.convit_heads//2 - 1)], device=self.device)

            # Repeat for each layer. These have shape [L, H]
            self.convit_slopes = nn.Parameter(convit_slopes.repeat(num_layers, 1), requires_grad=True)
            self.convit_intercepts = nn.Parameter(convit_intercepts.repeat(num_layers, 1), requires_grad=True)
            self.convit_offsets = nn.Parameter(convit_offsets.repeat(num_layers, 1), requires_grad=True)

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [T, T]. Each entry (i, j) contains (i - j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)

        elif relative_pos_encoding == "alibi":
            # FIXED bias for relative offsets. Each head has a different function.
            # Code from https://github.com/ofirpress/attention_with_linear_biases/issues/5

            # For each entry in the attention matrix, store the matching index in relative_bias_table
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = torch.abs(coords_t[:, None] - coords_t[None, :])  # [T, T]. Each entry (i, j) contains ABS|i-j|. Note that this is different from the above eRPE approach.
            self.register_buffer("relative_coords", relative_coords)

            self.slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [num_heads]
            self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * self.relative_coords  # Broadcasting: [H, 1, 1] * [T, T] -> [H, T, T]


    def locality_loss_erpe(self):
        """
        Regularizes the relative biases - offsets further from the zero
        have a higher penalty if their probability is high.
        This can be seen as an Earth-Mover distance from the one-hot distribution
        (1 for my own timestep, 0 otherwise)
        """
        penalties = torch.arange(-self.seq_len+1, self.seq_len, device=self.device).abs() / self.seq_len

        ## Relative bias table should have shape  [L, 2T-1, H]
        biases_post_softmax = F.softmax(self.relative_bias_table, dim=1)
        x = (biases_post_softmax * penalties.unsqueeze(1)).sum(dim=1).mean()
        return x


    def locality_loss_attention(self):
        """
        Regularizes attention matrices to be close to diagonal
        """
        # Entry (i, j) contains |i-j|/seq_len
        device = self.attn_matrices.device
        penalty_matrix = (torch.arange(self.seq_len, device=self.device).unsqueeze(0) - torch.arange(self.seq_len, device=self.device).unsqueeze(1)).abs() / self.seq_len  # [T, T]

        # self.attn_matrices is [B, n_matrices, T, T]
        return (self.attn_matrices * penalty_matrix).sum(dim=-1).mean()


    def posenc_smoothness_loss(self, logger):
        """
        Computes smoothness of both absolute and relative positional embedding tables.
        TODO: Currently these are computed slightly differently, maybe make this consistent?
        """
        # Smoothness of absolute position encoding
        smoothness_loss = 0.
        if "learnable" in self.pos_encoding:
            # self.pos_embed has shape [T, D]
            # smoothness_loss += (torch.norm(self.pos_embed[1:, :] - self.pos_embed[:-1, :], dim=1)).mean()
            smoothness_loss += (self.pos_embed[1:, :] - self.pos_embed[:-1, :]).abs().mean()

        # Smoothness of relative position encodings
        if "erpe" in self.relative_pos_encoding or "convit" in self.relative_pos_encoding:
            # self.relative_bias_table has shape [L, 2*T-1, H]
            # rel_smoothness = ((self.relative_bias_table[:, 1:, :] - self.relative_bias_table[:, :-1, :]) ** 2).mean()
            rel_smoothness = (self.relative_bias_table[:, 1:, :] - self.relative_bias_table[:, :-1, :]).abs().mean()
            smoothness_loss += rel_smoothness

        return smoothness_loss


    def initialize_weights(self):
        # NOTE: pos_embed initialization is moved to setup_pos_embed
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)  # TODO
            # trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def create_var_embedding(self, dim):
        # number of variables x embedding dim: [V, D]
        var_embed = nn.Parameter(torch.zeros(self.img_size[1], dim), requires_grad=True)

        # Initialize with sincos, not sure if this is correct
        sincos = get_1d_sincos_pos_embed_from_grid(var_embed.shape[-1], np.arange(self.img_size[1]))
        var_embed.data.copy_(torch.from_numpy(sincos).float())
        return var_embed


    def aggregate_variables(self, x: torch.Tensor):
        """
        Input x: [B, V, T, D]
        Returns: [B, T, D]
        """
        b, _, t, _ = x.shape
        x = rearrange(x, "b v t d -> (b t) v d")  # Permute/reshape to [B*T, V, D]

        # self.var_query: [1, 1, D]
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)  # [B*T, 1, D]. The query only has 1 variable.

        # Pass query, key, value through var_agg (MultiheadAttention)
        x, _ = self.var_agg(var_query, x, x)  # [B*T, 1, D]  # Aggregate variables' embeddings, based on how similar each variable in "x" is to the query embedding
        x = rearrange(x, '(b t) 1 d -> b t d', t=t)  
        return x


    def forward_encoder(self, x: torch.Tensor, plot_dir: str = None):
        """
        Initial part of the forward method, including the patching, positional encoding setup, Transformer encoder
        Input: x: [B, T_orig, V]
        Returns x: [B, T, D]. attn_weights: [B, L*H, T, T]
        """

        if self.agg_vars:
            # Tokenize each variable separately. Each patch contains one variable, P timesteps.
            x = rearrange(x, "b t_orig v -> b v t_orig")  # [B, V, T_orig]
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [B, V, T, P]
            x = self.embed_layer(x)  # [B, V, T, D]

            # add variable embedding
            var_embed = self.var_embed  # [V, D]
            var_embed = var_embed.unsqueeze(0).unsqueeze(2)  # [1, V, 1, D]
            x = x + var_embed  # [B, V, T, D]

            # variable aggregation
            x = self.aggregate_variables(x)  # [B, T, D]

        elif self.conv_transformer:
            x = rearrange(x, "b t_orig v -> b v t_orig")  # [B, V (variables), T_orig (original timesteps)]
            x = self.embed_layer(x)  # [B, D (embed_dim), T (num_patches)]
            x = rearrange(x, "b d t -> b t d")  # [B, T, D]
        else:
            x = rearrange(x, "b t_orig v -> b v t_orig")
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [B, V, T (num_patches), P (patch_size)]
            x = rearrange(x, "b v t p -> b t (v p)")  # [B, T, V*P]
            x = self.embed_layer(x)  # [B, T, D]

        # Add ABSOLUTE pos embedding if using.
        # At this point, X should be [B, T, D], and pos_embed should be [T, D]. (T = number of patches along time dimension)
        if self.pos_embed is not None and self.where_to_add_abspos == "start_add":
            # CURRENT: add the positional embedding
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # Construct mask for relative positional encoding.
        offset_mask = None
        if "erpe" in self.relative_pos_encoding:
            # self.relative_bias_table: [L (layers), 2T-1, H (heads)]
            # self.relative_coords: [T, T] - ID of offset between timesteps
            # To construct relative embedding matrix, flatten the "offset matrix" (relative_coords),
            # and use these as indices into the relative bias table.
            # Then reshape to construct the real bias matrix (same shape as attention matrix)
            num_heads = self.relative_bias_table.shape[2]
            flattened_indices = self.relative_coords.flatten()  # [T*T]
            offset_mask = self.relative_bias_table.index_select(dim=1, index=flattened_indices) # [L, T*T, H]
            offset_mask = torch.clamp(offset_mask, min=-10000, max=10000)  # Prevent values from getting too extreme
            offset_mask = offset_mask / self.relpos_temp  # self.relpos_temp has shape [L, 1, H] so broadcasting works
            offset_mask = rearrange(offset_mask, 'l (t0 t1) h -> l h t0 t1', t0=self.seq_len)  # [L. H, T, T]
            offset_mask = offset_mask.repeat((1, x.shape[0], 1, 1))  # [L, B*H, T, T]

        elif self.relative_pos_encoding == "convit":
            # Set up bias table
            # Construct a tensor of all offsets, ranging from [-T+1, T-1]. Unsqueeze to shape [2T-1, 1, 1]
            rel_offsets = torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(1).unsqueeze(2) - (self.seq_len - 1)

            # convit_slopes, convit_offsets, convit_intercepts have shape [L, H].
            # When we subtract from rel_offsets, the result will have shape [2T-1, L, H]
            bias_table = -1.0 * self.convit_slopes * torch.abs(rel_offsets - self.convit_offsets) + self.convit_intercepts # Distance to "focus pixel", [2T-1, H_convit]
            self.relative_bias_table = rearrange(bias_table, 't l h -> l t h')  # relative_bias_table: [L, 2T-1, H]
            if plot_dir is not None:
                print("CONVIT: Offset", self.convit_offsets, "Intercepts", self.convit_intercepts, "Slope", self.convit_slopes)

            # Compute the actual offset matrix
            num_heads = self.relative_bias_table.shape[1]
            flattened_indices = self.relative_coords.flatten()  # [T*T]
            offset_mask = self.relative_bias_table.index_select(dim=1, index=flattened_indices) # [L, T*T, H]
            offset_mask = rearrange(offset_mask, 'l (t0 t1) h -> l h t0 t1', t0=self.seq_len)  # [L. H, T, T]
            offset_mask = offset_mask.repeat((1, x.shape[0], 1, 1))  # [L, B*H, T, T]

        elif self.relative_pos_encoding == "convit_half":
            # Set up bias table
            convit_biases = -1.0 * self.convit_slopes * torch.abs(torch.arange(0, 2*self.seq_len-1, device=self.device).unsqueeze(0).unsqueeze(2) - (self.seq_len-1+self.convit_offsets)) + self.convit_intercepts # Distance to "focus pixel", [2T-1, H_convit]
            bias_table = torch.cat([self.normal_biases, convit_biases], dim=2)
            self.relative_bias_table = bias_table
            if plot_dir is not None:
                print("CONVIT: Offset", self.convit_offsets, "Intercepts", self.convit_intercepts, "Slope", self.convit_slopes)

            # Compute the actual offset matrix
            num_heads = self.relative_bias_table.shape[2]
            flattened_indices = self.relative_coords.flatten()  # [T*T]
            offset_mask = self.relative_bias_table.index_select(dim=1, index=flattened_indices).reshape(self.num_layers, self.seq_len, self.seq_len, num_heads)  # [L, T, T, H]
            offset_mask = rearrange(offset_mask, 'l t0 t1 h -> l h t0 t1')  # [L, H, T, T]
            offset_mask = offset_mask.repeat((1, x.shape[0], 1, 1))  # [L, B*H, T, T]

        elif self.relative_pos_encoding == "alibi":
            # self.alibi: [H, T, T]
            offset_mask = self.alibi.repeat((self.num_layers, x.shape[0], 1, 1))  # Repeat along the layer and batch dimension: TransformerEncoder expects mask to be [L, B*H, T, T]

        # If some positions are not allowed to attend, either use the Boolean mask, or if combining with
        # relative position encoding, set those mask entries to -inf
        if self.invalid_mask is not None:
            if offset_mask is None:
                # If there is no relative position offset mask, create one from the
                # invalid mask with shape [L, T, T]. At each layer, the mask is True
                # at positions that are NOT ALLOWED to attend (too far).
                offset_mask = self.invalid_mask.repeat(self.num_layers, 1, 1)
            else:
                # Note: invalid_mask has shape [L, T, T], but offset_mask computed from
                # relative dimension is of shape [B*H, T, T]
                offset_mask[:, :, self.invalid_mask] = float("-inf")

        # Pass through encoder
        x, attn_weights, embeddings_layers = self.transformer_encoder(x, masks=offset_mask, plot_dir=plot_dir)  # x: [B, T, D]. attn_weights: [L, B, H, T, T], embeddings_layers: [L, B, T, D]
        return x, attn_weights, embeddings_layers


    def forward(self, x, plot_dir=None):
        """Forward pass through the model.

        Args:
            x: `[B, T_orig, V]` shape.
            plot_dir: if provided, plot attention matrices and distances between timestep feature vectors at each layer.
        Returns:
            preds (torch.Tensor): `[B]` shape. Predicted output.
        """

        # ENCODER forward pass
        preds, attn_weights_enc, embeddings_layers = self.forward_encoder(x, plot_dir=plot_dir)  # preds: [B, T, D], attn_weights_enc: [L, B, H, T, T], embeddings_layers: [L, B, T, D]

        # Pooling
        preds, pooling_attn = utils.forward_pooling(self, preds)

        # Save attention in case we use it for regularization later
        self.attn_matrices = attn_matrices = rearrange(attn_weights_enc, "l b h t0 t1 -> b (l h) t0 t1")  # Reshape to [B, L*H (num matrices), T, T]
        self.pooling_attn = pooling_attn

        if plot_dir is not None:
            # ALL VISUALIZATIONS of timestep distances/similarities, positional encodings, and
            # attention matrices. Exception: attention breakdown is in Attention_Rel_Scl.forward()

            # Plot an example input and distances between timesteps (just for comparison with later)
            orig_input = x.detach().cpu()  # [B, T_orig, V]
            visualization_utils.plot_time_series(orig_input[0, :, :].numpy(), os.path.join(plot_dir, 'example_x0.png'))

            # Compute similarities/distances between timestep feature vectors at each layer.
            # Start from the input variables
            feature_distances_layers = [torch.linalg.norm(orig_input.unsqueeze(1) - orig_input.unsqueeze(2), dim=3)]  # [B, T_orig, T_orig]
            similarity_matrix_layers = [F.cosine_similarity(orig_input.unsqueeze(1), orig_input.unsqueeze(2), dim=3)]  # [B, T_orig, T_orig]

            # Continue to latent embeddings (before encoder, and after each layer)
            for layer_idx in range(self.num_layers + 1):
                embed = embeddings_layers[layer_idx, :, :, :].detach().cpu()  # [B, T, D]

                # The unsqueeze operations convert embed into [B, 1, T, D] and [B, T, 1, D].
                # Subtracting them produces [B, T, T, D] (due to broadcasting), then take norm/similarity over the D dimension
                # Result is [B, T, T]
                feature_distances_layers.append(torch.linalg.norm(embed.unsqueeze(1) - embed.unsqueeze(2), dim=3))  # Append [B, T, T]
                similarity_matrix_layers.append(F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(2), dim=3))  # Append [B, T, T]

            # Create a plot, where each row is an exmaple, and each column is a layer.
            # Plot distances between timestep feature vectors at each layer
            n_rows = 5  # Examples to plot
            n_cols = len(feature_distances_layers)
            # feature_distances_layers = torch.stack(feature_distances_layers, dim=1)  # List of [B, T, T] -> [B, L, T, T]
            min_value, max_value = utils.approx_min_max(feature_distances_layers)
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout='constrained')
            for r in range(n_rows):
                for c in range(n_cols):
                    im = axeslist[r, c].imshow(feature_distances_layers[r][c, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)
                    if r == 0:
                        if c == 0:
                            axeslist[r, c].set_title("Initial input")
                        if c < n_cols-1:
                            axeslist[r, c].set_title(f"Before layer {c+1}")
                        else:
                            axeslist[r, c].set_title(f"Final")
                    if c == 0:
                        axeslist[r, c].set_ylabel(f"Example {r+1}", rotation=0, size='large', labelpad=30)
            fig.colorbar(im, ax=axeslist, shrink=0.4)  #[r,c])
            fig.suptitle("Distance between timestep feature vectors")
            plt.savefig(os.path.join(plot_dir, 'timestep_distances.png'))
            plt.close()

            # Plot cosine similarity between timestep feature vectors (at each layer)
            n_cols = len(similarity_matrix_layers)
            # similarity_matrix_layers = torch.stack(similarity_matrix_layers, dim=1)  # List of [B, T, T] -> [B, L, T, T]
            min_value, max_value = utils.approx_min_max(similarity_matrix_layers)
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout='constrained')
            for r in range(n_rows):
                for c in range(n_cols):
                    im = axeslist[r, c].imshow(similarity_matrix_layers[r][c, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)
                    if r == 0:
                        if c == 0:
                            axeslist[r, c].set_title("Initial input")
                        if c < n_cols-1:
                            axeslist[r, c].set_title(f"Before layer {c+1}")
                        else:
                            axeslist[r, c].set_title(f"Final")
                    if c == 0:
                        axeslist[r, c].set_ylabel(f"Example {r+1}", rotation=0, size='large', labelpad=30)
            fig.colorbar(im, ax=axeslist, shrink=0.4)  # [r,c])
            fig.suptitle("Cos similarity between timestep feature vectors")
            plt.savefig(os.path.join(plot_dir, 'timestep_similarities.png'))
            plt.close()

            # Plot attention matrices: for each example, plot random subset of layers/heads. attn_weights_enc: [L, B, H, T, T]
            min_value, max_value = utils.approx_min_max(attn_matrices)

            # Random subset of heads/layers
            n_matrices = attn_matrices.shape[1]  # Total number of attention maps per example (L*H)
            n_cols = self.num_layers * 2
            matrix_indices = np.sort(np.random.choice(np.arange(n_matrices), n_cols, replace=False))
            fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout="constrained")
            for r in range(n_rows):
                for c in range(n_cols):
                    m = matrix_indices[c]
                    im = axeslist[r, c].imshow(attn_matrices[r, m, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value)  #, vmin=0, vmax=3/attn_weights_layers.shape[2])  #0/attn_weights_layers.shape[1])
                    if r == 0:
                        layer_idx = m // self.num_heads
                        head_idx = m % self.num_heads
                        axeslist[r, c].set_title(f"Layer {layer_idx}, Head {head_idx}")
                    if c == 0:
                        axeslist[r, c].set_ylabel(f"Example {r+1}", rotation=0, size='large', labelpad=30)
            fig.colorbar(im, ax=axeslist, shrink=0.4)  # [r, c])
            fig.suptitle("Example attention matrices")
            plt.savefig(os.path.join(plot_dir, 'attention_matrices.png'))
            plt.close()

            # Visualize absolute positional encoding
            if "learnable" in self.pos_encoding:
                visualization_utils.visualize_absolute_posenc(self.pos_embed, plot_dir)

            # Plot relative positional encoding for each layer
            if "erpe" in self.relative_pos_encoding or "convit" in self.relative_pos_encoding:
                # self.relative_bias_table is [L (layers), 2T-1 (time offsets), H (heads)]
                # self.relpos_temp is [L, 1, H]
                if "erpe" in self.relative_pos_encoding:
                    bias_table = self.relative_bias_table / self.relpos_temp
                else:
                    bias_table = self.relative_bias_table

                # Line plots
                n_rows = 2  # Pre-softmax and post-softmax
                n_cols = self.num_layers
                fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows), layout="constrained")
                timesteps = np.arange(bias_table.shape[1]) - (self.seq_len - 1)
                for layer_idx in range(self.num_layers):
                    # Pre-softmax
                    for head_idx in range(bias_table.shape[2]):
                        axeslist[0, layer_idx].plot(timesteps, bias_table[layer_idx, :, head_idx].cpu().detach().numpy(),
                                                    label=f"Head {head_idx}")  # marker="o", linestyle="-"
                    axeslist[0, layer_idx].set_xlabel("Offset from current timestep")
                    axeslist[0, layer_idx].set_ylabel("Bias (pre-softmax)")
                    axeslist[0, layer_idx].set_title(f"Layer {layer_idx}: Pre-softmax biases")
                    axeslist[0, layer_idx].legend()

                    # Post-softmax
                    for head_idx in range(bias_table.shape[2]):
                        axeslist[1, layer_idx].plot(timesteps, F.softmax(bias_table[layer_idx, :, head_idx], dim=0).cpu().detach().numpy(),
                                                    label=f"Head {head_idx}")
                    axeslist[1, layer_idx].set_xlabel("Offset from current timestep")
                    axeslist[1, layer_idx].set_ylabel("Bias (post-softmax)")
                    axeslist[1, layer_idx].set_title(f"Layer {layer_idx}: Post-softmax biases")
                    axeslist[1, layer_idx].legend()
                fig.suptitle("Relative attention biases")
                plt.savefig(os.path.join(plot_dir, 'relative_pos_biases.png'))
                plt.close()

                # # Shaded plot
                # min_value, max_value = utils.approx_min_max(bias_table)
                # n_rows = 1
                # n_cols = self.num_layers
                # fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(0.15*n_cols*bias_table.shape[2]+3, 0.03*n_rows*bias_table.shape[1]), layout="constrained")
                # for c in range(n_cols):  # Loop through each layer
                #     im = axeslist[c].imshow(bias_table[c, :, :].detach().cpu().numpy(), vmin=min_value, vmax=max_value, aspect=0.2, interpolation='none')  # stretch each column horizontally 5x
                #     axeslist[c].set_xlabel("Head number")
                #     axeslist[c].set_ylabel("Relative offset (middle is 0)")
                #     axeslist[c].set_title(f"Layer {c}")
                # fig.colorbar(im, ax=axeslist, shrink=0.4)
                # fig.suptitle("Relative attention biases")
                # plt.savefig(os.path.join(plot_dir, 'relative_pos_offsets.png'))
                # plt.close()

            # Visualize SeqPool attention weights
            if self.pool in ["seqpool", "seqpool_multihead", "seqpool_multihead_smoothed"]:
                visualization_utils.visualize_pooling_attn(pooling_attn, plot_dir)

        return preds, attn_weights_enc, pooling_attn


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

    def forward(self, src: Tensor, masks: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, plot_dir: str = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required). Must be of shape [B, T, D]
            masks: for each layer, the mask for the src sequence (optional). Shape: [L, B*H, T, T]
            src_key_padding_mask: the mask for the src keys per batch (optional).
            plot_dir: if provided, plot attention matrices and distances between timestep feature vectors

        Shape:
            see the docs in Transformer class.
        
        Returns:
            output: [B, T, D]
            attn_weights_layers: [L, B, H, T, T], attention matrix for each layer, example, head
            embeddings_layers: [L+1, B, T, D], embeddings at start and after each layer
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
        elif masks is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and masks were both supplied"
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

        attn_weights_layers = []
        embeddings_layers = [output]  # Also store the pre-encoder embedding

        # Compute forward pass
        for layer_idx, mod in enumerate(self.layers):
            mask = masks[layer_idx] if masks is not None else None
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers, plot_dir=plot_dir)  # output: [B, T, D], attn_weights: [B, H, T, T]
            attn_weights_layers.append(attn_weights)
            embeddings_layers.append(output)

        attn_weights_layers = torch.stack(attn_weights_layers, dim=0)  # [L, B, H, T, T]
        embeddings_layers = torch.stack(embeddings_layers, dim=0)  # [L+1, B, T, D]
        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights_layers, embeddings_layers

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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_pos_encoding='none', where_to_add_relpos='before', conv_projection=False):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        # if where_to_add_relpos == "before":
        #     # PyTorch's implementation of MultiheadAttention only allows mask to be applied before softmax.
        #     # TODO: Currently commenting this out so we can use the customized verson below.
        #     # Note: we could also use Attention_Rel_Scl here. TODO - check that they behave the same way
        #     self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        #     assert conv_projection == False, "conv_projection is only supported for custom attention (Attention_Rel_Scl)"
        # else:
        
        if relative_pos_encoding == "rope":
            self.self_attn = AttentionWithRoPE(d_model, nhead, attn_drop=dropout, proj_drop=dropout)
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
            src: the sequence to the encoder layer (required). Shape: [B, T, D]
            src_mask: the mask for the src sequence (optional). Shape: [B*H, T, T]
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if type(self.self_attn) == Attention_Rel_Scl:
            # Attention_Rel_Scl allows plot_dir
            src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask, average_attn_weights=False, plot_dir=plot_dir)  # src2: [B, T, D], attn_output_weights: [B, H, T, T]
        else:
            src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask, average_attn_weights=False)  # src2: [B, T, D], attn_output_weights: [B, H, T, T]

        src = src + self.dropout1(src2)  # [B, T, D]
        src = rearrange(src, 'b t d -> b d t')  # Convert to [B, D, T] only for normalization (which expects channel dim first)
        src = self.norm1(src)
        src = rearrange(src, 'b d t -> b t d')  # Restore [B, T, D]
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # [B, T, D]
        src = rearrange(src, 'b t d -> b d t')
        src = self.norm2(src)
        src = rearrange(src, 'b d t -> b t d')  # Restore [B, T, D]
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

        self.dropout = nn.Dropout(dropout)
        self.gating_param = nn.Parameter(torch.ones(num_heads))  # torch.cat([-1*torch.ones(num_heads//2), torch.ones(num_heads//2)]))


    def forward(self, query, key, value, attn_mask, plot_dir=None, **kwargs):
        """
        Input (query/key/value) should be [B, T, D]. They can be identical
        as the linear projections happen inside the method.
        Mask should be [B*H, T, T]

        Output is [B, T, D], and attn matrix [B, H, T, T]
        """
        assert query.shape == key.shape
        assert query.shape == value.shape

        if self.where_to_add_relpos == "only_relpos":
            # If only_relpos, we don't need to calcualte content attention - just
            # set it to 0
            content_attn = torch.zeros((query.shape[0], self.num_heads, query.shape[1], query.shape[1]), device=query.device)  # [B, H, T, T]
        else:
            # Calculate content attention
            # self.key, self.value, self.query output [B, T, D]. Reshape/permute to extract the head dimension.
            k = self.key(key)  # [B, T, D]
            k = rearrange(k, 'b t (h d_h) -> b h d_h t', h=self.num_heads)  # Split embedding dimensions into heads, permute to [B, H, d_head, T]
            q = self.query(query)  # [B, T, D]
            q = rearrange(q, 'b t (h d_h) -> b h t d_h', h=self.num_heads)  # Split embedding dimensions into heads, permute to [B, H, T, d_head]
            # # k shape = [B, H, d_head, T]
            # # v,q shape = [B, H, T, d_head]
            content_attn = torch.matmul(q, k) * self.scale  # attn shape [B, H, T, T]

        # Calculate value in all cases
        v = self.value(value)  # [B, T, D]
        v = rearrange(v, 'b t (h d_h) -> b h t d_h', h=self.num_heads)  # Split embedding dimensions into heads, permute to [B, H, T, d_head]

        if attn_mask is not None:
            # Reshape attn_mask from [B*H, T, T] to [B, H, T, T]
            attn_mask = rearrange(attn_mask, '(b h) t0 t1 -> b h t0 t1', h=self.num_heads)

        # Perform softmax
        if (self.where_to_add_relpos in ['before', 'only_relpos']) and attn_mask is not None:
            # Add mask (relative position encoding) before softmax if specified
            attn = F.softmax(content_attn + attn_mask, dim=-1)
        else:
            # Take softmax of content attention first (relative position encoding added later)
            attn = F.softmax(content_attn, dim=-1)
            content_attn = attn

        if attn_mask is not None:
            if self.where_to_add_relpos == 'after':
                # In this case, content_attn has been passed through softmax already
                attn = content_attn + attn_mask
            elif self.where_to_add_relpos == "after_gating":
                # In this case, content_attn has been passed through softmax already
                gating = self.gating_param.view(1,-1,1,1)  # [1, H, 1, 1]

                # both content_attn and attn_mask should be [B, H, T, T]
                attn = (1.-torch.sigmoid(gating))*content_attn + torch.sigmoid(gating)*F.softmax(attn_mask, dim=-1)  # First term is original content attention, second term is position attention
                attn /= attn.sum(dim=-1).unsqueeze(-1)

            if plot_dir is not None:
                if self.where_to_add_relpos == "after_gating":
                    print("Gating (Pr position)", torch.sigmoid(self.gating_param))

                # PLOTTING ONLY
                # Plot attention breakdown (content/position) for a single example, 'n_rows' heads
                n_rows = 4
                n_cols = 3
                fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout="constrained")

                for r in range(n_rows):
                    head_num = r * (attn.shape[1] // n_rows)
                    max_value = 0.1  #/attn.shape[2]
                    content_attn_head = content_attn[0, head_num, :, :]
                    if self.where_to_add_relpos in ["after_gating"]:
                        pos_attn_head = F.softmax(attn_mask[0, head_num, :, :], dim=-1)
                    else:
                        pos_attn_head = attn_mask[0, head_num, :, :]
                    total_attn_head = attn[0, head_num, :, :]
                    axeslist[r, 0].imshow(content_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    axeslist[r, 1].imshow(pos_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    im = axeslist[r, 2].imshow(total_attn_head.detach().cpu().numpy(), vmin=0, vmax=max_value)
                    if r == 0:
                        if self.where_to_add_relpos == "after_gating":
                            axeslist[r, 0].set_title("Content attn\n(post-softmax)")
                            axeslist[r, 1].set_title("Position attn\n(post-softmax)")
                            axeslist[r, 2].set_title("Combined:\n(1-w)*Content + w*Position")
                        elif self.where_to_add_relpos in ["before", "only_relpos"]:
                            axeslist[r, 0].set_title("Content attn\n(unnormalized)")
                            axeslist[r, 1].set_title("Position attn\n(unnormalized)")
                            axeslist[r, 2].set_title("Combined:\nsoftmax(Content + Position)")
                        else:
                            assert self.where_to_add_relpos == "after"
                            axeslist[r, 0].set_title("Content attn\n(post-softmax)")
                            axeslist[r, 1].set_title("Position attn\n(unnormalized)")
                            axeslist[r, 2].set_title("Combined:\nsoftmax(Content) + Position")
                fig.colorbar(im, ax=axeslist[r])
                fig.suptitle("Attn breakdown, single example (each row is one head)")
                plt.savefig(os.path.join(plot_dir, 'attention_breakdown.png'))
                plt.close()

        out = torch.matmul(attn, v)  # [B, H, T, T] * [B, H, T, d_head] -> [B, H, T, d_head]
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = rearrange(out, 'b h t d_h -> b t (h d_h)')  # Reunify the heads, output is [B, T, D]
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
        # init_gating = torch.ones(self.num_heads)*2 # Second half of heads prefer position attention
        # init_gating[0:self.num_heads//2] = -2  # First half of heads prefer content attention
        init_gating = torch.ones(self.num_heads)
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
            # print("Gating (Pr position)", torch.sigmoid(self.gating_param))
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

    Source: https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py
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
        Input [batch, in_chans, time] or [B, V, T]
        Output [batch, embed_dim, time] or [B, D, T]
        """
        x = self.proj(x)

        if self.norm:
            x = self.norm(x)
        return x