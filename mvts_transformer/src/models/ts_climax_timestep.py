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
        if config['model'] == 'climax_smooth_plot':
            return TSTEncoder(config['d_model'], config['d_model'], config['num_heads'],
                              d_ff=config['dim_feedforward'], dropout=config['dropout'],
                              activation=config['activation'], n_layers=config['num_layers'])
    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'climax_smooth_plot':
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
        local_mask=-1,
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
        self.device = device
        self.local_mask = local_mask

        # variable tokenization: separate embedding layer for each input variable

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables

        if self.agg_vars:
            self.embed_layer = nn.Linear(patch_size, embed_dim)

            # variable aggregation: a learnable query and a single-layer cross attention
            self.var_embed = self.create_var_embedding(embed_dim)
            self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
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

        encoder_layer = TransformerBatchNormEncoderLayer(
                embed_dim, num_heads, feedforward_dim, drop_rate * (1.0 - freeze))
        self.transformer_encoder = TransformerEncoder(encoder_layer, depth)
        # self.head_linear = nn.Linear(embed_dim, embed_dim // 2)

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(p=drop_rate)

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # final linear layer
        self.output_layer = nn.Linear(embed_dim * int((max_seq_len - patch_size) / stride + 1), num_classes)

        # Local mask. Restrict which pairs of timesteps can pay attention to each other
        if self.local_mask >= 0:

            # Note that if relative positional encoding is being used, this is redundant with "relative_coords"
            indices = torch.arange(0, int((max_seq_len - patch_size) / stride + 1))
            distance_matrix = torch.abs(indices.reshape((1, -1)) - indices.reshape((-1, 1)))  # [seq_len, seq_len]
            self.invalid_mask = torch.zeros((len(indices), len(indices))).bool()  # [seq_len, seq_len]
            self.invalid_mask[distance_matrix > self.local_mask] = True
            print(self.invalid_mask)
        else:
            self.invalid_mask = None


    def setup_posenc(self, pos_encoding, relative_pos_encoding, seq_len, emb_dim, num_heads):

        # For absolute positional encodings, it is a matrix of size [seq_len, emb_dim]
        if pos_encoding == "learnable":
            # Simple learnable vector for each position
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, emb_dim), requires_grad=True)
            nn.init.uniform_(self.pos_embed, -0.02, 0.02)
        elif pos_encoding == "learnable_sin_init":
            # Simple learnable vector for each position, initialized with sinusoidal features
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, emb_dim), requires_grad=True)
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1],
                np.arange(int((self.max_len - self.patch_size) / self.stride + 1))
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

        if relative_pos_encoding == "erpe":
            # eRPE: learnable bias for each relative offset between timesteps
            # define a parameter table of relative position bias
            # TODO. In the original eRPE paper, this is applied AFTER softmax, but in
            # here we do it BEFORE softmax because of PyTorch's MultiheadAttention
            # implementation. Try to fix this later!
            self.relative_bias_table = nn.Parameter(torch.zeros(2*seq_len-1, num_heads), requires_grad=True)  # Relative offsets range from (seq_len-1) to -(seq_len-1), inclusive

            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [seq_len, seq_len]. Each entry (i, j) contains (i-j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)
        elif relative_pos_encoding == "erpe_alibi_init":
            # Calculate initial bias table using ALIBI linear functions for each head. Note that the linaer function is multiplying "slope" with absolute |distance|.
            slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [num_heads]
            bias_table_init = torch.zeros(2*seq_len-1, num_heads)
            bias_table_init[0:self.seq_len-1] = torch.arange(start=self.seq_len-1, end=0, step=-1, device=self.device).unsqueeze(1) * slopes
            bias_table_init[self.seq_len-1:] = torch.arange(start=0, end=self.seq_len, device=self.device).unsqueeze(1) * slopes
            self.relative_bias_table = nn.Parameter(bias_table_init, requires_grad=True)  # Relative offsets range from (seq_len-1) to -(seq_len-1), inclusive

            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = coords_t[:, None] - coords_t[None, :]  # [seq_len, seq_len]. Each entry (i, j) contains (i-j)
            relative_coords += seq_len - 1  # shift to start from 0
            self.register_buffer("relative_coords", relative_coords)
        elif relative_pos_encoding == "alibi":
            # FIXED bias for relative offsets. Each head has a different function.
            # Code from https://github.com/ofirpress/attention_with_linear_biases/issues/5

            # get pair-wise relative position index for each token inside the window
            coords_t = torch.arange(seq_len, device=self.device)
            relative_coords = torch.abs(coords_t[:, None] - coords_t[None, :])  # [seq_len, seq_len]. Each entry (i, j) contains ABS|i-j|. Note that this is different from the above eRPE approach.
            self.register_buffer("relative_coords", relative_coords)

            self.slopes = torch.tensor(get_slopes(num_heads), device=self.device)*-1  # [num_heads]
            # print("Slopes", self.slopes)
            self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * self.relative_coords  # Broadcasting: [num_heads, 1, 1] * [seq_len, seq_len] -> [num_heads, seq_len, seq_len]
            # self.alibi = self.alibi.view(1, num_heads, seq_len, seq_len)  # [1, num_heads, seq_len, seq_len]
            # print("Final bias", self.alibi)


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

        if "erpe" in self.relative_pos_encoding:
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

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, T, V]` shape.
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

        else:
            x = x.permute(0, 2, 1) # B, V, T
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # B, V, num_patches, patch_size
            x = x.permute(0, 2, 1, 3)  # B, num_patches, V, patch_size
            x = x.reshape((x.shape[0], x.shape[1], -1))  # B, num_patches, V*patch_size
            x = self.embed_layer(x)

        # Add ABSOLUTE pos embedding if using.
        # At this point, X should be [batch, seq_len, embed_dim], and pos_embed should be [seq_len, embed_dim]. (seq_len = number of patches along time dimension)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks. NOT USED ANYMORE
        # for blk in self.blocks:
        #     x = blk(x)

        # Construct mask for relative positional encoding.
        offset_mask = None
        if "erpe" in self.relative_pos_encoding:
            # self.relative_bias_table: [2*seq_len-1, num_heads]
            # self.relative_coords: [seq_len, seq_len] - ID of offset between timesteps
            # To construct relative embedding matrix, flatten the "offset matrix" (relative_coords),
            # and use these as indices into the relative bias table.
            # Then reshape to construct the real bias matrix (same shape as attention matrix)
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

        # Change to [seq_len, batch, embed_dim] to align with Pytorch convention
        x = x.permute((1, 0, 2))

        x, attn_weights = self.transformer_encoder(x, mask=offset_mask)  # after encoder. x: [seq_len, batch, embed_dim]. attn_weights: [batch, n_layer*n_head, seq_len, seq_len]
        # x = self.head_linear(x)
        # x = self.norm(x)

        x = x.permute((1, 0, 2))  # Permute back to [batch, seq_len, embed_dim]
        return x, attn_weights

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[batch_size, seq_length, feat_dim]` shape.
        Returns:
            preds (torch.Tensor): `[B]` shape. Predicted output.
        """
        B, T, V = x.shape
        preds, attn_weights = self.forward_encoder(x)
        preds = self.act(preds)
        preds = self.dropout1(preds)
        preds = preds.reshape(preds.shape[0], -1)
        preds = self.output_layer(preds)

        output_layer_weights = self.output_layer.weight.cpu().detach().numpy()
        output_layer_weights = output_layer_weights.reshape(T, -1)
        weight_by_timestep = np.sum(np.abs(output_layer_weights), -1)
        weight_by_timestep = np.expand_dims(weight_by_timestep, axis=0)
        plt.imshow(weight_by_timestep, cmap='plasma', aspect="auto")
        plt.savefig(os.path.join("/home/fs01/kjc256/tser/graphs", "output_layer_weights_by_timestep.png"))
        plt.close()
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

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required). Must be of shape [TIME, BATCH, EMBED_DIM]
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

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
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)  # output: [seq_len, batch, embed_dim], attn_weights: [batch, num_heads, seq_len, seq_len]
            if attn_weights_layers is None:
              attn_weights_layers = attn_weights
            else:
              attn_weights_layers = torch.cat((attn_weights_layers, attn_weights), dim=1)  # attn_weights: [batch, n_layers*num_heads, seq_len, seq_len]

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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
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
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required). Shape: [seq_len, batch, embed_dim]
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, average_attn_weights=False)  # src2: [seq_len, batch_size, d_model], attn_output_weights: [batch, num_heads, seq_len, seq_len]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src, attn_output_weights