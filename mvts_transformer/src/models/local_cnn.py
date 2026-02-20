"""
Code for 'Local CNN' baselines, which are highly-local convolutional
networks (or per-timestep MLP), followed by global pooling (optional).

We use the following letters to annotate shapes:
B: batch (examples)
T: timesteps
T': timesteps after CNN
D: 'channels' (embedding dimension)
"""



import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils, visualization_utils
from einops import rearrange
from models.ts_climax import ClimaX
from torch.nn.utils.parametrizations import spectral_norm

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
        raise NotImplementedError()
    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'local_cnn':
            return LocalCNN(in_channels=feat_dim, max_len=max_seq_len, num_classes=num_labels, embed_dim=config['d_model'],
                            patch_size=config['patch_length'], stride=config['stride'],
                            conv_type=config['conv_type'], pool=config['pool'],
                            pos_encoding=config['pos_encoding'], where_to_add_abspos=config['where_to_add_abspos'],
                            num_heads=config['num_heads'])
        # elif config['model'] == 'local_cnn2':
        #     return LocalCNN2(in_channels=feat_dim, max_len=max_seq_len, num_classes=num_labels, embed_dim=config['d_model'],
        #                     patch_size=config['patch_length'], stride=config['stride'],
        #                     conv_type=config['conv_type'], pool=config['pool'],
        #                     pos_encoding=config['pos_encoding'], where_to_add_abspos=config['where_to_add_abspos'],
        #                     num_heads=config['num_heads'], conv_dropout=config['dropout'],
        #                     use_batch_norm=config['local_cnn2_batch_norm'],
        #                     use_spectral_norm=config['local_cnn2_spectral_norm'])
        else:
            raise ValueError("Unsupported model", config['model'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


class LocalCNN(nn.Module):
    def __init__(self, in_channels, max_len=144, num_classes=1, embed_dim=256, 
                 patch_size=1, stride=1, conv_type="per_timestep", pool="seqpool_multihead",
                 pos_encoding="learnable_sin_init", where_to_add_abspos="before_pooling_concat",
                 num_heads=16):
        """
        conv_type can be hierarchical or local

        pool can be seqpool (attention pooling in Compact Convolutional Transformer paper), seqpool_multihead,
        average, or linear.

        if posenc is True, add a positional encoding right before the pooling. NOT IMPLEMENTED YET
        """
        super().__init__()
        self.in_channels = in_channels
        self.max_len = max_len
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        self.conv_type = conv_type
        self.pool = pool
        self.pos_encoding = pos_encoding
        self.where_to_add_abspos = where_to_add_abspos
        self.num_heads = num_heads

        # Embedding layer
        if self.where_to_add_abspos == "start_concat":
            # If concatenating pos emb at start, use half of the dimensions for content and half for position
            content_embed_dim = embed_dim // 2
        else:
            content_embed_dim = embed_dim

        self.embed_layer = nn.Linear(patch_size*in_channels, content_embed_dim)  # Each patch has patch_size*num_variables (P*V) elements. Map to embed_dim/2 (D/2).
        self.seq_len = int((max_len - patch_size) / stride + 1)  # Number of patches in time dimension, AFTER PATCHING

        if self.conv_type == "hierarchical":  # Gradually reduces number of timesteps
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(embed_dim, embed_dim, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(embed_dim, embed_dim, 3, 1),
                nn.AvgPool1d(2, 2),
            )
        elif self.conv_type == "local":  # No reducing number of timesteps
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 5, 1, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(embed_dim, embed_dim, 5, 1, dilation=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(embed_dim, embed_dim, 5, 1, dilation=5, padding='same'),
            )
        elif self.conv_type == "per_timestep":
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 1, 1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Conv1d(embed_dim, embed_dim, 1, 1),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Conv1d(embed_dim, embed_dim, 1, 1),
            )
        elif self.conv_type == "lstm":
            self.conv = nn.LSTM(embed_dim, embed_dim, num_layers=2, batch_first=True)
        else:
            raise ValueError("invalid conv_type")

        # Calculate number of timesteps after conv. 
        # TODO I recall there is a better way to do this, but could not find yet
        if self.conv_type == "lstm":
            self.output_seq_len = self.seq_len 
        else:
            input = torch.randn((1, self.embed_dim, self.seq_len))  # [B, D, T]. Conv expects this order
            output = self.conv(input)  # [B, D, T]
            self.output_seq_len = output.shape[2]

        # Positional embedding
        # Note: if absolute positional embedding is added inside the pooling attention,
        # the embedding size is equal to the number of heads. Otherwise it's the normal embedding dim.
        if where_to_add_abspos in ["pooling_before_softmax", "pooling_gating"]:
            if self.pool == "seqpool":
                absolute_emb_dim = 1
            else:
                absolute_emb_dim = num_heads
        else:
            absolute_emb_dim = content_embed_dim
        ClimaX.setup_absolute_posenc(self, pos_encoding, self.output_seq_len, absolute_emb_dim)

        # Setup pooling
        ClimaX.setup_pooling(self, embed_dim, num_heads, self.output_seq_len, num_classes)

        # Only used if pool == 'linear'
        self.dropout1 = nn.Dropout(0.1)
        self.act = F.gelu

        # Only used if where_to_add_abspos is 'start_add'
        self.pos_drop = nn.Dropout(p=0.1)


    def forward(self, x, plot_dir=None, return_embeddings=False):
        """
        x should have shape [B, T, input_vars]
        """
        # Patching if desired
        x = rearrange(x, "b t_orig v -> b v t_orig")
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [B, V, T (num_patches), P (patch_size)]
        x = rearrange(x, "b v t p -> b t (v p)")  # [B, T, V*P]
        x = self.embed_layer(x)  # [B, T, D]

        # Add ABSOLUTE pos embedding (if adding at start)
        # At this point, X should be [B, T, D], and pos_embed should be [T, D]. (T = number of patches along time dimension)
        if self.pos_embed is not None and self.where_to_add_abspos == "start_add":
            # CURRENT: add the positional embedding
            x = x + self.pos_embed
            x = self.pos_drop(x)
        elif self.pos_embed is not None and self.where_to_add_abspos == "start_concat":
            # Repeat pos embed along batch dimension, then concatenate along embedding dimension
            x = torch.cat((x, self.pos_embed.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)), dim=2)  # [B, T, D]

        # Main convolutional (local) backbone
        if self.conv_type == "lstm":
            x = self.conv(x)[0]  # LSTM input and output is [B, T, D], no need to convert
        else:
            x = rearrange(x, 'b t d -> b d t')  # Move the "channel" (D) dimension forward, to [B, D, T]
            x = self.conv(x)  # Convert to [batch, channel, time']  (may be fewer timesteps)
            x = rearrange(x, 'b d t -> b t d')  # Change back to [B, T, D] for compatibility with pooling

        # Pooling
        print("Before pool", x.shape)
        preds, pooling_attn = ClimaX.forward_pooling(self, x)
        print("Preds shape", preds.shape, pooling_attn.shape)
        self.pooling_attn = pooling_attn

        # Visualize positional embedding
        if plot_dir is not None and "learnable" in self.pos_encoding:
            visualization_utils.visualize_absolute_posenc(self.pos_embed, plot_dir)

        # Visualize SeqPool attention weights
        if plot_dir is not None and self.pool in ["seqpool", "seqpool_multihead", "seqpool_multihead_smoothed"]:
            visualization_utils.visualize_pooling_attn(pooling_attn, plot_dir)

        if return_embeddings:
            warnings.warn("TODO: LocalCNN doesn't support returning layer embeddings yet.")
            return preds, None, pooling_attn, None

        return preds, None, pooling_attn


    def pool_smoothness_loss(self):
        return ClimaX.pool_smoothness_loss(self)

    def jacobian_loss(self, input):
        return ClimaX.jacobian_loss(self, input)

    def predict_summed(self, func, input):
        return ClimaX.predict_summed(self, func, input)



# class LocalCNN2(nn.Module):
#     def __init__(self, in_channels, max_len=144, num_classes=1, embed_dim=256, 
#                  patch_size=1, stride=1, conv_type="per_timestep", pool="seqpool_multihead",
#                  pos_encoding="learnable_sin_init", where_to_add_abspos="before_pooling_concat",
#                  num_heads=16, conv_dropout=0.1, use_batch_norm=False, use_spectral_norm=False):
#         """
#         EXPERIMENTAL - does not work well

#         UPDATED VERSION of LocalCNN that supports residual connections + spectral normalization
#         conv_type can be hierarchical or local

#         pool can be seqpool (attention pooling in Compact Convolutional Transformer paper), seqpool_multihead,
#         average, or linear.

#         if posenc is True, add a positional encoding right before the pooling. NOT IMPLEMENTED YET
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.max_len = max_len
#         self.num_classes = num_classes
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.stride = stride
#         self.conv_type = conv_type
#         self.pool = pool
#         self.pos_encoding = pos_encoding
#         self.where_to_add_abspos = where_to_add_abspos
#         self.conv_dropout = conv_dropout

#         # Embedding layer
#         self.embed_layer = nn.Linear(patch_size*in_channels, embed_dim)  # Each patch has patch_size*num_variables (P*V) elements. Map to embed_dim (D).
#         self.seq_len = int((max_len - patch_size) / stride + 1)  # Number of patches in time dimension, AFTER PATCHING

#         # If using spectral norm, this is a function on the layer.
#         # Otherwise it's an identity function
#         print("DROPOUT", self.conv_dropout)
#         if use_spectral_norm:
#             print("USING SPECTRAL NORM")
#             spectral_norm_fn = spectral_norm
#         else:
#             spectral_norm_fn = lambda x: x
#         if use_batch_norm:
#             print("USING BATCH NORM")
#             self.act1 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.BatchNorm1d(embed_dim),
#                 nn.ReLU(),
#             )
#             self.act2 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.BatchNorm1d(embed_dim),
#                 nn.ReLU(),
#             )
#             self.act3 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.BatchNorm1d(embed_dim),
#                 nn.ReLU(),
#             )
#             self.act4 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.BatchNorm1d(embed_dim),
#                 nn.ReLU(),
#             )
#         else:
#             self.act1 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.ReLU(),
#             )
#             self.act2 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.ReLU(),
#             )
#             self.act3 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.ReLU(),
#             )
#             self.act4 = nn.Sequential(
#                 nn.Dropout1d(self.conv_dropout),
#                 nn.ReLU(),
#             )

#         # NOTE: Block order is found in "Identity Mappings in Deep Residual Networks"
#         if self.conv_type == "hierarchical":  # Gradually reduces number of timesteps
#             self.conv1 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 3, 1))
#             self.conv2 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 3, 1))
#             self.conv3 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 3, 1))
#             self.conv4 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 3, 1))
#             self.pool1 = nn.AvgPool1d(2, 2)
#             self.pool2 = nn.AvgPool1d(2, 2)
#             self.pool3 = nn.AvgPool1d(2, 2)
#         elif self.conv_type == "local":  # No reducing number of timesteps
#             self.conv1 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 5, 1, padding='same'))
#             self.conv2 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 5, 1, padding='same'))
#             self.conv3 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 5, 1, dilation=3, padding='same'))
#             self.conv4 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 5, 1, dilation=5, padding='same'))
#         elif self.conv_type == "per_timestep":
#             self.conv1 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 1, 1))
#             self.conv2 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 1, 1))
#             self.conv3 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 1, 1))
#             self.conv4 = spectral_norm_fn(nn.Conv1d(embed_dim, embed_dim, 1, 1))
#         else:
#             raise ValueError("invalid conv_type")

#         # Calculate number of timesteps after conv. 
#         # TODO I recall there is a better way to do this, but could not find yet
#         input = torch.randn((1, self.embed_dim, self.seq_len))  # [B, D, T]. Conv expects this order
#         output = self.conv(input)  # [B, D, T]
#         self.output_seq_len = output.shape[2]

#         # Positional embedding
#         # Note: if absolute positional embedding is added inside the pooling attention,
#         # the embedding size is equal to the number of heads. Otherwise it's the normal embedding dim.
#         if where_to_add_abspos in ["pooling_before_softmax", "pooling_gating"]:
#             if self.pool == "seqpool":
#                 absolute_emb_dim = 1
#             else:
#                 absolute_emb_dim = num_heads
#         else:
#             absolute_emb_dim = embed_dim
#         ClimaX.setup_absolute_posenc(self, pos_encoding, self.output_seq_len, absolute_emb_dim)

#         # Setup pooling
#         ClimaX.setup_pooling(self, embed_dim, num_heads, self.output_seq_len, num_classes)

#         # Only used if pool == 'linear'
#         self.dropout1 = nn.Dropout(0.1)
#         self.act = F.gelu

#         # Only used if where_to_add_abspos is 'start_add'
#         self.pos_drop = nn.Dropout(p=0.1)


#     def conv(self, x):
#         if self.conv_type == "hierarchical":
#             x = self.pool1(x)
#         x = x + self.conv2(self.act2(self.conv1(self.act1(x))))
#         if self.conv_type == "hierarchical":
#             x = self.pool2(x)
#         # x = x + self.conv4(self.act4(self.conv3(self.act3(x))))
#         # if self.conv_type == "hierarchical":
#         #     x = self.pool3(x)
#         return x


#     def forward(self, x, plot_dir=None):
#         """
#         x should have shape [B, T, input_vars]
#         """
#         # Patching if desired
#         x = rearrange(x, "b t_orig v -> b v t_orig")
#         x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [B, V, T (num_patches), P (patch_size)]
#         x = rearrange(x, "b v t p -> b t (v p)")  # [B, T, V*P]
#         x = self.embed_layer(x)  # [B, T, D]

#         # Add ABSOLUTE pos embedding (if adding at start)
#         # At this point, X should be [B, T, D], and pos_embed should be [T, D]. (T = number of patches along time dimension)
#         if self.pos_embed is not None and self.where_to_add_abspos == "start_add":
#             # CURRENT: add the positional embedding
#             x = x + self.pos_embed
#             x = self.pos_drop(x)

#         # Main convolutional (local) backbone
#         x = rearrange(x, 'b t d -> b d t')  # Move the "channel" (D) dimension forward, to [B, D, T]
#         x = self.conv(x)  # Convert to [batch, channel, time']  (may be fewer timesteps)
#         x = rearrange(x, 'b d t -> b t d')  # Change back to [B, T, D] for compatibility with pooling

#         # Pooling
#         preds, pooling_attn = ClimaX.forward_pooling(self, x)

#         # Visualize positional embedding
#         if plot_dir is not None and "learnable" in self.pos_encoding:
#             visualization_utils.visualize_absolute_posenc(self.pos_embed, plot_dir)

#         # Visualize SeqPool attention weights
#         if plot_dir is not None and self.pool in ["seqpool", "seqpool_multihead", "seqpool_multihead_smoothed"]:
#             visualization_utils.visualize_pooling_attn(pooling_attn, plot_dir)

#         return preds, None, pooling_attn


