"""
Code for 'Local CNN' baselines, which are highly-local convolutional
networks (or per-timestep MLP), followed by global pooling (optional).

We use the following letters to annotate shapes:
B: batch (examples)
T: timesteps
T': timesteps after CNN
D: 'channels' (embedding dimension)
"""



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils, visualization_utils
from einops import rearrange



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

        # Embedding layer
        self.embed_layer = nn.Linear(patch_size*in_channels, embed_dim)  # Each patch has patch_size*num_variables (P*V) elements. Map to embed_dim (D).
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
        else:
            raise ValueError("invalid conv_type")

        # Calculate number of timesteps after conv. 
        # TODO I recall there is a better way to do this, but could not find yet
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
            absolute_emb_dim = embed_dim
        utils.setup_absolute_posenc(self, pos_encoding, self.output_seq_len, absolute_emb_dim)

        # Setup pooling
        utils.setup_pooling(self, embed_dim, num_heads, self.output_seq_len, num_classes)

        # Only used if pool == 'linear'
        self.dropout1 = nn.Dropout(0.1)
        self.act = F.gelu

        # Only used if where_to_add_abspos is 'start_add'
        self.pos_drop = nn.Dropout(p=0.1)


    def forward(self, x, plot_dir=None):
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

        # Main convolutional (local) backbone
        x = rearrange(x, 'b t d -> b d t')  # Move the "channel" (D) dimension forward, to [B, D, T]
        x = self.conv(x)  # Convert to [batch, channel, time']  (may be fewer timesteps)
        x = rearrange(x, 'b d t -> b t d')  # Change back to [B, T, D] for compatibility with pooling

        # Pooling
        preds, pooling_attn = utils.forward_pooling(self, x)

        # Visualize positional embedding
        if plot_dir is not None and "learnable" in self.pos_encoding:
            visualization_utils.visualize_absolute_posenc(self.pos_embed, plot_dir)

        # Visualize SeqPool attention weights
        if plot_dir is not None and self.pool in ["seqpool", "seqpool_multihead", "seqpool_multihead_smoothed"]:
            visualization_utils.visualize_pooling_attn(pooling_attn, plot_dir)

        return preds, None, pooling_attn

