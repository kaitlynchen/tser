from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from einops import rearrange
from collections import OrderedDict
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from models.ts_climax import TransformerEncoder
from utils import utils

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                                             norm=config['normalization_layer'], freeze=config['freeze'],
                                             attention_type=config['attention_type'])
        elif config['model'] == 'transformer':
            return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'],
                                        attention_type=config['attention_type'])

    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                            config['num_heads'],
                                                            config['num_layers'], config['dim_feedforward'],
                                                            num_classes=num_labels,
                                                            dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                            activation=config['activation'],
                                                            norm=config['normalization_layer'], freeze=config['freeze'],
                                                            attention_type=config['attention_type'])
        elif config['model'] == 'transformer':
            return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                       config['num_heads'],
                                                       config['num_layers'], config['dim_feedforward'],
                                                       num_classes=num_labels,
                                                       dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                       activation=config['activation'],
                                                       norm=config['normalization_layer'], freeze=config['freeze'],
                                                       include_cls_token=config["class_token"],
                                                       attention_type=config['attention_type'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(
        "activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        # this stores the variable in the state_dict (used for non-trainable variables)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        # requires_grad automatically set to True
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", attention_type='dot'):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        
        if attention_type == 'L2':
            self.self_attn = SimpleL2Attention(d_model, nhead, dropout, attention_type)
            self.custom_attn = True
        else: 
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
            self.custom_attn = False
            
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # normalizes each feature across batch samples and time steps
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs): # modified https://stackoverflow.com/questions/77078717/typeerror-transformerbatchnormencoderlayer-forward-got-an-unexpected-keyword
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.custom_attn:
            src_batch_first = src.permute(1, 0, 2)  # [T, B, D] -> [B, T, D]
            src2, attn_weights = self.self_attn(src_batch_first, src_batch_first, src_batch_first)
            src2 = src2.permute(1, 0, 2)  # [B, T, D] -> [T, B, D]
        else:
            src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask, 
                                  need_weights=True, average_attn_weights=False)
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src, attn_weights


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, attention_type='dot'):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation, attention_type=attention_type)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks, **kwargs):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # (seq_length, batch_size, d_model)
        output, attn_weights_layers, embeddings_layers = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks)
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(output)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        # (batch_size, seq_length, feat_dim)
        output = self.output_layer(output)
        
        # Plot attention matrices if plot_dir is provided
        plot_dir = kwargs.get('plot_dir', None)
        if plot_dir is not None:
            self._plot_attention_matrices(attn_weights_layers, plot_dir)

        return output
    
    def _plot_attention_matrices(self, attn_weights_layers, plot_dir):
        """
        Plot attention matrices.
        
        Args:
            attn_weights_layers: [L, B, H, T, T] or [L, B, T, T] attention weights
            plot_dir: directory to save plots
        """
        if len(attn_weights_layers.shape) == 5:
            # Reshape to [B, L*H, T, T]
            attn_matrices = rearrange(attn_weights_layers, "l b h t0 t1 -> b (l h) t0 t1")
            num_layers = attn_weights_layers.shape[0]
            num_heads = attn_weights_layers.shape[2]
        else:
            # Reshape to [B, L, T, T] and treat each layer as one "head"
            attn_matrices = rearrange(attn_weights_layers, "l b t0 t1 -> b l t0 t1")
            num_layers = attn_weights_layers.shape[0]
            num_heads = 1
        
        # Plot attention matrices: for each example, plot random subset of layers/heads
        min_value, max_value = utils.approx_min_max(attn_matrices)
        
        # Number of examples to plot
        n_rows = min(5, attn_matrices.shape[0])  # Plot up to 5 examples
        
        # Random subset of heads/layers
        n_matrices = attn_matrices.shape[1]  
        n_cols = min(num_layers * 2 if num_heads > 1 else num_layers, n_matrices) 
        
        matrix_indices = np.sort(np.random.choice(np.arange(n_matrices), n_cols, replace=False))
        fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout="constrained")
        
        for r in range(n_rows):
            for c in range(n_cols):
                m = matrix_indices[c]
                im = axeslist[r, c].imshow(
                    attn_matrices[r, m, :, :].detach().cpu().numpy(), 
                    vmin=min_value, vmax=max_value
                )
                if r == 0:
                    if num_heads > 1:
                        layer_idx = m // num_heads
                        head_idx = m % num_heads
                        axeslist[r, c].set_title(f"Layer {layer_idx}, Head {head_idx}")
                    else:
                        axeslist[r, c].set_title(f"Layer {m} (averaged)")
                if c == 0:
                    axeslist[r, c].set_ylabel(f"Example {r+1}", rotation=0, size='large', labelpad=30)
        
        fig.colorbar(im, ax=axeslist, shrink=0.4)
        fig.suptitle("Example attention matrices (TSTransformerEncoder)")
        plt.savefig(os.path.join(plot_dir, 'attention_matrices_encoder.png'))
        plt.close()


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, include_cls_token=False, attention_type='dot'):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.include_cls_token = include_cls_token

        if include_cls_token:
            self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len+1)
            self.output_layer = self.build_output_module(d_model, max_len + 1, num_classes)
        else:
            self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)
            self.output_layer = self.build_output_module(d_model, max_len, num_classes)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, feat_dim))

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation, attention_type=attention_type)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks=None, **kwargs):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        if padding_masks is None:
            padding_masks = torch.ones((X.shape[0], X.shape[1]), dtype=torch.bool, device=X.device)

        if self.include_cls_token:
            # Expand the class token to the full batch
            n = X.shape[0] # batch_size
            batch_class_token = self.class_token.expand(n, -1, -1)
            X = torch.cat([batch_class_token, X], dim=1)
            # FIXME: hard coded for cuda
            pad_class_token = torch.ones((X.shape[0], 1), dtype=torch.bool).cuda()
            padding_masks = torch.cat((padding_masks, pad_class_token), dim=1)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding

        output, attn_weights_layers, embeddings_layers = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        if self.include_cls_token:
            # Classifier "token" as used by standard language architectures
            output = output[:, 0]
        else:
            output = self.act(output) # the output transformer encoder/decoder embeddings don't include non-linearity
            output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
            output = self.dropout1(output)

        output = output * padding_masks.unsqueeze(-1) # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1) # (batch_size, seq_length * d_model)

        output = self.output_layer(output)  # (batch_size, num_classes)
        
        plot_dir = kwargs.get('plot_dir', None)
        if plot_dir is not None:
            self._plot_attention_matrices(attn_weights_layers, plot_dir)
        
        return output, attn_weights_layers, None

    def predict_summed(self, func, input, padding_masks):
        """Helper function for jacobian computation"""
        return func(input, padding_masks).sum()

    def jacobian_loss(self, input, padding_masks=None):
        """
        Computes geometric complexity: gradient of model output w.r.t. inputs.
        Penalizes the L2 norm per example (input gradient regularization).
        
        Args:
            input: [B, T, V] input tensor
            padding_masks: [B, T] boolean mask (1=keep, 0=padding). Optional.
        
        Returns:
            scalar loss - mean Frobenius norm of input Jacobians
        """        
        was_training = self.training
        self.eval()
        
        if padding_masks is None:
            padding_masks = torch.ones((input.shape[0], input.shape[1]), 
                                      dtype=torch.bool, device=input.device)
        
        X_min = input.min(dim=0, keepdim=True).values
        X_max = input.max(dim=0, keepdim=True).values
        random_vals = torch.rand_like(input, device=input.device) * (X_max - X_min) + X_min
        
        # Compute Jacobian: gradient of output w.r.t. input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            jacobian = torch.func.jacrev(self.predict_summed, argnums=1)(
                self.forward, random_vals, padding_masks
            )  # [B, T, V]
        
        jacobian = jacobian.reshape(jacobian.shape[0], -1)  # [B, T*V]
        jacobian_loss = torch.linalg.norm(jacobian, dim=1).mean()
        
        self.train(was_training)
        return jacobian_loss

    def _plot_attention_matrices(self, attn_weights_layers, plot_dir):
        """
        Plot attention matrices.
        
        Args:
            attn_weights_layers: [L, B, H, T, T] or [L, B, T, T] attention weights
            plot_dir: directory to save plots
        """
        # Check if head dimension exists
        if len(attn_weights_layers.shape) == 5:
            # Reshape to [B, L*H, T, T]
            attn_matrices = rearrange(attn_weights_layers, "l b h t0 t1 -> b (l h) t0 t1")
            num_layers = attn_weights_layers.shape[0]
            num_heads = attn_weights_layers.shape[2]
        else:
            # Reshape to [B, L, T, T] and treat each layer as one "head"
            attn_matrices = rearrange(attn_weights_layers, "l b t0 t1 -> b l t0 t1")
            num_layers = attn_weights_layers.shape[0]
            num_heads = 1
        
        # Plot attention matrices: for each example, plot random subset of layers/heads
        min_value, max_value = utils.approx_min_max(attn_matrices)
        
        # Number of examples to plot
        n_rows = min(5, attn_matrices.shape[0])  # Plot up to 5 examples
        
        # Random subset of heads/layers
        n_matrices = attn_matrices.shape[1]
        n_cols = min(num_layers * 2 if num_heads > 1 else num_layers, n_matrices) 
        
        matrix_indices = np.sort(np.random.choice(np.arange(n_matrices), n_cols, replace=False))
        fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), layout="constrained")
        
        for r in range(n_rows):
            for c in range(n_cols):
                m = matrix_indices[c]
                im = axeslist[r, c].imshow(
                    attn_matrices[r, m, :, :].detach().cpu().numpy(), 
                    vmin=min_value, vmax=max_value
                )
                if r == 0:
                    if num_heads > 1:
                        layer_idx = m // num_heads
                        head_idx = m % num_heads
                        axeslist[r, c].set_title(f"Layer {layer_idx}, Head {head_idx}")
                    else:
                        axeslist[r, c].set_title(f"Layer {m} (averaged)")
                if c == 0:
                    axeslist[r, c].set_ylabel(f"Example {r+1}", rotation=0, size='large', labelpad=30)
        
        fig.colorbar(im, ax=axeslist, shrink=0.4)
        fig.suptitle("Example attention matrices")
        plt.savefig(os.path.join(plot_dir, 'attention_matrices.png'))
        plt.close()


class SimpleL2Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, attention_type='dot'):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.scale = emb_size ** -0.5 
        # make scale a learnable parameter
        # self.scale = nn.Parameter(torch.ones(num_heads) * (emb_size ** -0.5))
        
        self.batch_first = False
        self._qkv_same_embed_dim = True
        
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        
        self.key.weight.data.copy_(torch.eye(emb_size))
        self.value.weight.data.copy_(torch.eye(emb_size))
        self.query.weight.data.copy_(torch.eye(emb_size))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, **kwargs):
        k = self.key(key)  # [B, T, D]
        k = rearrange(k, 'b t (h d_h) -> b h d_h t', h=self.num_heads)  # [B, H, d_head, T]
        q = self.query(query)  # [B, T, D]
        q = rearrange(q, 'b t (h d_h) -> b h t d_h', h=self.num_heads)  # [B, H, T, d_head]
        v = self.value(value)  # [B, T, D]
        v = rearrange(v, 'b t (h d_h) -> b h t d_h', h=self.num_heads)  # [B, H, T, d_head]

        if self.attention_type == 'L2':
            # L2 attention computation
            q_norm_sq = torch.sum(q**2, dim=-1, keepdim=True)  # [B, H, T, 1]
            k_norm_sq = torch.sum(k**2, dim=-2, keepdim=True)  # [B, H, 1, T]
            dot_product = torch.matmul(q, k)  # [B, H, T, T]
            content_attn = -0.5 * (q_norm_sq + k_norm_sq - 2 * dot_product) * self.scale
        else:
            content_attn = torch.matmul(q, k) * self.scale  # [B, H, T, T]

        attn = F.softmax(content_attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, H, T, d_head]
        out = rearrange(out, 'b h t d_h -> b t (h d_h)')  # [B, T, D]
        
        return out, attn.mean(dim=1)  
