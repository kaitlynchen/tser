from typing import Optional, Any
import copy
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d
from timm.models.vision_transformer import Block, trunc_normal_
from timm.models.layers import DropPath

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

    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'convit':
            return ClimaX(list([feat_dim]), img_size=list(data.feature_df.shape), max_seq_len=max_seq_len, patch_size=config['patch_length'],
                          stride=config['stride'], embed_dim=config['d_model'], depth=config['num_layers'], decoder_depth=config['num_decoder_layers'],
                          num_heads=config['num_heads'], feedforward_dim=config['dim_feedforward'], num_classes=num_labels, freeze=config['freeze'],
                          local_up_to_layer=config['num_gpsa_layers'], smooth_attention=config['smooth_attention'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))
    
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
        depth=12,
        decoder_depth=2,
        num_heads=16,
        feedforward_dim=256,
        num_classes=0,
        freeze=False,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        local_up_to_layer=10,
        locality_strength=1.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.BatchNorm1d,
        smooth_attention=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.default_vars = default_vars
        self.max_len = max_seq_len
        self.local_up_to_layer = local_up_to_layer
        self.unfold_dim = int((max_seq_len - patch_size) / stride + 1)
        self.will_smooth = smooth_attention

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.var_embed = self.create_var_embedding(embed_dim)
        self.embed_layer = nn.Linear(patch_size, embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(int((max_seq_len - patch_size) / stride + 1), embed_dim), requires_grad=True)

        # classifier token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * patch_size - 1), num_heads))  # 2*Wt-1, nH

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, patch_dim=self.unfold_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                   norm_layer=norm_layer, use_gpsa=True, locality_strength=locality_strength)
                if i < local_up_to_layer else 
                Block(dim=embed_dim, patch_dim=self.unfold_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                   drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                   use_gpsa=False)
                for i in range(depth)
            ]
        )
        # self.norm = nn.LayerNorm(embed_dim // 2)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, embed_dim // 2))
        self.head = nn.Sequential(*self.head)

        self.norm = norm_layer(embed_dim)
        self.head_linear = nn.Linear(embed_dim, embed_dim // 2)

        # --------------------------------------------------------------------------

        self.initialize_weights()

        # final linear layer: embed_dim * output of unfold * classifier token size
        self.output_layer = nn.Linear(embed_dim * (int((max_seq_len - patch_size) / stride + 1) + 1), num_classes) if num_classes > 0 else nn.Identity()

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
        B = x.shape[0]

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
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # add pos embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        for u, blk in enumerate(self.blocks):
            has_cls_token = u >= self.local_up_to_layer
            if u == self.local_up_to_layer:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x, has_cls_token)
            
        x = x.permute(0, 2, 1) # B, D, L
        x = self.norm(x)

        return x

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: `[batch_size, seq_length, feat_dim]` shape. 
        Returns:
            preds (torch.Tensor): `[B]` shape. Predicted output.
        """
        preds = self.forward_encoder(x)
        preds = preds.reshape(preds.shape[0], -1)
        preds = self.output_layer(preds)       

        return preds

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)       
        self.v = nn.Linear(dim, dim, bias=qkv_bias)       
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(2, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape        
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = torch.unsqueeze(self.rel_indices, 0).unsqueeze(-1)
        pos_score = pos_score.expand(B, -1, -1, 2).float()
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):
        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5 # don't remove this
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    def local_init(self, locality_strength=1.):
        
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = self.num_heads
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            position = h1
            self.pos_proj.weight.data[position,1] = -1
            self.pos_proj.weight.data[position,0] = 2*(h1-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = num_patches
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        ind = ind**2 # TODO: square or abs value?

        device = self.qk.weight.device
        self.rel_indices = ind.to(device)

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or 1/head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = N 
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        ind = ind**2
        distances = ind.to('cuda')

        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= N
        
        if return_map:
            return dist, attn_map
        else:
            return dist
            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, patch_dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm1d, use_gpsa=True, **kwargs):
        super().__init__()
        self.patch_dim = patch_dim
        self.norm1 = norm_layer(patch_dim)
        self.norm_cls = norm_layer(patch_dim + 1)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(patch_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, has_cls_token=False):
        if has_cls_token:
            x = self.norm_cls(x)
        else:
            x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))

        if has_cls_token:
            x = self.norm_cls(x)
        else:
            x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        return x
