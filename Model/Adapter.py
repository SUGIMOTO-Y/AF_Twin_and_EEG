import torch
import torch.nn as nn
import numpy as np
import math
from Model.Model import *
from Model.ModelUtility import *

class AdapterCLIP(CLIP):
    def __init__(self,EEGViT_params, IMGViT_params, Adapter_params):
        super(AdapterCLIP, self).__init__(EEGViT_params, IMGViT_params)     
        if Adapter_params['Encoder_option'] == 'both':
            self.EEGEncoder = AdapterEEGVisionTransformer(Adapter_params, **EEGViT_params)
            self.IMGEncoder = AdapterIMGVisionTransformer(Adapter_params, **IMGViT_params)
        elif Adapter_params['Encoder_option'] == 'eeg':
            self.EEGEncoder = AdapterEEGVisionTransformer(Adapter_params, **EEGViT_params)
            self.IMGEncoder = IMGVisionTransformer(**IMGViT_params)
        elif Adapter_params['Encoder_option'] == 'img':
            assert 'No implementation.'
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, EEG, is_step = 1):
        if is_step == 1:
            IMG_features = self.IMGEncoder(image)
            EEG_features = self.EEGEncoder(EEG)
            return IMG_features, EEG_features

class AdapterEEGVisionTransformer(EEGVisionTransformer):
    def __init__(self, Adapter_params, num_electrodes: int, chunk_size: int, Tpatch_size: int,  Cpatch_size: int,  width: int,  layers: int,  heads: int,  embed_dim: int,  ):
        super(AdapterEEGVisionTransformer, self).__init__(num_electrodes, chunk_size, Tpatch_size,  Cpatch_size,  width,  layers,  heads,  embed_dim)
        self.transformer = AdapterTransformer(width, layers, heads, Adapter_params=Adapter_params)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class AdapterIMGVisionTransformer(IMGVisionTransformer):
    def __init__(self, Adapter_params,input_resolution: int, patch_size: int, width: int, layers: int, heads: int, embed_dim: int,):
        super(AdapterIMGVisionTransformer, self).__init__(input_resolution, patch_size, width, layers, heads, embed_dim,)
        self.transformer = AdapterTransformer(width, layers, heads, Adapter_params=Adapter_params)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

class AdapterTransformer(Transformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, Adapter_params=None):
        super(AdapterTransformer, self).__init__(width, layers, heads, attn_mask)
        self.resblocks = nn.Sequential(*[AdapterResidualAttentionBlock(width, heads, attn_mask, Adapter_params) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class AdapterResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, Adapter_params=None):
        super(AdapterResidualAttentionBlock, self).__init__(d_model, n_head, attn_mask)
        self.ffn_option = Adapter_params['ffn_option']
        self.adapter = Adapter(d_model, **Adapter_params)
        
    def forward(self, x: torch.Tensor):
        if self.ffn_option == 'parallel':
            adapt_x = self.adapter(x, add_residual=False)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.ffn_option == 'sequential':
            x = self.adapter(x)
        elif self.ffn_option == 'parallel':
            x = x + adapt_x
        return x
    
##############################################
'''
from https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py#L45
Chen, Shoufa, et al. "Adaptformer: Adapting vision transformers for scalable 
visual recognition." Advances in Neural Information Processing Systems 
35 (2022): 16664-16678.
'''
class Adapter(nn.Module):
    def __init__(self, d_model, down_size, init_option, adapter_scalar, adapter_layernorm_option, ffn_option, Encoder_option, dropout = 0.0, ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = down_size
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        self.scale = nn.Parameter(torch.ones(1))if adapter_scalar == "learnable_scalar" else float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output