"""
stictched together by code from nanogpt and vit implementation in huggingface

Some layer map:
                                                    huggingface    <--->     nanogpt
       
    model.encoder.layer[0].attention.attention (ViTSelfAttention)  <--->  QKV
    model.encoder.layer[0].attention.output (ViTSelfOutput)        <--->  multi-head fusing
    model.encoder.layer[0].intermediate.dense (ViTIntermediate)    <--->  c_fc 
    model.encoder.layer[0].output.dense (ViTOutput)                <--->  c_proj 
    model.encoder.layer[0].layernorm_before                        <--->  ln_1
    model.encoder.layer[0].layernorm_after                         <--->  ln_2
    
    model.layernorm                                                <--->  ln_f
    
note: in ViT layernorm has no bias, it's just 
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
"""

import math
from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig():
    
    def __init__(self, model_size):
        self.block_size: int = 1024
        self.dropout: float = 0.0
        self.bias: bool = True 
        
        if model_size == "small":
            self.n_layer: int = 12
            self.n_head: int = 12
            self.n_embd: int = 768
        
        if model_size == 'medium':
            self.n_layer = 24
            self.n_head = 16
            self.n_embd = 1024
        
        if model_size == 'large':
            self.n_layer = 36
            self.n_head = 20
            self.n_embd = 1280
        

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_score = False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if not attn_score:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head 
            y = self.resid_dropout(self.c_proj(y))
            return y, att
            
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT_Classification(nn.Module):

    def __init__(self, config, huggingface_vit, num_class, return_logits = False):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self.return_logits = return_logits

        self.transformer = nn.ModuleDict(dict(
            pte = copy.deepcopy(huggingface_vit.vit.embeddings), # picture to embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer block
            ln_f = nn.LayerNorm(config.n_embd)), # laynorm before classification layer
        )
        self.classifier = nn.Linear(config.n_embd, num_class)
        self.softmax = nn.Softmax(dim=1)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
        self.init_from_huggingface(huggingface_vit)
    
    def init_from_huggingface(self, huggingface_vit):
        for idx in range(len(self.transformer.h)):

            self.transformer.h[idx].attn.c_attn_q.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.query.weight.data)
            self.transformer.h[idx].attn.c_attn_k.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.key.weight.data)
            self.transformer.h[idx].attn.c_attn_v.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.value.weight.data)
            
            self.transformer.h[idx].attn.c_attn_q.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.query.bias.data)
            self.transformer.h[idx].attn.c_attn_k.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.key.bias.data)
            self.transformer.h[idx].attn.c_attn_v.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.attention.value.bias.data)

            
            self.transformer.h[idx].attn.c_proj.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.output.dense.weight.data)
            self.transformer.h[idx].attn.c_proj.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].attention.output.dense.bias.data)
            
            self.transformer.h[idx].mlp.c_fc.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].intermediate.dense.weight.data)
            self.transformer.h[idx].mlp.c_fc.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].intermediate.dense.bias.data)
            
            self.transformer.h[idx].mlp.gelu = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].intermediate.intermediate_act_fn)
            
            self.transformer.h[idx].mlp.c_proj.weight.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].output.dense.weight.data)
            self.transformer.h[idx].mlp.c_proj.bias.data = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].output.dense.bias.data)
            
            self.transformer.h[idx].ln_1 = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].layernorm_before)
            self.transformer.h[idx].ln_2 = copy.deepcopy(huggingface_vit.vit.encoder.layer[idx].layernorm_after)
            
        self.transformer.ln_f = copy.deepcopy(huggingface_vit.vit.layernorm)
        
    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        return n_params

    def forward(self, pixel_values, interpolate_pos_encoding = None):
        
        # pixel_value: (batch_size, num_channels, height, width)
        
        device = pixel_values.device

        x = self.transformer.pte(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        for block in self.transformer.h:
            block.attn.flash = False
            x = block(x)
        x = self.transformer.ln_f(x) #[B, T, D]

        logits = self.classifier(x[:, 0, :]) # [B, D] take [CLS]
        output = self.softmax(logits)
        
        if self.return_logits:
            return logits
        else:
            return output
