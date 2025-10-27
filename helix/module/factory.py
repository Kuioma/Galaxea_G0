from typing import Optional
import torch
from torch import Tensor
from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
import math
from .attention import GQA



def mlp_factory(cfg:dict):
    dim = cfg["dim"]
    hidden_dim = int(dim*cfg["hidden_dim_ratio"])
    if "output_dim" in cfg.keys():
        output_dim = cfg["output_dim"]
    else:
        output_dim = None
    if cfg["ffn"]["type"] == "SwiGLU":
        hidden_dim = int(2*hidden_dim/3)
        mlp = SwiGLU(dim,hidden_dim)
        return mlp
    elif ffn_type == "Linear":
        ffn_1 = nn.Linear(dim,hidden_dim)
        ffn_2 = nn.Linear(hidden_dim,output_dim) if output_dim else nn.Linear(hidden_dim,dim)
    else:
        raise ValueError(f"{ffn_type} not implement")
    
    if activate_type == "ReLU":
        activate_layer = nn.ReLU()
    elif cfg["activate"]["type"] == "SiLU":
        activate_layer = nn.SiLU()
    else:
        raise ValueError(f"{activate_type} not implement")
    
    return nn.Sequential(
        ffn_1,
        activate_layer,
        ffn_2
    )

def norm_factory(cfg:dict):
    norm_type = cfg["type"]
    if norm_type == "RMSNorm":
        if "eps" in cfg.keys():
            eps = cfg["eps"]
        dim = cfg["dim"]
        return nn.RMSNorm(dim,eps=eps)
    elif norm_type == "LayerNorm":
        if "eps" in cfg.keys():
            eps = cfg["eps"]
        else: eps = 0.00001
        dim = cfg["dim"]
        return nn.LayerNorm(dim,eps=eps)
    elif cfg["type"] == "AdaLayerNorm":
        if "eps" in cfg.keys():
            eps = cfg["eps"]
        else:
            eps = None
        dim = cfg["dim"]
        return AdaLayerNorm(dim, norm_eps=eps)
    else:
        raise ValueError(f"{norm_type} not implement")

def attention_factory(cfg:dict):
    attention_type = cfg["type"]
    if attention_type == "GQA":
        return GQA(cfg)
    else:
        raise ValueError(f"{attention_type} not implement")

def positional_emb_factory(cfg:dict):
    if cfg['type'] == "sinusoidal":
        return SinusoidalPositionalEmb(embed_dim=cfg['hidden_dim'], max_seq_length=cfg['seq_len'])

class SinusoidalPositionalEmb(nn.Module):
    """
    from diffusers
    Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x

class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model:int,
                  hidden_dim:int):
        super().__init__()        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # 门控值
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False) # 压缩层
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) # 被门控的值

    def forward(self, x:Float[Tensor,"... len dim"]):
        gate = self.w1(x)           
        gated_value = self.w3(x)     
        # 激活门并与值相乘
        activated_gate = F.silu(gate) # SiLU (Swish) 激活
        gated = activated_gate * gated_value # 逐元素相乘

        output = self.w2(gated)     
        return output
    
        