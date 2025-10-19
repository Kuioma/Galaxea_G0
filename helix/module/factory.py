import torch
from torch import Tensor
from jaxtyping import Float
import torch.nn as nn
import torch.nn.functional as F
from attention import GQA


def mlp_factory(cfg:dict):
    dim = cfg["dim"]
    hidden_dim = dim*cfg["hidden_dim_ratio"]
    if cfg["ffn"]["type"] == "SwiGLU":
        hidden_dim = int(2*hidden_dim/3)
        mlp = SwiGLU(dim,hidden_dim)
        return mlp
    elif cfg["ffn"]["type"] == "Linear":
        ffn_1 = nn.Linear(dim,hidden_dim)
        ffn_2 = nn.Linear(hidden_dim,dim)
    else:
        raise ValueError(f"{cfg["ffn"]["type"]} not implement")
    
    if cfg["activate"]["type"] == "ReLU":
        activate_layer = nn.ReLU()
    else:
        raise ValueError(f"{cfg["ffn"]["type"]} not implement")
    
    return nn.Sequential(
        ffn_1,
        activate_layer,
        ffn_2
    )

def norm_factory(cfg:dict):
    if cfg["type"] == "RMSNorm":
        if "eps" in cfg.keys():
            eps = cfg["eps"]
        dim = cfg["dim"]
        return nn.RMSNorm(dim,eps=eps)
    elif cfg["type"] == "LayerNorm":
        if "eps" in cfg.keys():
            eps = cfg["eps"]
        dim = cfg["dim"]
        return nn.LayerNorm(dim,eps=eps)
    else:
        raise ValueError(f"{cfg["type"]} not implement")

def attention_factory(cfg:dict):
    if cfg["type"] == "GQA":
        return GQA(cfg)
    else:
        raise ValueError(f"{cfg["type"]} not implement")


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
    
        