from typing import Optional
import math
import torch
import torch.nn as nn
from einops import rearrange

from .factory import norm_factory, positional_emb_factory, mlp_factory
from .attention import multihead_attention


class TimestepEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.max_period = max_period

        # 两层 MLP, SiLU 激活
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor):
        """
        timesteps: (batch,) 的整数或浮点时间步
        """
        # ---------- Sinusoidal embedding ----------
        half_dim = self.mlp[0].in_features // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )
        args = torch.einsum('b,d->bd', timesteps.float(), freqs)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.mlp[0].in_features % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        # ---------- Pass through MLP ----------
        emb = self.mlp(emb)
        return emb

class BasicCrossAttentionTransformerBlock(nn.Module):
    def __init__(
            self,
            query_dim: int,
            context_dim: int,
            n_head: int,
            hidden_dim: int,
            dropout = 0.0,
            norm_type: str = "Layernorm",
            positional_emb_type: Optional[str] = None,
            max_seq_len_in_sin_pos_emb: Optional[int] = None

    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.positional_emb_type = positional_emb_type
        self.dropout = dropout
        self.norm_type = norm_type

        if positional_emb_type == "sinusoidal":
            self.positional_emb = positional_emb_factory({"type": "sinusoidal", "hidden_dim": self.hidden_dim, "seq_len": max_seq_len_in_sin_pos_emb})
        else:
            self.positional_emb = None
        
        self.q_proj = nn.Linear(self.query_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.context_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.context_dim, self.hidden_dim)

        self.norm1 = norm_factory({"type": self.norm_type, "dim": hidden_dim})
        self.dropout1 = nn.Dropout(self.dropout)

        # MLP
        self.norm2 = norm_factory({"type": self.norm_type, "dim": hidden_dim})
        self.mlp = mlp_factory({"dim": hidden_dim, "ffn":{"type": "Linear"}, "activate":{"type": "ReLU"}, "hidden_dim_ratio":2}) #geglu

        self.final_dropout = nn.Dropout(self.dropout)

    def forward(self, query, context, attention_mask):
        # TODO: gr00t 会多输入一个t_emb去在做norm的时候多做一个W&B（adanorm）
        
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)
        if self.positional_emb:
            q = self.positional_emb(q)
            k = self.positional_emb(k)

        attn_out = multihead_attention(q,k,v,self.n_head,mask=attention_mask)
        attn_out = self.dropout1(self.norm1(attn_out))

        out = self.mlp(attn_out)
        out = self.final_dropout(out)
        out = self.norm2(out+q)
        return out


class DiT(nn.Module):
    def __init__(self,
                 query_dim,
                 context_dim,
                 hidden_dim,
                 n_head,
                 n_layers: int,
                 output_dim: int,
                 dropout = 0.0,
                 norm_type: str = "Layernorm",
                 positional_emb_type: Optional[str] = None,
                 max_seq_len_in_sin_pos_emb: Optional[int] = None,
                 eps: int = 1e-5
                 ):
        super().__init__()

        # self.timestep_encoder = TimestepEncoder(input_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicCrossAttentionTransformerBlock(
                query_dim=query_dim,
                context_dim=context_dim,
                n_head=n_head,
                hidden_dim=hidden_dim,
                dropout=dropout,
                norm_type=norm_type,
                positional_emb_type=positional_emb_type,
                max_seq_len_in_sin_pos_emb=max_seq_len_in_sin_pos_emb               
            ) for _ in range(n_layers)
        ])

        self.norm = norm_factory({"type": norm_type, "dim": hidden_dim, "eps":eps})
        # self.norm = norm_factory({"type":"AdaLayerNorm", "dim": hidden_dim, "eps":eps})
        self.proj_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, query, context, attention_mask): #TODO: timestep好像暂时没有用到
        # t_emb = self.timestep_encoder(timestep)

        for idx, block in enumerate(self.transformer_blocks):
            out = block(query, context, attention_mask)
        out = self.norm(out)
        return self.proj_out(out)


if __name__ == "__main__":
    query_dim = 20
    context_dim = 25
    hidden_dim = 10
    n_head = 2
    n_layers = 5
    output_dim = 20
    norm_type: str = "LayerNorm"
    positional_emb_type: Optional[str] = "sinusoidal"
    max_seq_len_in_sin_pos_emb = 1000
    dit = DiT(query_dim, context_dim, hidden_dim, n_head, n_layers, output_dim, norm_type=norm_type, positional_emb_type=positional_emb_type, max_seq_len_in_sin_pos_emb=max_seq_len_in_sin_pos_emb)

    seq_l = 12
    q = torch.rand((5,seq_l, query_dim))
    context = torch.rand(5,seq_l+1, context_dim)
    attention_mask = torch.triu(torch.ones(q.shape[1], context.shape[1]), diagonal=1).bool()

    out_tensor = dit(q,context,attention_mask)