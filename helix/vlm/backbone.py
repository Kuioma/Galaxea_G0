from module import factory
from module.transformer import TransformerDecoder
from jaxtyping import Int,Float
import torch
from torch import Tensor
import torch.nn as nn


class Vlm(nn.Module):
    def __init__(self,
                 cfg:dict):
        super().__init__()
        self.layer_num = cfg["layer_num"]
        self.dim = cfg["emb_dim"]
        self.vocab_size = cfg["vocab_size"]
        self.last_norm_layer = factory.norm_factory(cfg["last_norm_layer"])
        self.output_layer = nn.Linear(self.dim,self.vocab_size)
        self.token_embed_layer = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.dim)
        self.transformer_layers = nn.ModuleList([
            TransformerDecoder(cfg["transformer"]) for i in range(self.layer_num)
        ])

    def forward(self,
                tokens:Int[Tensor,"... l"]):
        transformer_embed = {f"layer_{i}":None for i in range(self.layer_num)}
        tokens_embed = self.token_embed_layer(tokens)
        for layer_index,layer in enumerate(self.transformer_layers):
            tokens_embed = layer(tokens_embed)
            transformer_embed[f"layer_{layer_index}"] = tokens_embed
        tokens_embed = self.last_norm_layer(tokens_embed)
        result = self.output_layer(tokens_embed)
        return result,transformer_embed
    
    def load_param():
        pass