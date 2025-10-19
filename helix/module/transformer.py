import torch
import torch.nn as nn
from attention import multihead_attention
from factory import attention_factory,mlp_factory,norm_factory
# 单decoder
# attention+MLP
# 输入输出维度
# normLayer 类型
# mask类型
# norm layer 位置


class TransformerDecoder(nn.Module):
    def __init__(self,config:dict):
        super().__init__()
        attention_layer = attention_factory(config["attention"])
        mlp_layer = mlp_factory(config["mlp"])
        norm_layer_1 = norm_factory(config["norm"])
        norm_layer_2 = norm_factory(config["norm"])
        self.norm_pos = config["norm_pos"]
        self.layers = nn.ModuleDict({
            'attention': attention_layer,
            'mlp': mlp_layer,
            'norm_layer_1':norm_layer_1,
            'norm_layer_2':norm_layer_2
        })

    def forward(self,x):
        if self.norm_pos == "pre_norm":
            self.pre_norm_forward(x)
        elif self.norm_pos == "post_norm":
            self.post_norm_forward(x)
        elif self.norm_pos == "post_norm_inside":
            self.post_norm_inside_forward(x)

    def pre_norm_forward(self,x):
        residual = x
        x = self.layers['attention'](self.layers['norm_layer_1'](x))
        x = residual + x
        residual = x
        x = self.layers['mlp'](self.layers['norm_layer_2'](x))
        x = residual + x
        return x

    def post_norm_forward(self,x):
        residual = x
        x = self.layers['attention'](x)
        x = self.layers['norm_layer_1'](residual + x)
        residual = x
        x = self.layers['mlp'](x)
        x = self.layers['norm_layer_2'](residual + x)
        return x

    def post_norm_inside_forward(self,x):
        residual = x
        x = self.layers['norm_layer_1'](self.layers['attention'](x))
        x = residual + x
        residual = x
        x = self.layers['norm_layer_2'](self.layers['mlp'](x))
        x = residual + x
        return x