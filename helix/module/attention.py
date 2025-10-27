import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange,repeat,reduce,einsum
from jaxtyping import Float, Int,Bool


def mask_factory(mask_type):
    if mask_type == "causal":
        return lambda x,y: torch.tril(torch.ones([x.shape[-2],y.shape[-2]],dtype=torch.bool, device=x.device))
    elif mask_type == "none":
        return lambda x: None
    else:
        raise ValueError(f"{mask_type} not implement")

def multihead_attention(q: Float[Tensor, "... len dim"],
                        k: Float[Tensor, "... len dim"],
                        v: Float[Tensor, "... len dim"],
                        head_num: int = 1,
                        RoPE: bool = False,
                        QKnorm: bool = False,
                        mask:Bool[Tensor, "... len_q len_k"] = None,
                        ):
    # TODO 
    # 补上RoPE部分
    b,l,d = q.shape
    output_dim = d
    assert output_dim%head_num == 0
    head_dim = torch.tensor(output_dim/head_num)
    if RoPE:
        pass
    q = rearrange(q, '... q_l (h h_d) -> ... h q_l h_d',h= head_num)
    k = rearrange(k, '... k_l (h h_d) -> ... h h_d k_l',h= head_num)
    v = rearrange(v, '... k_l (h h_d) -> ... h k_l h_d',h= head_num)
    attn = einsum(q,k,'... q_l h_d,... h_d k_l -> ... q_l k_l')
    if mask is not None:
        assert mask.shape[-2:] == attn.shape[-2:],"mask shape error"
        attn = attn.masked_fill(mask,float('-inf'))
        pass
    attn = attn/torch.sqrt(head_dim)
    attn = torch.softmax(attn,dim=-1)
    result = einsum(attn,v,'... q_l k_l,... k_l h_d -> ... q_l h_d')
    result = rearrange(result,'... h q_l h_d -> ... q_l (h h_d)')
    return result

class GQA(nn.Module):
    def __init__(self,config:dict):
        super().__init__()
        self.embed_dim = config["emb_dim"]
        self.query_head_num = config["query_head_num"]
        self.head_dim = config["head_dim"]
        self.group_num = config["group_num"]
        self.QKnorm = config["QKNorm"]
        self.RoPE = config["RoPE"]
        self.mask_func = mask_factory(config["mask_type"])
        #assert self.embed_dim%self.query_head_num == 0,"embed_dim%query_head_num != 0"
        assert self.query_head_num%self.group_num == 0,"query_head_num%group_num != 0"
        self.repeat_num = int(self.query_head_num/self.group_num)
        self.kv_head_num = int(self.group_num)

        self.W_q = nn.Linear(self.embed_dim,self.query_head_num*self.head_dim)
        self.W_k = nn.Linear(self.embed_dim,self.kv_head_num*self.head_dim)
        self.W_v = nn.Linear(self.embed_dim,self.kv_head_num*self.head_dim)
        self.W_O = nn.Linear(self.query_head_num*self.head_dim,self.embed_dim)

    def forward(self,x:Float[Tensor, "... len dim"]):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        if self.mask_func is not None:
            mask = self.mask_func(q,k)  
        k = repeat(k,'... d -> ... (d n)',n=self.repeat_num)
        v = repeat(v,'... d -> ... (d n)',n=self.repeat_num)
        attn = multihead_attention(q,k,v,mask=mask,head_num=self.query_head_num,QKnorm=self.QKnorm,RoPE=self.RoPE)
        result = self.W_O(attn)
        return result

def test_repeat():
    x = torch.tensor([1,2,3])
    x  = repeat(x, 'd -> (d n)',n=3) #111222333
    x = repeat(x,'d -> (n d)',n=3) #123123123
    print(x)

if __name__ == "__main__":
    test()






