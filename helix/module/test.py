import attention
import factory
import transformer
import torch

def test_multihead_attention():
    # Basic test case
    q = torch.randn(2, 4, 8)  # batch_size=2, seq_len=4, dim=8
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    multihead_attention = attention.multihead_attention
    # Test with single head
    result = multihead_attention(q, k, v, head_num=1)
    assert result.shape == (2, 4, 8)
    
    # Test with multiple heads
    result = multihead_attention(q, k, v, head_num=4)
    assert result.shape == (2, 4, 8)
    
    # Test with mask
    mask = torch.ones(2, 4, 4).bool()
    result = multihead_attention(q, k, v, head_num=2, mask=mask)
    assert result.shape == (2, 4, 8)

def test_gqa():
    # Configuration for GQA
    config = {
        "emb_dim": 64,
        "query_head_num": 8,
        "head_dim": 8,
        "group_num": 4,
        "attention": {
            "head_num": 8,
            "RoPE": False,
            "QKnorm": False
        },
        "mask_type": "causal"
    }
    
    gqa_layer = attention.GQA(config)
    
    # Test forward pass
    x = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, emb_dim=64
    output = gqa_layer(x)
    assert output.shape == (2, 10, 64)
    
    # Test with different sequence length
    x = torch.randn(1, 5, 64)
    output = gqa_layer(x)
    assert output.shape == (1, 5, 64)

import torch
import torch.nn as nn
from helix.module.transformer import TransformerDecoder

def test_transformer_decoder_initialization():
    """Test initialization of TransformerDecoder with different configurations"""
    
    # Configuration for testing
    config = {
        "attention": {
            "type": "GQA",
            "emb_dim": 64,
            "head_dim": 128,
            "group_num": 4,
            "QKNorm": False,
            "RoPE": False,
            "query_head_num": 16,
            "mask_type": "causal"
            # "dropout": 0.1
        },
        "mlp": {
            "ffn":{
                "type": "SwiGLU",
            },
            "activate":{
                "type": "ReLU"
            },
            "hidden_dim_ratio":2,
            "dim": 64,
        },
        "norm": {
            "type": "RMSNorm",
            "eps": 1e-5,
            "dim": 64
        },
        "norm_pos": "pre_norm"
    }
    
    # Test initialization
    decoder = TransformerDecoder(config)
    assert isinstance(decoder, nn.Module)
    assert hasattr(decoder, 'layers')
    assert hasattr(decoder, 'norm_pos')

def test_transformer_decoder_pre_norm():
    """Test TransformerDecoder with pre_norm configuration"""
    
    config = {
        "attention": {
            "type": "GQA",
            "emb_dim": 64,
            "head_dim": 128,
            "group_num": 4,
            "query_head_num": 16,
            "mask_type": "causal",
            "RoPE": False,
            "QKNorm": False,
            # "dropout": 0.1
        },
        "mlp": {
            "ffn":{
                "type": "SwiGLU",
            },
            "activate":{
                "type": "ReLU"
            },
            "hidden_dim_ratio":2,
            "dim": 64,
        },
        "norm": {
            "type": "RMSNorm",
            "eps": 1e-5,
            "dim": 64
        },
        "norm_pos": "pre_norm"
    }
    
    decoder = TransformerDecoder(config)
    x = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, emb_dim=64
    output = decoder(x)
    assert output.shape == x.shape

def test_transformer_decoder_post_norm():
    """Test TransformerDecoder with post_norm configuration"""
    
    config = {
        "attention": {
            "type": "GQA",
            "emb_dim": 64,
            "head_dim": 128,
            "group_num": 4,
            "query_head_num": 16,
            "mask_type": "Casual"
            # "dropout": 0.1
        },
        "mlp": {
            "ffn":{
                "type": "Swish",
            },
            "activate":{
                "type": "ReLU"
            },
            "hidden_dim_ratio":2,
            "dim": 64,
        },
        "norm": {
            "type": "RMSNorm",
            "eps": 1e-5,
            "dim": 64
        },
        "norm_pos": "post_norm"
    }
    
    decoder = TransformerDecoder(config)
    x = torch.randn(2, 10, 64)
    output = decoder(x)
    assert output.shape == x.shape

def test_transformer_decoder_post_norm_inside():
    """Test TransformerDecoder with post_norm_inside configuration"""
    
    config = {
        "attention": {
            "type": "GQA",
            "emb_dim": 64,
            "head_dim": 128,
            "group_num": 4,
            "query_head_num": 16,
            "mask_type": "Casual"
            # "dropout": 0.1
        },
        "mlp": {
            "ffn":{
                "type": "Swish",
            },
            "activate":{
                "type": "ReLU"
            },
            "hidden_dim_ratio":2,
            "dim": 64,
        },
        "norm": {
            "type": "RMSNorm",
            "eps": 1e-5,
            "dim": 64
        },
        "norm_pos": "post_norm_inside"
    }
    
    decoder = TransformerDecoder(config)
    x = torch.randn(2, 10, 64)
    output = decoder(x)
    assert output.shape == x.shape

def test_transformer_decoder_different_sizes():
    """Test TransformerDecoder with different input sizes"""
    
    config = {
        "attention": {
            "type": "GQA",
            "head_dim": 64,
            "query_head_num": 16,
            "group_num": 4,
            "mask_type": "Casual"
            # "dropout": 0.1
        },
        "mlp": {
            "ffn":{
                "type": "Swish",
            },
            "activate":{
                "type": "ReLU"
            },
            "hidden_dim_ratio":2,
            "dim": 64,
        },
        "norm": {
            "type": "RMSNorm",
            "eps": 1e-5,
            "dim": 64
        },
        "norm_pos": "pre_norm"
    }
    
    decoder = TransformerDecoder(config)
    
    # Test with different batch sizes and sequence lengths
    x1 = torch.randn(1, 5, 128)
    output1 = decoder(x1)
    assert output1.shape == x1.shape
    
    x2 = torch.randn(4, 20, 128)
    output2 = decoder(x2)
    assert output2.shape == x2.shape

def test_transformer_decoder_gradient_flow():
    """Test that gradients flow properly through the decoder"""
    
    config = {
        "attention": {
            "type": "multihead",
            "emb_dim": 64,
            "head_num": 8,
            "dropout": 0.0  # Disable dropout for deterministic testing
        },
        "mlp": {
            "type": "feedforward",
            "emb_dim": 64,
            "hidden_dim": 128,
            "dropout": 0.0  # Disable dropout for deterministic testing
        },
        "norm": {
            "type": "layer_norm",
            "dim": 64
        },
        "norm_pos": "pre_norm"
    }
    
    decoder = TransformerDecoder(config)
    decoder.train()  # Set to training mode
    
    x = torch.randn(2, 10, 64, requires_grad=True)
    output = decoder(x)
    
    # Compute a simple loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape

if __name__ == "__main__":
    # test_transformer_decoder_initialization()
    test_transformer_decoder_pre_norm()
    # test_transformer_decoder_post_norm()
    # test_transformer_decoder_post_norm_inside()
    # test_transformer_decoder_different_sizes()
    # test_transformer_decoder_gradient_flow()
    print("All tests passed!")