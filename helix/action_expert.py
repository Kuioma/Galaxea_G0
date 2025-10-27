import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from einops import rearrange

from module.modeling_DiT import DiT
from module.factory import mlp_factory

class FlowMatchingActionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config.action_expert
        self.model = DiT(
            query_dim=self.cfg.DiT.query_dim,
            context_dim=self.cfg.DiT.context_dim,
            hidden_dim=self.cfg.DiT.hidden_dim,
            n_head=self.cfg.DiT.n_head,
            n_layers=self.cfg.DiT.n_layers,
            output_dim=self.cfg.DiT.output_dim,
            norm_type=self.cfg.DiT.norm_type,
            positional_emb_type=self.cfg.DiT.positional_emb_type,
            max_seq_len_in_sin_pos_emb=self.cfg.DiT.max_seq_len_in_sin_pos_emb)
        self.beta_dist = Beta(self.cfg.Beta.alpha, self.cfg.Beta.beta)
        self.state_encoder = mlp_factory({"dim": self.cfg.State.dim, "output_dim": self.cfg.DiT.query_dim, "ffn":{"type": "Linear"}, "activate":{"type": "ReLU"}, "hidden_dim_ratio":2})
        self.action_encoder = mlp_factory({"dim": self.cfg.Action.dim, "output_dim": self.cfg.DiT.query_dim, "ffn":{"type": "Linear"}, "activate":{"type": "ReLU"}, "hidden_dim_ratio":2})
        self.action_decoder = mlp_factory({"dim": self.cfg.DiT.output_dim, "output_dim": self.cfg.Action.dim, "ffn":{"type": "Linear"}, "activate":{"type": "ReLU"}, "hidden_dim_ratio":0.5})

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_sample - sample) / self.config.noise_sample #XXX: 不是很清楚为什么要这样设置,gr00t里的默认值是0.999
    
    def forward(self, backbone_feature, action, state):
        state_feat = self.state_encoder(state)
        t = self.sample_time(action.shape[0], device=action.device, dtype=action.dtype)

        noise_action = torch.randn(action.shape, device=action.device, dtype=action.dtype)
        t = rearrange(t, "B -> B 1 1")
        noisy_traj = t*action + (1-t)*noise_action
        gt = action - noise_action
        action_feat = self.action_encoder(noisy_traj)

        state_action_feat = torch.cat((state_feat, action_feat), dim=1)
        
        attention_mask = torch.triu(torch.ones(action.shape[-2], action.shape[-2]), diagonal=1).bool() if self.cfg.DiT.attention_mask else None
        latent_v = self.model(state_action_feat, backbone_feature, attention_mask=attention_mask)

        pred_v = self.action_decoder(latent_v)[:,:self.cfg.Action.chunk_size,:]
        loss = F.mse_loss(pred_v, gt)
        return loss
    
    @torch.no_grad()
    def get_action(self, backbone_feature, state):
        state_feat = self.state_encoder(state)
        state_feat = rearrange(state_feat, "b d -> b 1 d")
        batch_size = state.shape[0]
        actions = torch.randn((batch_size, self.cfg.Action.chunk_size, self.cfg.Action.dim),
                              device=state.device,
                              dtype=state.dtype)
        
        num_steps = self.cfg.num_inference_timesteps
        dt = 1.0 / num_steps

        def step(actions):
            action_feat = self.action_encoder(actions)
            state_action_feat = torch.cat((state_feat, action_feat), dim=1)
            
            attention_mask = torch.triu(torch.ones(state_action_feat.shape[-2], backbone_feature.shape[-2]), diagonal=1).bool() if self.cfg.DiT.attention_mask else None
            latent_action = self.model(state_action_feat, backbone_feature, attention_mask=attention_mask)
            pred_v = self.action_decoder(latent_action)[:,:self.cfg.Action.chunk_size,:]
            return pred_v

        # Run denoising steps.
        for _ in range(num_steps):
            pred_v = step(actions)
            actions = actions + dt*pred_v
        return actions
    
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_path = "./helix/cfg/action_expert_cfg.yaml"
    cfg = OmegaConf.load(config_path)
    action_expert = FlowMatchingActionExpert(cfg)
    
    backbone_feat = torch.rand((5,10,cfg.action_expert.DiT.context_dim))
    state_feat = torch.rand((5, cfg.action_expert.State.dim))
    action = action_expert.get_action(backbone_feat, state_feat)
    print(action.shape)
                



