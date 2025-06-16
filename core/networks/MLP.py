import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, obs_dim: int, embed_dim: int = 64):
        """
        obs_dim: (B, obs_dim * 2)
        embed_dim: (B, embed_dim)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, env: torch.Tensor):
        """
        env: [B, 2 * obs_dim]
        return: [B, embed_dim]
        """
        return self.net(env)
