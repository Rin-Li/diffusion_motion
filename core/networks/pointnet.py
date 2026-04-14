import torch
import torch.nn as nn


class PointCloudEncoder(nn.Module):
    def __init__(self, point_dim: int = 2, hidden_dims=None, embed_dim: int = 64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        dims = [point_dim] + list(hidden_dims) + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Mish())
        self.mlp = nn.Sequential(*layers)

    def forward(self, xcloud: torch.Tensor) -> torch.Tensor:
        """
        xcloud: [B, N, point_dim]
        return: [B, embed_dim]
        """
        if xcloud is None:
            raise ValueError("xcloud is required when use_xcloud is enabled.")
        x = self.mlp(xcloud)  # (B, N, embed_dim)
        x = torch.max(x, dim=1).values  # (B, embed_dim)
        return x
