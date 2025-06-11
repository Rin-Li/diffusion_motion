from core.networks.unet import TemporalUnet_WCond
from config.plane_test import PlaneTestConfig
import torch

config = PlaneTestConfig()

net = TemporalUnet_WCond(
    horizon=config.horizon,
    transition_dim=config.transition_dim,
    network_config=config.network_config
)
noised_action = torch.randn(1, config.horizon, config.action_dim)
grid = torch.randn(1, 1, 8, 8)
diffusion_iter = torch.zeros((1,))

noise = net(
    x=noised_action,
    time=diffusion_iter,
    walls_loc=grid
)

print("Noise shape:", noise.shape)