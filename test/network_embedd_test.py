from core.networks.embeddUnet import ConditionalUnet1D
from config.plane_test_embeed import PlaneTestEmbedConfig
import torch

config = PlaneTestEmbedConfig()
net = ConditionalUnet1D(
    input_dim=config.network_config["unet_config"]["action_dim"],
    global_cond_dim=config.network_config["vit_config"]["num_classes"] + config.network_config["mlp_config"]["embed_dim"],
    network_config=config.network_config
)
noised_action = torch.randn(1, config.network_config["unet_config"]["action_horizon"], config.network_config["unet_config"]["action_dim"])
grid = torch.randn(1, 1, 8, 8)
diffusion_iter = torch.zeros((1,))
env_cond = torch.randn(1, 4)  # Assuming obs_dim is 4
noise = net(
    sample=noised_action,
    timestep=diffusion_iter,
    map_cond=grid,
    env_cond=env_cond
)
print("Noise shape:", noise.shape)
