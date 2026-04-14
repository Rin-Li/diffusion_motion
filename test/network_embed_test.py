from core.networks.embedUnet import ConditionalUnet1D
from config.plane_test_embed import PlaneTestEmbedConfig
import torch
import numpy as np
from data_generator.data_generator_grid import DataGeneratorGrid
from utils.xcloud_utils import sample_point_cloud_from_grid

config = PlaneTestEmbedConfig()
net = ConditionalUnet1D(
    input_dim=config.network_config["unet_config"]["action_dim"],
    global_cond_dim=config.network_config["xcloud_encoder"]["embed_dim"] + config.network_config["mlp_config"]["embed_dim"],
    network_config=config.network_config
)
noised_action = torch.randn(1, config.network_config["unet_config"]["action_horizon"], config.network_config["unet_config"]["action_dim"])
grid = torch.randn(1, 1, 8, 8)
diffusion_iter = torch.zeros((1,))
env_cond = torch.randn(1, 4)  # Assuming obs_dim is 4
xcloud = torch.randn(1, config.network_config["xcloud_encoder"]["total_points"], 2)
noise = net(
    sample=noised_action,
    timestep=diffusion_iter,
    map_cond=grid,
    env_cond=env_cond,
    xcloud=xcloud,
)
print("Noise shape:", noise.shape)


def test_pointcloud_from_grid_generator():
    bounds = [(0, 8), (0, 8)]
    gen = DataGeneratorGrid(bounds=bounds, num_samples=1, rng=123)
    dataset = gen.generate(smooth=False, interp=20)
    grid = dataset["map"][0].astype(np.float32)  # (H, W)
    total_points = config.network_config["xcloud_encoder"]["total_points"]
    xcloud_np = sample_point_cloud_from_grid(grid, total_points)

    assert xcloud_np.shape == (total_points, 2), "xcloud shape mismatch"
    assert np.isfinite(xcloud_np).all(), "xcloud contains non-finite values"


def smoke_test_forward_with_xcloud():
    bounds = [(0, 8), (0, 8)]
    gen = DataGeneratorGrid(bounds=bounds, num_samples=1, rng=123)
    dataset = gen.generate(smooth=False, interp=20)
    grid = dataset["map"][0].astype(np.float32)
    map_cond = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    start = torch.from_numpy(dataset["start"][0].astype(np.float32)).unsqueeze(0)
    goal = torch.from_numpy(dataset["goal"][0].astype(np.float32)).unsqueeze(0)
    env_cond = torch.cat([start, goal], dim=-1)  # (1, 4)

    total_points = config.network_config["xcloud_encoder"]["total_points"]
    xcloud_np = sample_point_cloud_from_grid(grid, total_points)
    xcloud = torch.from_numpy(xcloud_np).unsqueeze(0)  # (1, N, 2)

    noised_action = torch.randn(
        1,
        config.network_config["unet_config"]["action_horizon"],
        config.network_config["unet_config"]["action_dim"],
    )
    diffusion_iter = torch.zeros((1,))

    with torch.no_grad():
        out = net(
            sample=noised_action,
            timestep=diffusion_iter,
            map_cond=map_cond,
            env_cond=env_cond,
            xcloud=xcloud,
        )
    assert out.shape == noised_action.shape, "Output shape mismatch"


if __name__ == "__main__":
    test_pointcloud_from_grid_generator()
    smoke_test_forward_with_xcloud()
