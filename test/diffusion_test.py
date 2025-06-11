from core.networks.unet import TemporalUnet_WCond
from core.networks.diffusion import GaussianDiffusionPB
from config.plane_test import PlaneTestConfig
import torch

horizon = 32
observation_dim = 2
action_dim = 0
batch_size = 2


config = PlaneTestConfig()
net = TemporalUnet_WCond(
    horizon=config.horizon,
    transition_dim=config.transition_dim,
    network_config=config.network_config
)
diffusion = GaussianDiffusionPB(
    model=net,
    horizon=config.horizon,
    action_dim=config.action_dim,
    observation_dim=observation_dim,
    diff_config=config.diff_config
)
cond = {
    0: torch.tensor([[0.0, 0.0], [1.0, 1.0]]),  
    -1: torch.tensor([[0.9, 0.9], [-0.9, -0.9]])  
}
walls_loc = torch.rand(batch_size, 1, 8, 8)
diffusion.eval()
diffusion.model.eval() 
with torch.no_grad():
    traj = diffusion.conditional_sample(cond=cond, walls_loc=walls_loc, use_ddim=True)
    print("Sampled Trajectory:", traj.shape)  