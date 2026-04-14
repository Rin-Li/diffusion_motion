import sys
sys.path.insert(0, '.')

import torch
from config.plane_continuous import PlaneContinuousConfig
from core.networks.embedUnet import ConditionalUnet1D

config = PlaneContinuousConfig()

cnn_output_dim = config.network_config['cnn_config']['output_dim']
mlp_embed_dim  = config.network_config['mlp_config']['embed_dim']

net = ConditionalUnet1D(
    input_dim       = config.action_dim,
    global_cond_dim = cnn_output_dim + mlp_embed_dim,
    network_config  = config.network_config,
    is_cnn          = config.is_CNN,
)

total_params = sum(p.numel() for p in net.parameters())
print(f'Total parameters: {total_params:,}')

dummy_action = torch.randn(2, config.horizon, config.action_dim)
dummy_map    = torch.randn(2, 1, 64, 64)
dummy_env    = torch.randn(2, 4)
dummy_t      = torch.zeros(2).long()

noise = net(
    sample    = dummy_action,
    timestep  = dummy_t,
    map_cond  = dummy_map,
    env_cond  = dummy_env,
)
print(f'Noise shape: {noise.shape}')   # expected: torch.Size([2, 48, 2])
