import torch


class FlowMatching:
    # x0: noise, x1: target trajectory, t in [0, 1]

    def __init__(self, num_infer_steps: int = 10):
        self.num_infer_steps = num_infer_steps

    def sample_timesteps(self, batch_size: int, device) -> torch.Tensor:
        # (B,)
        return torch.rand(batch_size, device=device)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x0, x1: (B, T, D)   t: (B, 1, 1)
        # x_t = (1 - t) * x0 + t * x1
        return (1.0 - t) * x0 + t * x1

    def get_velocity_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # d/dt [(1-t)*x0 + t*x1] = x1 - x0
        return x1 - x0

    def get_inference_timesteps(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, self.num_infer_steps + 1)

    def euler_step(self, xt: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        # x_{t+dt} = x_t + v * dt
        return xt + v * dt
