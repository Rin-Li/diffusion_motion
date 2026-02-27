"""
Scenario and path-generation utilities.

Responsibilities:
  - Build a random test environment (grid + start/goal)
  - Run the diffusion policy to generate a trajectory
"""

import numpy as np
import torch

from utils.dataset_utils import (
    in_collision,
    random_rectangles,
    rectangles_to_grid,
    sample_start_goal,
)


def create_test_scenario(bounds, cell_size, origin, rng, max_rectangles, device="cuda"):
    """
    Sample a random obstacle map and a collision-free (start, goal) pair.

    Returns:
        start_tensor   : torch.Tensor [1, obs_dim]
        goal_tensor    : torch.Tensor [1, obs_dim]
        obstacle_map   : torch.Tensor [1, 1, H, W]
    """
    nx, ny = (int((hi - lo) / cell_size) for lo, hi in bounds)

    rects = random_rectangles(bounds=bounds, rng=rng, max_rectangles=max_rectangles)
    grid = rectangles_to_grid(
        nx=nx, ny=ny, bounds=bounds, cell_size=cell_size, rects=rects
    )

    # Resample until both start and goal are collision-free
    collision = True
    while collision:
        start, goal = sample_start_goal(
            grid=grid, bounds=bounds, cell_size=cell_size, origin=origin, rng=rng
        )
        collision = in_collision(goal, grid, cell_size, origin) or in_collision(
            start, grid, cell_size, origin
        )

    obstacle_map = (
        torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    )
    start_tensor = torch.from_numpy(start).unsqueeze(0).to(device)
    goal_tensor = torch.from_numpy(goal).unsqueeze(0).to(device)

    return start_tensor, goal_tensor, obstacle_map


def generate_path(policy, start, goal, obstacles, initial_action=None):
    """
    Run the diffusion policy to generate a trajectory.

    Args:
        policy        : PlaneDiffusionPolicy
        start         : torch.Tensor [1, obs_dim]
        goal          : torch.Tensor [1, obs_dim]
        obstacles     : torch.Tensor [1, 1, H, W]
        initial_action: torch.Tensor [1, pred_horizon, action_dim]  (Gaussian noise)
                        If None, sampled automatically.

    Returns:
        trajectory    : torch.Tensor [1, pred_horizon, action_dim]
        trajectory_all: list of np.ndarray (one per diffusion step)
    """
    device = start.device

    if initial_action is None:
        pred_horizon = policy.config["horizon"]
        action_dim = policy.config["action_dim"]
        initial_action = torch.randn((1, pred_horizon, action_dim), device=device)

    start_np = start.cpu().numpy()[0]
    goal_np = goal.cpu().numpy()[0]

    obs_dict = {
        "env": np.concatenate([start_np, goal_np]),  # [2 * obs_dim]
        "map": obstacles[0].cpu().numpy(),            # [1, H, W]
    }

    trajectory, trajectory_all = policy.predict_action(obs_dict, initial_action)

    return torch.tensor(trajectory, device=device).unsqueeze(0), trajectory_all
