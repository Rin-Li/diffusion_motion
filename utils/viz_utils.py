"""
Visualization utilities for diffusion policy evaluation.

Responsibilities:
  - Plot test results with collision-based coloring
  - Visualize a single trajectory
  - Export a diffusion denoising process as a GIF
"""

import io
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.dataset_utils import validate_path_collision_free


def show_multiple_with_collision_colors(
    grid_list, path_list, start_list, goal_list, indices, cols=5
):
    """
    Plot multiple test results.  Path color encodes outcome:
      - blue   : collision-free AND goal reached  (SUCCESS)
      - orange : collision-free but goal NOT reached
      - red    : collision detected

    Returns a dict with aggregate statistics.
    """
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    collision_count = 0
    success_count = 0

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        ax = axes[i]
        grid = np.array(grid_list[idx])
        path = np.array(path_list[idx])
        start = np.array(start_list[idx])
        goal = np.array(goal_list[idx])

        nx, ny = grid.shape[0], grid.shape[1]
        cell_size = 1.0
        bounds = [(0, nx * cell_size), (0, ny * cell_size)]
        origin = [bounds[0][0], bounds[1][0]]

        is_collision_free = validate_path_collision_free(path, grid, cell_size, origin)
        goal_distance = np.linalg.norm(path[-1] - goal)
        goal_reached = goal_distance < 0.5

        if is_collision_free and goal_reached:
            path_color, status = "blue", "SUCCESS"
            success_count += 1
        elif is_collision_free:
            path_color, status = "orange", "NO GOAL"
        else:
            path_color, status = "red", "COLLISION"
            collision_count += 1

        ax.set_aspect("equal")
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title(f"#{idx}: {status}\nGoal dist: {goal_distance:.2f}")

        xs = np.arange(nx) * cell_size + origin[0]
        ys = np.arange(ny) * cell_size + origin[1]
        for ix in range(nx):
            for iy in range(ny):
                if grid[ix, iy]:
                    ax.add_patch(
                        plt.Rectangle(
                            (xs[ix], ys[iy]), cell_size, cell_size,
                            color="gray", alpha=0.5,
                        )
                    )

        ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=2, alpha=0.8)
        ax.scatter(path[:, 0], path[:, 1], color=path_color, s=15, alpha=0.6)
        ax.plot(start[0], start[1], "go", ms=8, label="Start")
        ax.plot(goal[0], goal[1], "ro", ms=8, label="Goal")
        ax.grid(True, alpha=0.3)

    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    total = len(indices)
    print(f"\n=== Test Results ===")
    print(f"Total tests   : {total}")
    print(f"Success       : {success_count}  ({success_count/total:.2%})")
    print(f"Collision     : {collision_count} ({collision_count/total:.2%})")
    plt.savefig("test_results.pdf", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "total_tests": total,
        "success_count": success_count,
        "collision_count": collision_count,
        "success_rate": success_count / total,
        "collision_rate": collision_count / total,
    }


def visualize_result(trajectory, start, goal, obstacles, save_path=None):
    """Visualize a single generated trajectory over the obstacle map."""
    traj_np = trajectory[0].cpu().numpy()
    start_np = start[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    obs_np = obstacles[0, 0].cpu().numpy()  # [H, W]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(obs_np, origin="lower", cmap="gray_r", extent=[0, 8, 0, 8], alpha=0.5)
    ax1.plot(start_np[0], start_np[1], "go", markersize=10, label="Start")
    ax1.plot(goal_np[0], goal_np[1], "ro", markersize=10, label="Goal")
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 8)
    ax1.set_title("Generated Trajectory")
    ax1.set_xticks(np.arange(0, 9))
    ax1.set_yticks(np.arange(0, 9))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position")
    ax2.set_title("Trajectory Components")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def visualize_trajectory_gif(
    action_history: Union[np.ndarray, torch.Tensor],
    start: Union[np.ndarray, torch.Tensor],
    goal: Union[np.ndarray, torch.Tensor],
    obstacles: Union[np.ndarray, torch.Tensor],
    save_path: str = "trajectory_evolution.gif",
    fps: int = 5,
):
    """Export the full diffusion denoising process as an animated GIF."""

    def _to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    action_history = _to_numpy(action_history)
    start = _to_numpy(start).squeeze()
    goal = _to_numpy(goal).squeeze()
    obstacles = _to_numpy(obstacles)
    if obstacles.ndim == 4:        # [B, C, H, W]
        obstacles = obstacles[0, 0]
    elif obstacles.ndim == 3:      # [C, H, W]
        obstacles = obstacles[0]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for step_idx, traj in enumerate(action_history):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(
            obstacles, cmap="gray_r", origin="lower",
            extent=[0, obstacles.shape[1], 0, obstacles.shape[0]], alpha=0.25,
        )
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2, label="Path")
        ax.scatter(traj[:, 0], traj[:, 1], c="blue", s=12, alpha=0.7)
        ax.scatter(start[0], start[1], c="green", s=80, marker="o", label="Start")
        ax.scatter(goal[0], goal[1], c="red", s=80, marker="o", label="Goal")
        ax.set_xlim(0, obstacles.shape[1])
        ax.set_ylim(0, obstacles.shape[0])
        ax.set_title(f"Diffusion step {step_idx}/{len(action_history) - 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    iio.imwrite(save_path, frames, duration=1 / fps)
    print(f"GIF saved to: {Path(save_path).resolve()}")
