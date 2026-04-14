import io
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.dataset_utils import validate_path_collision_free


MNP_INSPIRED_PALETTE = {
    "background":    "#F6F4FB",
    "obstacle_fill": "#8E7CC3",
    "obstacle_edge": "#6F5AA8",
    "start":         "#2F6BFF",
    "goal":          "#E24A4A",
    "success":       "#4C6EF5",
    "no_goal":       "#F59F00",
    "collision":     "#E03131",
    "path_marker":   "#FFFFFF",
    "grid":          "#D8D2EA",
}


def style_continuous_axis(ax, xlim, ylim, *, show_grid=True):
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_facecolor(MNP_INSPIRED_PALETTE["background"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_grid:
        ax.grid(True, color=MNP_INSPIRED_PALETTE["grid"], alpha=0.45, linewidth=0.7)


def draw_circle_obstacles(ax, obstacles, *, alpha=0.42):
    for center, radius in obstacles:
        ax.add_patch(plt.Circle(
            center, radius,
            facecolor=MNP_INSPIRED_PALETTE["obstacle_fill"],
            edgecolor=MNP_INSPIRED_PALETTE["obstacle_edge"],
            linewidth=1.2, alpha=alpha,
        ))


def draw_start_goal(ax, start, goal, *, size=42):
    ax.scatter(start[0], start[1], c=MNP_INSPIRED_PALETTE["start"],
               s=size, zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter(goal[0],  goal[1],  c=MNP_INSPIRED_PALETTE["goal"],
               s=size, zorder=5, edgecolors="white", linewidths=0.8)


def show_multiple_with_collision_colors(
    grid_list, path_list, start_list, goal_list, indices, cols=5
):
    # path color: success=blue, no_goal=orange, collision=red
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    collision_count = 0
    success_count   = 0

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        ax    = axes[i]
        grid  = np.array(grid_list[idx])
        path  = np.array(path_list[idx])
        start = np.array(start_list[idx])
        goal  = np.array(goal_list[idx])

        nx, ny    = grid.shape[0], grid.shape[1]
        cell_size = 1.0
        bounds    = [(0, nx * cell_size), (0, ny * cell_size)]
        origin    = [bounds[0][0], bounds[1][0]]

        is_cf     = validate_path_collision_free(path, grid, cell_size, origin)
        goal_dist = np.linalg.norm(path[-1] - goal)

        if is_cf and goal_dist < 0.5:
            path_color, status = MNP_INSPIRED_PALETTE["success"], "SUCCESS"
            success_count += 1
        elif is_cf:
            path_color, status = MNP_INSPIRED_PALETTE["no_goal"], "NO GOAL"
        else:
            path_color, status = MNP_INSPIRED_PALETTE["collision"], "COLLISION"
            collision_count += 1

        style_continuous_axis(ax, bounds[0], bounds[1])
        ax.set_title(f"#{idx}: {status}\nGoal dist: {goal_dist:.2f}")

        xs = np.arange(nx) * cell_size + origin[0]
        ys = np.arange(ny) * cell_size + origin[1]
        for ix in range(nx):
            for iy in range(ny):
                if grid[ix, iy]:
                    ax.add_patch(plt.Rectangle(
                        (xs[ix], ys[iy]), cell_size, cell_size,
                        facecolor=MNP_INSPIRED_PALETTE["obstacle_fill"],
                        edgecolor=MNP_INSPIRED_PALETTE["obstacle_edge"],
                        linewidth=0.6, alpha=0.42,
                    ))

        ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=2, alpha=0.8)
        ax.scatter(path[:, 0], path[:, 1], color=path_color, s=15, alpha=0.55,
                   edgecolors=MNP_INSPIRED_PALETTE["path_marker"], linewidths=0.2)
        draw_start_goal(ax, start, goal)

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
        "total_tests":     total,
        "success_count":   success_count,
        "collision_count": collision_count,
        "success_rate":    success_count / total,
        "collision_rate":  collision_count / total,
    }


def visualize_result(trajectory, start, goal, obstacles, save_path=None):
    traj_np  = trajectory[0].cpu().numpy()
    start_np = start[0].cpu().numpy()
    goal_np  = goal[0].cpu().numpy()
    obs_np   = obstacles[0, 0].cpu().numpy()  # (H, W)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(MNP_INSPIRED_PALETTE["background"])
    style_continuous_axis(ax, (0, 8), (0, 8))
    ax.imshow(obs_np, origin="lower", cmap="Purples",
              extent=[0, 8, 0, 8], alpha=0.3, vmin=0.0, vmax=1.0)
    ax.plot(traj_np[:, 0], traj_np[:, 1],
            color=MNP_INSPIRED_PALETTE["success"], linewidth=2)
    draw_start_goal(ax, start_np, goal_np, size=55)
    ax.set_title("Generated Trajectory")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_trajectory_gif(
    action_history: Union[np.ndarray, torch.Tensor],
    start: Union[np.ndarray, torch.Tensor],
    goal: Union[np.ndarray, torch.Tensor],
    obstacles: Union[np.ndarray, torch.Tensor],
    save_path: str = "trajectory_evolution.gif",
    fps: int = 5,
):
    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    action_history = _np(action_history)
    start          = _np(start).squeeze()
    goal           = _np(goal).squeeze()
    obstacles      = _np(obstacles)
    if obstacles.ndim == 4:    # (B, C, H, W)
        obstacles = obstacles[0, 0]
    elif obstacles.ndim == 3:  # (C, H, W)
        obstacles = obstacles[0]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    H, W = obstacles.shape

    frames = []
    for step_idx, traj in enumerate(action_history):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(MNP_INSPIRED_PALETTE["background"])
        style_continuous_axis(ax, (0, W), (0, H))
        ax.imshow(obstacles, cmap="Purples", origin="lower",
                  extent=[0, W, 0, H], alpha=0.25, vmin=0.0, vmax=1.0)
        ax.plot(traj[:, 0], traj[:, 1],
                color=MNP_INSPIRED_PALETTE["success"], linewidth=2)
        ax.scatter(traj[:, 0], traj[:, 1],
                   c=MNP_INSPIRED_PALETTE["success"], s=12, alpha=0.7)
        draw_start_goal(ax, start, goal, size=80)
        ax.set_title(f"step {step_idx}/{len(action_history) - 1}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(iio.imread(buf))

    iio.imwrite(save_path, frames, duration=1 / fps)
    print(f"GIF saved to: {Path(save_path).resolve()}")
