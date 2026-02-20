from utils.dataset_utils import (sample_start_goal, 
                                 random_rectangles, 
                                 rectangles_to_grid, 
                                 validate_path_collision_free, 
                                 in_collision)
import numpy as np
import torch
from core.diffusion.diffusion import PlaneDiffusionPolicy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import imageio.v3 as iio   



def create_test_scenario(bounds, cell_size, origin, rng, max_rectangles, device='cuda'):
    """Create a test scenario with a 3x2 rectangular obstacle in the center of an 8x8 grid"""
    nx, ny = (
    int((hi - lo) / cell_size) for lo, hi in bounds)

    rects = random_rectangles(
        bounds=bounds,
        rng=rng,
        max_rectangles=max_rectangles,
    )
    
    grid = rectangles_to_grid(
        nx=nx,
        ny=ny,
        bounds=bounds,
        cell_size=cell_size,
        rects=rects,
    )
    
    collision = True
    while collision:
        start, goal = sample_start_goal(
            grid=grid,
            bounds=bounds,
            cell_size=cell_size,
            origin=origin,
            rng=rng,
        )
        
        collision = in_collision(goal, grid, cell_size, origin) or in_collision(start, grid, cell_size, origin)
    
    obstacle_map = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    start_tensor = torch.from_numpy(start).unsqueeze(0).to(device)
    goal_tensor = torch.from_numpy(goal).unsqueeze(0).to(device)
    
    

    return start_tensor, goal_tensor, obstacle_map

def generate_path(policy: PlaneDiffusionPolicy, start, goal, obstacles):
    device = start.device
    obs_dim = start.shape[1]
    pred_horizon = policy.config["horizon"]

    # Fake observation sample (all zeros)
    fake_obs_sample = np.zeros((pred_horizon, obs_dim), dtype=np.float32)


    start_np = start.cpu().numpy()[0]
    goal_np = goal.cpu().numpy()[0]
    env_cond = np.concatenate([start_np, goal_np])  # [2 * obs_dim]


    map_cond = obstacles[0].cpu().numpy()  # shape: [1, 8, 8]


    obs_dict = {
        "sample": fake_obs_sample,
        "env": env_cond,
        "map": map_cond,
    }


    trajectory, trajectory_all = policy.predict_action(obs_dict)  # shape: [pred_horizon, obs_dim]

    return torch.tensor(trajectory, device=device).unsqueeze(0), trajectory_all  # [1, H, obs_dim]



def show_multiple_with_collision_colors(grid_list, path_list, start_list, goal_list, indices, cols=5):
    """
    Modified show_multiple function that colors paths based on collision status:
    - Red: collision detected
    - Blue: collision-free
    """
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Fix the axes handling
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
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
        
        # Check collision status using your validation function
        is_collision_free = validate_path_collision_free(path, grid, cell_size, origin)
        
        # Check if goal is reached (within threshold)
        goal_distance = np.linalg.norm(path[-1] - goal)
        goal_reached = goal_distance < 0.5
        
        # Determine path color and status
        if is_collision_free and goal_reached:
            path_color = "blue"
            status = "SUCCESS"
            success_count += 1
        elif is_collision_free:
            path_color = "orange"
            status = "NO GOAL"
        else:
            path_color = "red"
            status = "COLLISION"
            collision_count += 1
        
        ax.set_aspect("equal")
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title(f"#{idx}: {status}\nGoal dist: {goal_distance:.2f}")
        
        # Draw obstacles
        xs = np.arange(nx) * cell_size + origin[0]
        ys = np.arange(ny) * cell_size + origin[1]
        for ix in range(nx):
            for iy in range(ny):
                if grid[ix, iy]:
                    rect = plt.Rectangle(
                        (xs[ix], ys[iy]),
                        cell_size,
                        cell_size,
                        color="gray",
                        alpha=0.5,
                    )
                    ax.add_patch(rect)
        
        # Draw path with collision-based color
        ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=2, alpha=0.8)
        ax.scatter(path[:, 0], path[:, 1], color=path_color, s=15, alpha=0.6)
        
        # Draw start and goal
        ax.plot(start[0], start[1], "go", ms=8, label="Start")
        ax.plot(goal[0], goal[1], "ro", ms=8, label="Goal")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Print statistics
    total_tests = len(indices)
    print(f"\n=== Test Results ===")
    print(f"Total tests: {total_tests}")
    print(f"Successful paths: {success_count}")
    print(f"Collision paths: {collision_count}")
    print(f"Success rate: {success_count/total_tests:.2%}")
    print(f"Collision rate: {collision_count/total_tests:.2%}")
    plt.savefig("test_results.pdf", dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'total_tests': total_tests,
        'success_count': success_count,
        'collision_count': collision_count,
        'success_rate': success_count/total_tests,
        'collision_rate': collision_count/total_tests
    }
    
def visualize_result(trajectory, start, goal, obstacles, save_path=None):
    """Visualize the generated trajectory with obstacle map overlay"""
    # Convert to numpy
    traj_np = trajectory[0].cpu().numpy()
    start_np = start[0].cpu().numpy()
    goal_np = goal[0].cpu().numpy()
    obs_np = obstacles[0, 0].cpu().numpy()  # shape: [8, 8]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Trajectory with obstacles
    ax1.imshow(obs_np, origin='lower', cmap='gray_r', extent=[0, 8, 0, 8], alpha=0.5)  # transpose to match axis

    # ax1.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2, label='Generated Path')
    # ax1.plot(traj_np[:, 0], traj_np[:, 1], 'bo', markersize=3, alpha=0.6)
    ax1.plot(start_np[0], start_np[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_np[0], goal_np[1], 'ro', markersize=10, label='Goal')
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 8)
    ax1.set_title('Generated Trajectory')
    ax1.set_xticks(np.arange(0, 9))
    ax1.set_yticks(np.arange(0, 9))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Trajectory over time
    # ax2.plot(range(len(traj_np)), traj_np[:, 0], 'r-', label='X coordinate')
    # ax2.plot(range(len(traj_np)), traj_np[:, 1], 'b-', label='Y coordinate')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Position')
    ax2.set_title('Trajectory Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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

    action_history = (
        action_history.detach().cpu().numpy()
        if isinstance(action_history, torch.Tensor)
        else np.asarray(action_history)
    )
    start = start.detach().cpu().numpy() if isinstance(start, torch.Tensor) else np.asarray(start)
    goal = goal.detach().cpu().numpy() if isinstance(goal, torch.Tensor) else np.asarray(goal)
    obstacles = (
        obstacles.detach().cpu().numpy() if isinstance(obstacles, torch.Tensor) else np.asarray(obstacles)
    )

    start = start.squeeze()
    goal = goal.squeeze()
    if obstacles.ndim == 4:                      # [B, C, H, W]
        obstacles = obstacles[0, 0]
    elif obstacles.ndim == 3:                    # [C, H, W]
        obstacles = obstacles[0]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for step_idx, traj in enumerate(action_history):
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(
            obstacles,
            cmap="gray_r",
            origin="lower",
            extent=[0, obstacles.shape[1], 0, obstacles.shape[0]],
            alpha=0.25,
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
    print(f"GIF Save to: {Path(save_path).resolve()}")