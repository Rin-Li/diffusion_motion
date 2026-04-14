import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.scenario_utils import create_test_scenario, generate_path
from utils.viz_utils import (
    MNP_INSPIRED_PALETTE,
    draw_circle_obstacles,
    draw_start_goal,
    show_multiple_with_collision_colors,
    style_continuous_axis,
)
import torch
from utils.scenario_utils import create_test_scenario, generate_path, line_blocked
from utils.viz_utils import show_multiple_with_collision_colors
from utils.dataset_utils import validate_path_circle_collision_free
from utils.xcloud_utils import sample_point_cloud_from_grid
from core.datasets.plane_dataset_3ch import _gaussian_blob
from core.diffusion.policy import PlaneDiffusionPolicy
from core.diffusion.builder import build_noise_scheduler_from_config
from core.networks.embedUnet import ConditionalUnet1D
from config.plane_test_embed import PlaneTestEmbedConfig
from config.plane_continuous import PlaneContinuousConfig

# HYPERPARAMETER: choose test mode
#   "original"   – grid-based 8×8 map, ViT/MLP encoder
#   "continuous" – 64×64 3-channel map, CNN encoder
MODE = "continuous"

CKPT_NAME = "ckpt_continuous_final.ckpt"


# Original mode (unchanged)

def test_diffusion_policy(policy, config_dict, num_tests=20, device='cuda',):
    """
    Test diffusion policy using value_utils functions
    """
    # Test parameters
    bounds = np.array([[0.0, 8.0], [0.0, 8.0]])
    cell_size = 1.0
    origin = bounds[:, 0]
    max_rectangles = (3, 5)
    rng = np.random.default_rng(40)

    # Storage for results
    grid_list = []
    path_list = []
    start_list = []
    goal_list = []

    print(f"Generating {num_tests} test scenarios...")

    for i in range(num_tests):
        try:
            # Create test scenario using your value_utils function
            start, goal, obstacles = create_test_scenario(
                bounds=bounds,
                cell_size=cell_size,
                origin=origin,
                rng=rng,
                max_rectangles=max_rectangles,
                device=device
            )

            # Generate path using diffusion policy
            initial_action = torch.randn((1, config_dict["horizon"], config_dict["action_dim"]), device=device)
            trajectory, _ = generate_path(policy, start, goal, obstacles, initial_action)

            # Convert to numpy for visualization
            grid_np = obstacles[0, 0].cpu().numpy()
            path_np = trajectory[0].cpu().numpy()
            start_np = start[0].cpu().numpy()
            goal_np = goal[0].cpu().numpy()

            # Store results
            grid_list.append(grid_np)
            path_list.append(path_np)
            start_list.append(start_np)
            goal_list.append(goal_np)

            print(f"Test {i+1}/{num_tests} completed")

        except Exception as e:
            print(f"Error in test {i+1}: {str(e)}")
            continue

    print(f"Generated {len(path_list)} valid test cases")

    # Visualize results with collision-based coloring
    indices = list(range(len(path_list)))
    results = show_multiple_with_collision_colors(
        grid_list, path_list, start_list, goal_list, indices, cols=5
    )

    return results


# Continuous mode helpers

def create_test_scenario_continuous(bounds, rng, max_obstacles=5, map_resolution=64, device='cuda'):
    """
    Sample a random scenario with circular obstacles for continuous mode.

    Returns:
        start_tensor : torch.Tensor [1, 2]  (world coords)
        goal_tensor  : torch.Tensor [1, 2]  (world coords)
        map3         : np.ndarray   [3, H, W]  (obstacles, start blob, goal blob)
        obstacles    : list of (center, radius) — for continuous collision check
    """
    bounds_arr = np.asarray(bounds, dtype=float)
    xmin, xmax = bounds_arr[0]
    ymin, ymax = bounds_arr[1]
    res = map_resolution
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for _ in range(300):
        start = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        goal  = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        if np.linalg.norm(start - goal) < 2.0:
            continue

        n_obs = rng.integers(1, max_obstacles + 1)
        obstacles = []
        for _ in range(n_obs):
            center = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            radius = rng.uniform(0.3, 1.2)
            obstacles.append((center, radius))

        if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
            continue
        if any(np.linalg.norm(goal - c) <= r for c, r in obstacles):
            continue
        if not line_blocked(start, goal, obstacles):
            continue

        # Build 64×64 occupancy map (circular obstacles)
        occ = np.zeros((res, res), dtype=np.float32)
        for (cx, cy), r in obstacles:
            dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            occ[dist <= r] = 1.0

        # Add gaussian blobs (channels 1 and 2)
        start_blob = _gaussian_blob(start.astype(np.float32), bounds, shape=(res, res))
        goal_blob  = _gaussian_blob(goal.astype(np.float32),  bounds, shape=(res, res))
        map3 = np.stack([occ, start_blob, goal_blob], axis=0)  # (3, H, W)

        start_tensor = torch.from_numpy(start.astype(np.float32)).unsqueeze(0).to(device)
        goal_tensor  = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
        return start_tensor, goal_tensor, map3, obstacles

    raise RuntimeError("Could not generate a valid continuous scenario.")


def show_continuous_results(obstacles_list, path_list, start_list, goal_list, bounds, cols=5):
    """
    Visualize continuous-mode results with circle obstacles (like RRT_star.visualize_path).
    Collision detection uses validate_path_circle_collision_free from utils.
    """
    n = len(path_list)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]

    collision_count = 0
    success_count   = 0

    for i in range(n):
        ax = axes[i]
        obstacles = obstacles_list[i]
        path  = path_list[i]
        start = start_list[i]
        goal  = goal_list[i]

        is_cf = validate_path_circle_collision_free(path, obstacles)
        goal_dist = np.linalg.norm(path[-1] - goal)
        goal_reached = goal_dist < 0.5

        if is_cf and goal_reached:
            color, status = MNP_INSPIRED_PALETTE["success"], "SUCCESS"
            success_count += 1
        elif is_cf:
            color, status = MNP_INSPIRED_PALETTE["no_goal"], "NO GOAL"
        else:
            color, status = MNP_INSPIRED_PALETTE["collision"], "COLLISION"
            collision_count += 1

        style_continuous_axis(ax, (xmin, xmax), (ymin, ymax))
        draw_circle_obstacles(ax, obstacles)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, alpha=0.8)
        ax.scatter(
            path[:, 0],
            path[:, 1],
            color=color,
            s=10,
            alpha=0.55,
            edgecolors=MNP_INSPIRED_PALETTE["path_marker"],
            linewidths=0.2,
        )
        draw_start_goal(ax, start, goal)
        ax.set_title(f"#{i}: {status}\nGoal dist: {goal_dist:.2f}")

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    total = n
    print(f"\n=== Continuous Test Results ===")
    print(f"Total tests : {total}")
    print(f"Success     : {success_count}  ({success_count/total:.2%})")
    print(f"Collision   : {collision_count} ({collision_count/total:.2%})")
    plt.savefig("test_results_continuous.pdf", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "total_tests":     total,
        "success_count":   success_count,
        "collision_count": collision_count,
        "success_rate":    success_count / total,
        "collision_rate":  collision_count / total,
    }


# Continuous mode test function

def test_diffusion_continue(policy, config_dict, num_tests=20, device='cuda'):
    """
    Test the continuous-space diffusion policy.

    - Input map : 64×64 3-channel (obstacles, start blob, goal blob)
    - Trajectories are in continuous world coords [0, 8]
    - Visualization uses the full 64×64 map (high-res)
    """
    bounds = np.array([[0.0, 8.0], [0.0, 8.0]])
    rng = np.random.default_rng(42)

    obstacles_list = []
    path_list      = []
    start_list     = []
    goal_list      = []

    print(f"Generating {num_tests} continuous test scenarios...")

    for i in range(num_tests):
        try:
            start, goal, map3, obstacles = create_test_scenario_continuous(
                bounds=bounds, rng=rng, max_obstacles=5,
                map_resolution=64, device=device,
            )

            initial_action = torch.randn(
                (1, config_dict["horizon"], config_dict["action_dim"]), device=device
            )

            start_np = start.cpu().numpy()[0]
            goal_np  = goal.cpu().numpy()[0]

            obs_dict = {
                "env": np.concatenate([start_np, goal_np]),  # used only for hard clamp
                "map": map3,                                  # (3, 64, 64)
            }
            if config.network_config.get("use_xcloud", False):
                total_points = config.network_config.get("xcloud_encoder", {}).get("total_points", 128)
                obs_dict["xcloud"] = sample_point_cloud_from_grid(map3, total_points)
            trajectory, _ = policy.predict_action(obs_dict, initial_action)

            obstacles_list.append(obstacles)
            path_list.append(trajectory)          # (T, 2) world coords
            start_list.append(start_np)
            goal_list.append(goal_np)

            print(f"Test {i+1}/{num_tests} completed")

        except Exception as e:
            print(f"Error in test {i+1}: {str(e)}")
            continue

    print(f"Generated {len(path_list)} valid test cases")

    results = show_continuous_results(
        obstacles_list, path_list, start_list, goal_list,
        bounds=bounds, cols=5,
    )
    return results


# Main entry point

def main():
    ckpt_path = Path(__file__).parent / "ckpt" / CKPT_NAME
    device    = "cuda"

    if MODE == "original":
        print("=== Mode: original (grid 8×8) ===")
        config     = PlaneTestEmbedConfig()
        config_dict = config.to_dict()

        if config.network_config.get("use_xcloud", False):
            global_cond_dim = (
                config.network_config["xcloud_encoder"]["embed_dim"] +
                config.network_config["mlp_config"]["embed_dim"]
            )
        else:
            global_cond_dim = (
                config.network_config["vit_config"]["num_classes"] +
                config.network_config["mlp_config"]["embed_dim"]
            )

        net = ConditionalUnet1D(
            input_dim       = config.network_config["unet_config"]["action_dim"],
            global_cond_dim = global_cond_dim,
            network_config  = config.network_config,
            is_cnn          = config.is_CNN,
        )

    elif MODE == "continuous":
        print("=== Mode: continuous (64×64 CNN, 3-channel map) ===")
        config      = PlaneContinuousConfig()
        config_dict = config.to_dict()

        cnn_output_dim = config.network_config["cnn_config"]["output_dim"]
        net = ConditionalUnet1D(
            input_dim       = config.action_dim,
            global_cond_dim = cnn_output_dim,
            network_config  = config.network_config,
            is_cnn          = config.is_CNN,
        )

    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Choose 'original' or 'continuous'.")

    scheduler = build_noise_scheduler_from_config(config_dict)
    policy    = PlaneDiffusionPolicy(
        model=net, noise_scheduler=scheduler,
        config=config_dict, device=device,
    )

    print(f"Loading weights from: {ckpt_path}")
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")

    if MODE == "original":
        results = test_diffusion_policy(policy, config_dict, num_tests=100, device=device)
    else:
        results = test_diffusion_continue(policy, config_dict, num_tests=100, device=device)

    print("\n=== Final Results ===")
    print(f"Success Rate  : {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    return results


if __name__ == "__main__":
    results = main()
