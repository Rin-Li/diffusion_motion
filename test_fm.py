import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from config.plane_fm import PlaneFMConfig
from core.datasets.plane_dataset_3ch import _gaussian_blob
from core.flow_matching import FlowMatching, FlowMatchingPolicy
from core.networks.embedUnet import ConditionalUnet1D
from utils.dataset_utils import validate_path_circle_collision_free
from utils.viz_utils import (
    MNP_INSPIRED_PALETTE,
    draw_circle_obstacles,
    draw_start_goal,
    style_continuous_axis,
)
from utils.scenario_utils import line_blocked

CKPT_NAME  = "fm_final_1.ckpt"
NUM_TESTS  = 100
BOUNDS     = np.array([[0.0, 8.0], [0.0, 8.0]])
MAP_RES    = 64
MAX_OBS    = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def create_test_scenario(rng):
    bounds_arr = BOUNDS
    xmin, xmax = bounds_arr[0]
    ymin, ymax = bounds_arr[1]
    xs = np.linspace(xmin, xmax, MAP_RES)
    ys = np.linspace(ymin, ymax, MAP_RES)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for _ in range(300):
        start = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        goal  = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        if np.linalg.norm(start - goal) < 2.0:
            continue

        obstacles = []
        for _ in range(rng.integers(1, MAX_OBS + 1)):
            center = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            radius = rng.uniform(0.3, 1.2)
            obstacles.append((center, radius))

        if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
            continue
        if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
            continue
        if not line_blocked(start, goal, obstacles):
            continue

        occ = np.zeros((MAP_RES, MAP_RES), dtype=np.float32)
        for (cx, cy), r in obstacles:
            occ[np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= r] = 1.0

        ch1 = _gaussian_blob(start.astype(np.float32), BOUNDS.tolist(), shape=(MAP_RES, MAP_RES))
        ch2 = _gaussian_blob(goal.astype(np.float32),  BOUNDS.tolist(), shape=(MAP_RES, MAP_RES))
        map3 = np.stack([occ, ch1, ch2], axis=0)  # (3, H, W)

        return start.astype(np.float32), goal.astype(np.float32), map3, obstacles

    raise RuntimeError("Could not sample a valid scenario after 300 attempts.")


def show_results(obstacles_list, path_list, start_list, goal_list, cols=5, outfile="test_results_fm.pdf"):
    n    = len(path_list)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    xmin, xmax = BOUNDS[0]
    ymin, ymax = BOUNDS[1]

    success_count   = 0
    collision_count = 0

    for i in range(n):
        ax        = axes[i]
        path      = path_list[i]
        obstacles = obstacles_list[i]
        start     = start_list[i]
        goal      = goal_list[i]

        is_cf        = validate_path_circle_collision_free(path, obstacles)
        goal_dist    = np.linalg.norm(path[-1] - goal)
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
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, alpha=0.85)
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
        ax.set_title(f"#{i}: {status}\ndist={goal_dist:.2f}", fontsize=8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    total = n
    print(f"\n=== Flow Matching Test Results ===")
    print(f"Total   : {total}")
    print(f"Success : {success_count}  ({success_count/total:.2%})")
    print(f"Collision: {collision_count} ({collision_count/total:.2%})")

    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")
    plt.show()

    return {
        "total":          total,
        "success_count":  success_count,
        "collision_count": collision_count,
        "success_rate":   success_count / total,
        "collision_rate": collision_count / total,
    }


def main():
    ckpt_path = Path(__file__).parent / "ckpt" / CKPT_NAME

    config      = PlaneFMConfig()
    config_dict = config.to_dict()

    if config.network_config.get("use_xcloud", False):
        global_cond_dim = config.network_config["xcloud_encoder"]["embed_dim"]
    else:
        global_cond_dim = config.network_config["cnn_config"]["output_dim"]

    net = ConditionalUnet1D(
        input_dim       = config.action_dim,
        global_cond_dim = global_cond_dim,
        network_config  = config.network_config,
        is_cnn          = config.is_CNN,
    )

    fm     = FlowMatching(num_infer_steps=config.flow_matching["num_infer_steps"])
    policy = FlowMatchingPolicy(model=net, flow_matching=fm, config=config_dict, device=DEVICE)

    print(f"Loading weights from: {ckpt_path}")
    policy.load_weights(ckpt_path)
    print(f"Total parameters: {sum(p.numel() for p in net.parameters()):,}")

    rng = np.random.default_rng(42)
    obstacles_list, path_list, start_list, goal_list = [], [], [], []

    print(f"Running {NUM_TESTS} test scenarios...")
    for i in range(NUM_TESTS):
        try:
            start, goal, map3, obstacles = create_test_scenario(rng)

            obs_dict = {
                "env": np.concatenate([start, goal]),
                "map": map3,
            }
            x0   = torch.randn(1, config.horizon, config.action_dim, device=DEVICE)
            path = policy.predict_action(obs_dict, x0)  # (T, 2)

            obstacles_list.append(obstacles)
            path_list.append(path)
            start_list.append(start)
            goal_list.append(goal)

            if (i + 1) % 10 == 0 or (i + 1) == NUM_TESTS:
                print(f"  {i+1}/{NUM_TESTS}")
        except Exception as e:
            print(f"  test {i+1} failed: {e}")

    results = show_results(obstacles_list, path_list, start_list, goal_list, cols=5)

    print(f"\nSuccess rate : {results['success_rate']:.2%}")
    print(f"Collision rate: {results['collision_rate']:.2%}")
    return results


if __name__ == "__main__":
    main()
