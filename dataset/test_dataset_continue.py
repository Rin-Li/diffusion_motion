import numpy as np
import matplotlib.pyplot as plt


def show(obstacles, path, start, goal, bounds=[(0, 8), (0, 8)], figsize=(6, 6)):
    """Visualize a single sample with circular obstacles"""
    _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    
    # Draw circular obstacles
    for obs in obstacles:
        center = obs[:2]  # [x, y]
        radius = obs[2]   # r
        circle = plt.Circle(center, radius, color='gray', alpha=0.5)
        ax.add_patch(circle)
    
    # Draw path
    if path is not None:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
        ax.plot(path[:, 0], path[:, 1], 'bo', markersize=3)
    
    # Draw start and goal
    if start is not None:
        ax.plot(start[0], start[1], 'go', label='Start', markersize=10)
    if goal is not None:
        ax.plot(goal[0], goal[1], 'ro', label='Goal', markersize=10)
    
    ax.legend()
    ax.set_title("RRT* Path Planning")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_multiple(obstacles_list, path_list, start_list, goal_list, indices, 
                  bounds=[(0, 8), (0, 8)], cols=5, cell_figsize=4):
    """Visualize multiple samples in a grid layout with larger subplots"""
    n_samples = len(indices)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_figsize, rows * cell_figsize))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        # Draw circular obstacles
        obstacles = obstacles_list[idx]
        for obs in obstacles:
            center = obs[:2]
            radius = obs[2]
            circle = plt.Circle(center, radius, color='gray', alpha=0.5)
            ax.add_patch(circle)
        # Draw path
        path = np.array(path_list[idx])
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=1)
        # Draw start and goal
        start = start_list[idx]
        goal = goal_list[idx]
        ax.plot(start[0], start[1], 'go', markersize=5)
        ax.plot(goal[0], goal[1], 'ro', markersize=5)
        ax.set_title(f"Sample {idx}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: dataset_visualization.png")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    data_file = "rrt_2d_dataset_500.npz"
    print(f"Loading dataset from: {data_file}")
    
    data = np.load(data_file, allow_pickle=True)
    
    starts = data['starts']      # (num_samples,) each element is (2,)
    goals = data['goals']        # (num_samples,) each element is (2,)
    paths = data['paths']        # (num_samples,) each element is (50, 2)
    obstacles = data['obstacles'] # (num_samples,) each element is (n_obs, 3)
    
    print(f"\n=== Dataset Information ===")
    print(f"Number of samples: {len(paths)}")
    print(f"Path lengths: {[len(p) for p in paths]}")
    print(f"Obstacle counts per sample: {[len(obs) for obs in obstacles]}")
    
    # Display dataset statistics
    path_lengths = [len(p) for p in paths]
    print(f"\nPath length range: {min(path_lengths)} - {max(path_lengths)}")
    print(f"Average path length: {np.mean(path_lengths):.1f}")
    
    # Display details of first sample
    print(f"\n=== Sample 0 Details ===")
    print(f"Start position: {starts[0]}")
    print(f"Goal position: {goals[0]}")
    print(f"Path shape: {paths[0].shape}")
    print(f"Number of obstacles: {len(obstacles[0])}")
    print(f"Obstacle details (center_x, center_y, radius):\n{obstacles[0]}")
    
    # Visualize single sample
    print("\n=== Visualizing Sample 0 ===")
    show(obstacles[0], paths[0], starts[0], goals[0])
    
    # Visualize multiple samples
    print("\n=== Visualizing Multiple Samples ===")
    num_to_show = len(paths)
    indices = list(range(num_to_show))
    show_multiple(obstacles, paths, starts, goals, indices, cols=5)
    
    print("\nVisualization complete!")