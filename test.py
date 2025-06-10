import numpy as np
import matplotlib.pyplot as plt

class RRTVisualizer:
    def visualize_path(self, bounds, obstacles, path=None, raw_path=None, start=None, goal=None):
        _, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        print(obstacles)

        for obstacle in obstacles:
            circle = plt.Circle(obstacle[ : 2], obstacle[2], color='gray', alpha=0.5)
            ax.add_patch(circle)

        if raw_path is not None:
            raw_path = np.array(raw_path)
            ax.plot(raw_path[:, 0], raw_path[:, 1], 'r--', linewidth=1, label='Raw path')

        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Optimized path')
            ax.plot(path[:, 0], path[:, 1], 'bo', markersize=3)

        if start is not None:
            ax.plot(start[0], start[1], 'go', label='Start', markersize=8)
        if goal is not None:
            ax.plot(goal[0], goal[1], 'ro', label='Goal', markersize=8)

        ax.legend()
        ax.set_title("RRT* Path Planning Visualization")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    data = np.load("rrt_2d_dataset_500.npz", allow_pickle=True)

    starts = data['starts']
    goals = data['goals']
    paths = data['paths']
    obstacles_all = data['obstacles']

    idx = 100  #
    start = starts[idx]
    goal = goals[idx]
    path = paths[idx]
    obstacles = obstacles_all[idx]

    bounds = [(0, 8), (0, 8)]  

    vis = RRTVisualizer()
    vis.visualize_path(bounds, obstacles, path=path, start=start, goal=goal)
