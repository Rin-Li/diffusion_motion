import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load("rrt_star_grid_dataset.npz", allow_pickle=True)
training_data = data["arr_0"].item()  # 因为保存的是一个 dict

starts = np.array(training_data["start"])
goals = np.array(training_data["goal"])
obstacles_list = training_data["obstacles"]  # 每个障碍是整个 occupancy grid

# 选择一个例子可视化
idx = 0  # 改变这个可以查看不同的样本
start = starts[idx]
goal = goals[idx]
obstacles = obstacles_list[idx]  # 是个 2D bool array


def visualize_training_sample(start, goal, grid, bounds, cell_size):
    nx, ny = grid.shape
    origin = np.array([b[0] for b in bounds])
    
    xs = np.arange(nx) * cell_size + origin[0]
    ys = np.arange(ny) * cell_size + origin[1]
    
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title(f"Training Sample (start={start}, goal={goal})")

    # 画障碍
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

    # 起点终点
    ax.plot(start[0], start[1], "go", ms=8, label="start")
    ax.plot(goal[0], goal[1], "ro", ms=8, label="goal")
    ax.legend()
    plt.grid(True)
    plt.show()

# 假设你原始用的是这个 bounds 和 cell_size
bounds = [(0.0, 10.0), (0.0, 10.0)]
cell_size = 0.1

visualize_training_sample(start, goal, obstacles, bounds, cell_size)
