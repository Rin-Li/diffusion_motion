import numpy as np
import matplotlib.pyplot as plt
import torch

def show(grid, path, start, goal, figsize=(6, 6)):
    grid = np.array(grid)  
    print("grid.shape:", grid.shape)
    print("start:", start)
    print("goal:", goal)
    
    nx, ny = grid.shape[0], grid.shape[1]
    cell_size = 1  
    bounds = [(0, nx * cell_size), (0, ny * cell_size)]
    origin = [bounds[0][0], bounds[1][0]]  
    _, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title("RRT on occupancy grid (pruning=%s)" % ("on" if path is not None else "off"))

    xs = np.arange(nx) * cell_size + origin[0]
    ys = np.arange(ny) * cell_size + origin[1]
    for ix in range(nx):
        for iy in range(ny):
            if grid[ix, iy]:  # Use numpy indexing
                rect = plt.Rectangle(
                    (xs[ix], ys[iy]),
                    cell_size,
                    cell_size,
                    color="gray",
                    alpha=0.5,
                )
                ax.add_patch(rect)
    
    ax.plot(path[:, 0], path[:, 1], "b-", lw=2, label="path")
    ax.plot(start[0], start[1], "go", ms=8, label="start")
    ax.plot(goal[0], goal[1], "ro", ms=8, label="goal")
    ax.legend()
    plt.grid(True)
    plt.show()


train_data_set = np.load("train_data_set.npy", allow_pickle=True).item()

flat_start = np.array(train_data_set["start"], dtype=float)
flat_goal = np.array(train_data_set["goal"], dtype=float)
flat_map = np.array(train_data_set["map"], dtype=float)  
flat_paths = np.array(train_data_set["paths"], dtype=float)



idx = 3
print(flat_paths[idx])
show(flat_map[idx], path=flat_paths[idx], start=flat_start[idx], goal=flat_goal[idx])
train_data_set_flatten = {
    "start": flat_start,
    "goal": flat_goal,
    "map": flat_map,
    "paths": flat_paths,
}

np.save("train_data_set_flatten.npy", train_data_set_flatten, allow_pickle=True)
