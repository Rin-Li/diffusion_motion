import numpy as np
import matplotlib.pyplot as plt

def show(grid, path=None, raw_path=None, start=None, goal=None, figsize=(6, 6)):
    grid = np.array(grid)  # Convert to numpy array
    print(f"Grid shape: {grid.shape}")  # Debug info
    print(f"Grid dtype: {grid.dtype}")  # Debug info
    
    nx, ny = grid.shape[0], grid.shape[1]
    cell_size = 0.1  # Assuming each cell is 1x1 for simplicity
    bounds = [(0, nx * cell_size), (0, ny * cell_size)]
    origin = [bounds[0][0], bounds[1][0]]  # Extract origin coordinates properly
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

    def _draw(p, style, label):
        p = np.asarray(p)
        ax.plot(p[:, 0], p[:, 1], style, label=label)

    if raw_path is not None:
        _draw(raw_path, "r--", "raw")
    if path is not None:
        _draw(path, "b-", "pruned/opt")
        ax.plot(path[:, 0], path[:, 1], "bo", ms=3)
    if start is not None:
        ax.plot(start[0], start[1], "go", ms=8, label="start")
    if goal is not None:
        ax.plot(goal[0], goal[1], "ro", ms=8, label="goal")
    ax.legend()
    plt.grid(True)
    plt.show()


train_data_set = np.load("train_data_set.npy", allow_pickle=True).item()


for key in train_data_set:
    print(f"{key}: type={type(train_data_set[key])}, len={len(train_data_set[key])}")

flat_start = np.vstack(train_data_set["start"])   
flat_goal = np.vstack(train_data_set["goal"])    


flat_obstacles = [obs for sample in train_data_set["obstacles"] for obs in sample]
flat_obstacles = np.array(flat_obstacles, dtype=float)  


print("flat_start.shape:", flat_start.shape)
print("flat_goal.shape:", flat_goal.shape)
print("flat_obstacles.shape:", flat_obstacles.shape)
# idx = 100

# show(flat_obstacles[idx], start=flat_start[idx], goal=flat_goal[idx])
train_data_set_flatten = {
    "start": flat_start,
    "goal": flat_goal,
    "obstacles": flat_obstacles,
}

np.save("train_data_set_flatten.npy", train_data_set_flatten, allow_pickle=True)
