import numpy as np
import matplotlib.pyplot as plt

data = np.load("train_data_set.npy", allow_pickle=True)
training_data = data["arr_0"].item() 

starts = np.array(training_data["start"])
goals = np.array(training_data["goal"])
obstacles_list = training_data["map"]  


idx = 0  
start = starts[idx]
goal = goals[idx]
obstacles = obstacles_list[idx] 


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

    ax.plot(start[0], start[1], "go", ms=8, label="start")
    ax.plot(goal[0], goal[1], "ro", ms=8, label="goal")
    ax.legend()
    plt.grid(True)
    plt.show()

# 假设你原始用的是这个 bounds 和 cell_size
bounds = [(0.0, 10.0), (0.0, 10.0)]
cell_size = 0.1

visualize_training_sample(start, goal, obstacles, bounds, cell_size)
