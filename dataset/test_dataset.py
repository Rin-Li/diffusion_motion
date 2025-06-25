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
            if grid[ix, iy]:
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

def show_multiple(grid_list, path_list, start_list, goal_list, indices, cols=10):
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, idx in enumerate(indices):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        grid = np.array(grid_list[idx])
        path = np.array(path_list[idx])  
        start = np.array(start_list[idx])
        goal = np.array(goal_list[idx])
        
        nx, ny = grid.shape[0], grid.shape[1]
        cell_size = 1
        bounds = [(0, nx * cell_size), (0, ny * cell_size)]
        origin = [bounds[0][0], bounds[1][0]]
        
        ax.set_aspect("equal")
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title(f"Sample {idx}")
        

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
        

        ax.plot(path[:, 0], path[:, 1], "b-", lw=1)
        ax.plot(start[0], start[1], "go", ms=4)
        ax.plot(goal[0], goal[1], "ro", ms=4)
        ax.grid(True, alpha=0.3)
    

    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


train_data_set = np.load("./dataset/train_data_set.npy", allow_pickle=True).item()


flat_start = train_data_set["start"]  
flat_goal = train_data_set["goal"]  
flat_map = train_data_set["map"]     
flat_paths = train_data_set["paths"]  

print(f"Size: {len(flat_paths)}")
print(f"Range: {min(len(p) for p in flat_paths)} - {max(len(p) for p in flat_paths)}")


idx = 4
print(f"Sample {idx} path shape:", np.array(flat_paths[idx]).shape)
show(flat_map[idx], np.array(flat_paths[idx]), np.array(flat_start[idx]), np.array(flat_goal[idx]))


indices = list(range(min(100, len(flat_paths))))
show_multiple(flat_map, flat_paths, flat_start, flat_goal, indices)


train_data_set_flatten = {
    "start": flat_start,
    "goal": flat_goal,
    "map": flat_map,
    "paths": flat_paths,  
}

np.save("train_data_set_flatten.npy", train_data_set_flatten, allow_pickle=True)