import numpy as np
from typing import List, Tuple


def to_index(point: np.ndarray, cell_size, origin) -> np.ndarray:
    return np.floor((point - origin) / cell_size).astype(int)

def index_in_bounds(idx: np.ndarray, grid) -> bool:
    return np.all(idx >= 0) and np.all(idx < grid.shape)


def in_collision(point: np.ndarray, grid, cell_size, origin) -> bool:

    idx = to_index(point, cell_size, origin)
    if not index_in_bounds(idx, grid):
        return True  
    if grid[tuple(idx)]:
        return True
    

    offset_scale = cell_size * 0.1  
    offsets = [
        [0, 0],                                  
        [offset_scale, 0],                  
        [-offset_scale, 0],                  
        [0, offset_scale],                 
        [0, -offset_scale],               
        [offset_scale, offset_scale],         
        [-offset_scale, offset_scale],       
        [offset_scale, -offset_scale],        
        [-offset_scale, -offset_scale],           
        [offset_scale * 0.5, offset_scale * 0.5],
        [-offset_scale * 0.5, -offset_scale * 0.5],
        [offset_scale * 0.5, -offset_scale * 0.5],
        [-offset_scale * 0.5, offset_scale * 0.5],
    ]
    
    for offset in offsets:
        test_point = point + np.array(offset)
        test_idx = to_index(test_point, cell_size, origin)
        
        if not index_in_bounds(test_idx, grid):
            continue 
            
        if grid[tuple(test_idx)]:
            return True 
    
    return False

def sample_start_goal(bounds, grid, cell_size, origin, rng):
    xmin, ymin = bounds[:, 0]
    xmax, ymax = bounds[:, 1]
    for _ in range(1000):
        start = rng.uniform([xmin, ymin], [xmax, ymax])
        goal = rng.uniform([xmin, ymin], [xmax, ymax])
        if np.linalg.norm(start - goal) < 1.0:
            continue
        if in_collision(start, grid, cell_size, origin) or in_collision(goal, grid, cell_size, origin):
            continue
        return start, goal
    raise RuntimeError("Could not sample valid start/goal after many tries.")


def random_rectangles(max_rectangles, bounds, rng) -> List[Tuple[float, float, float, float]]:
    n_rects = rng.integers(max_rectangles[0], max_rectangles[1] + 1)
    rects: List[Tuple[float, float, float, float]] = []
    xmin, ymin = bounds[:, 0]
    xmax, ymax = bounds[:, 1]
    w_max = (xmax - xmin) * 0.3  # nothing too large
    h_max = (ymax - ymin) * 0.3

    for _ in range(n_rects):
        for _ in range(100):  # retry to place inside bounds
            w = rng.uniform(0.5, w_max)
            h = rng.uniform(0.5, h_max)
            x0 = rng.uniform(xmin, xmax - w)
            y0 = rng.uniform(ymin, ymax - h)
            rect = (x0, y0, w, h)
            if not rect_overlap(rect, rects):
                rects.append(rect)
                break
    return rects

def rect_overlap( rect: Tuple[float, float, float, float], rects: List[Tuple[float, float, float, float]]):
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    for rx0, ry0, rw, rh in rects:
        rx1, ry1 = rx0 + rw, ry0 + rh
        if not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1):
            return True  # intersection
    return False

def rectangles_to_grid(nx, ny, bounds, cell_size, rects: List[Tuple[float, float, float, float]]) -> np.ndarray:
    grid = np.zeros((nx, ny), dtype=bool)
    xs = (np.arange(nx) + 0.5) * cell_size + bounds[0, 0]
    ys = (np.arange(ny) + 0.5) * cell_size + bounds[1, 0]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    for x0, y0, w, h in rects:
        mask = (XX >= x0) & (XX <= x0 + w) & (YY >= y0) & (YY <= y0 + h)
        grid[mask] = True
    return grid


def segment_in_collision(a: np.ndarray, b: np.ndarray, grid, cell_size, origin) -> bool:
    dist = np.linalg.norm(b - a)
    if dist == 0.0:
        return in_collision(a, grid, cell_size, origin)
    
    n = int(dist / (cell_size * 0.25)) + 1
    n = max(n, 10)
    
    for t in np.linspace(0.0, 1.0, n):
        p = a + t * (b - a)
        if in_collision(p, grid, cell_size, origin):
            return True
    return False

def validate_path_collision_free(path: np.ndarray, grid, cell_size, origin) -> bool:
    if len(path) < 2:
        return True
    
    # Check if the path is within bounds
    for point in path:
        if in_collision(point, grid, cell_size, origin):
            return False
    
    # Check if each segment of the path is collision-free
    for i in range(len(path) - 1):
        if segment_in_collision(path[i], path[i + 1], grid, cell_size, origin):
            return False
    
    return True
