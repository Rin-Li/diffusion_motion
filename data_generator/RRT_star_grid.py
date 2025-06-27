import numpy as np
from math import log
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev  
from utils.dataset_utils import in_collision

class RRTStarGrid:
    class _Node:
        __slots__ = ("x", "parent", "cost")

        def __init__(self, x: np.ndarray, parent=None, cost: float = 0.0):
            self.x = x
            self.parent = parent
            self.cost = cost
            
    def __init__(
        self,
        bounds,
        grid: np.ndarray,
        cell_size: float,
        *,
        collect_training_data: bool = True,
        max_iter: int = 1000,
        step_size: float = 0.5,
        goal_tol: float = 0.5,
        goal_bias: float = 0.1,
        gamma_star: float = 1.5,
        min_points: int = 32,
        rng=None,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        assert self.dim in (2, 3), "Only 2-D or 3-D occupancy grids supported."
        self.grid = grid.astype(bool)
        self.cell_size = float(cell_size)
        self.origin = self.bounds[:, 0]  # lower workspace corner

        # sanity check on grid shape
        exp_shape = tuple(
            int(round((hi - lo) / self.cell_size)) for lo, hi in self.bounds
        )
        if self.grid.shape != exp_shape:
            raise ValueError(
                f"Grid shape {self.grid.shape} does not match bounds/resolution {exp_shape}"
            )

        # RRT* parameters
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tol = goal_tol
        self.goal_bias = goal_bias
        self.gamma_star = gamma_star
        self.min_points = min_points  # for path interpolation
        self.rng = np.random.default_rng(rng)
        self.collet_traing_data = collect_training_data

    def plan(self, start, goal, *, prune: bool = False, optimize: bool = False, interp_points: int = 50):
        raw_path = self._plan_raw(np.asarray(start), np.asarray(goal))
        if raw_path is None:
            return None  # planning failure

        path = np.asarray(raw_path)

        if prune:
            path = self._prune_path(path)
            # purning
            if not self._validate_path_collision_free(path):
                print("Warning: Pruned path has collisions, using raw path")
                path = np.asarray(raw_path)

        if optimize:
            smoothed = self._smooth_path(path, interp_points)
            if smoothed is not False:
                path = smoothed
            else:

                path = self._interpolate_path(path, min_points=self.min_points)


        if not self._validate_path_collision_free(path):
            print("Warning: Final path has collisions!")
            return None

        return path

    # Private
    # Original plane
    def _plan_raw(self, start: np.ndarray, goal: np.ndarray):
        if in_collision(start, self.grid, self.cell_size, self.origin) or in_collision(goal, self.grid, self.cell_size, self.origin):
            raise ValueError("Start or goal is inside an obstacle / outside grid.")

        nodes = [self._Node(start)]
        best_goal_node = None

        for it in range(1, self.max_iter + 1):
            x_rand = goal.copy() if self.rng.random() < self.goal_bias else self._sample_free()

            node_near = min(nodes, key=lambda n: np.linalg.norm(n.x - x_rand))
            x_new = self._steer(node_near.x, x_rand)
            # Collect the traning dataset
            if self._segment_in_collision(node_near.x, x_new):
                continue

            r_n = min(self.gamma_star * (log(it) / it) ** (1 / self.dim), self.step_size * 2)
            neighbour_ids = [
                idx
                for idx, n in enumerate(nodes)
                if np.linalg.norm(n.x - x_new) <= r_n
                and not self._segment_in_collision(n.x, x_new)
            ]
            parent_idx = min(
                neighbour_ids or [nodes.index(node_near)],
                key=lambda idx: nodes[idx].cost + np.linalg.norm(nodes[idx].x - x_new),
            )
            parent_node = nodes[parent_idx]
            new_cost = parent_node.cost + np.linalg.norm(parent_node.x - x_new)
            new_node = self._Node(x_new, parent=parent_node, cost=new_cost)
            nodes.append(new_node)

            # re-wire
            for idx in neighbour_ids:
                nbr = nodes[idx]
                potential = new_node.cost + np.linalg.norm(nbr.x - x_new)
                if potential < nbr.cost and not self._segment_in_collision(nbr.x, x_new):
                    nbr.parent, nbr.cost = new_node, potential

            # goal check
            if np.linalg.norm(x_new - goal) <= self.goal_tol and not self._segment_in_collision(x_new, goal):
                g_cost = new_node.cost + np.linalg.norm(x_new - goal)
                if best_goal_node is None or g_cost < best_goal_node.cost:
                    best_goal_node = self._Node(goal, parent=new_node, cost=g_cost)
                    break

        if best_goal_node is None:
            return None

        # back-track
        path = []
        node = best_goal_node
        while node is not None:
            path.append(node.x.copy())
            node = node.parent
        return path[::-1]
    
    # Prune the path by removing unnecessary points
    def _prune_path(self, path: np.ndarray) -> np.ndarray:
        if path.shape[0] < 3:
            return path  # nothing to prune

        pruned = [path[0]]
        anchor = path[0]
        for i in range(2, path.shape[0]):
            if self._segment_in_collision(anchor, path[i]):
                pruned.append(path[i - 1])
                anchor = path[i - 1]
        pruned.append(path[-1])
        return np.asarray(pruned)

    def _to_index(self, point: np.ndarray) -> np.ndarray:
        return np.floor((point - self.origin) / self.cell_size).astype(int)

    def _index_in_bounds(self, idx: np.ndarray) -> bool:
        return np.all(idx >= 0) and np.all(idx < self.grid.shape)
    
    # def in_collision(self, point: np.ndarray) -> bool:
    #     idx = self._to_index(point)
    #     if not self._index_in_bounds(idx):
    #         return True  
        
    #   
    #     for di in [-1, 0, 1]:
    #         for dj in [-1, 0, 1]:
    #             check_idx = idx + np.array([di, dj])
    #             if self._index_in_bounds(check_idx):
    #                 if self.grid[tuple(check_idx)]:
    #              
    #                     cell_center = (check_idx + 0.5) * self.cell_size + self.origin
    #                     if np.linalg.norm(point - cell_center) < self.cell_size * 0.5:
    #                         return True
    #     return False
    

    def _segment_in_collision(self, a: np.ndarray, b: np.ndarray) -> bool:
        dist = np.linalg.norm(b - a)
        if dist == 0.0:
            return in_collision(a, self.grid, self.cell_size, self.origin)
        
        n = int(dist / (self.cell_size * 0.25)) + 1
        n = max(n, 10)
        
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * (b - a)
            if in_collision(p, self.grid, self.cell_size, self.origin):
                return True
        return False

    def _validate_path_collision_free(self, path: np.ndarray) -> bool:
        if len(path) < 2:
            return True
        
        # Check if the path is within bounds
        for point in path:
            if in_collision(point, self.grid, self.cell_size, self.origin):
                return False
        
        # Check if each segment of the path is collision-free
        for i in range(len(path) - 1):
            if self._segment_in_collision(path[i], path[i + 1]):
                return False
        
        return True

    def _sample_free(self) -> np.ndarray:
        for _ in range(1000):
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            if not in_collision(x, self.grid, self.cell_size, self.origin):
                return x
        raise RuntimeError("Failed to sample a free point – maybe the map is full?")

    def _steer(self, x_from: np.ndarray, x_to: np.ndarray) -> np.ndarray:
        vec = x_to - x_from
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            return x_to.copy()
        return x_from + (vec / dist) * self.step_size

    def _smooth_path(self, path, n_interp):
        if path is None or len(path) < 2:
            return path
        
        pts = np.asarray(path)
        if len(pts) == 2:                      
            u = np.linspace(0.0, 1.0, self.min_points)
            smoothed = pts[0] + (pts[1] - pts[0]) * u[:, None]
        else:
            k = min(3, len(pts) - 1)
            tck, _ = splprep(pts.T, s=0, k=k)
            u = np.linspace(0.0, 1.0, self.min_points)       
            coords = splev(u, tck)                   
            smoothed = np.stack(coords, axis=1)        

        # Collision check
        if not self._validate_path_collision_free(smoothed):
            return False
        
        return smoothed
    
    def _interpolate_path(self, path, min_points):
        if len(path) >= min_points:
            return path
        
        # Calculate total path length
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        total_length = np.sum(distances)
        
        # Create interpolation points
        current_length = 0
        interpolated_path = [path[0]]
        
        target_segment_length = total_length / (min_points - 1)
        
        for i in range(len(path) - 1):
            segment_length = distances[i]
            segment_start = path[i]
            segment_end = path[i + 1]
            
            while current_length + segment_length >= target_segment_length * len(interpolated_path):
                remaining_length = target_segment_length * len(interpolated_path) - current_length
                ratio = remaining_length / segment_length
                
                new_point = segment_start + ratio * (segment_end - segment_start)
                interpolated_path.append(new_point)
                
                if len(interpolated_path) >= min_points:
                    break
            
            current_length += segment_length
            
            if len(interpolated_path) >= min_points:
                break
        
        if len(interpolated_path) < min_points:
            interpolated_path.append(path[-1])
        else:
            interpolated_path[-1] = path[-1]
        
        result = np.array(interpolated_path)
        
        # Check collision
        if not self._validate_path_collision_free(result):

            return path
        
        return result

    def show(self, path=None, raw_path=None, start=None, goal=None, figsize=(6, 6)):
        nx, ny = self.grid.shape
        _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_title("RRT on occupancy grid (pruning=%s)" % ("on" if path is not None else "off"))

        xs = np.arange(nx) * self.cell_size + self.origin[0]
        ys = np.arange(ny) * self.cell_size + self.origin[1]
        for ix in range(nx):
            for iy in range(ny):
                if self.grid[ix, iy]:
                    rect = plt.Rectangle(
                        (xs[ix], ys[iy]),
                        self.cell_size,
                        self.cell_size,
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


if __name__ == "__main__":
    bounds = [(0.0, 8.0), (0.0, 8.0)]
    cell = 1
    nx, ny = (int((b[1] - b[0]) / cell) for b in bounds)
    grid = np.zeros((nx, ny), dtype=bool)

    obstacles = [(np.array([4.0, 5.0]), 1.5), (np.array([10.0, 10.0]), 1.0)]
    Xs = (np.arange(nx) + 0.5) * cell + bounds[0][0]
    Ys = (np.arange(ny) + 0.5) * cell + bounds[1][0]
    XX, YY = np.meshgrid(Xs, Ys, indexing="ij")
    for centre, radius in obstacles:
        mask = (XX - centre[0]) ** 2 + (YY - centre[1]) ** 2 <= radius ** 2
        grid[mask] = True

    planner = RRTStarGrid(bounds, grid, cell, max_iter=200, step_size=0.5, goal_tol=0.3)
    start, goal = np.array([1.0, 1.0]), np.array([7.0, 7.0])

    path = planner.plan(start, goal, prune=True, optimize=True, interp_points=100)
    
    if path is None:
        print("No path found.")
    else:
        planner.show(path=path, start=start, goal=goal)