import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from utils.dataset_utils import circle_point_in_collision, circle_segment_dist

class RRTStar:
    class _Node:
        __slots__ = ("x", "parent", "cost")
        def __init__(self, x, parent=None, cost=0.0):
            self.x      = x            
            self.parent = parent      
            self.cost   = cost         
    
    def __init__(self, bounds, max_iter=5000, step_size=1.0, goal_tol=0.3, goal_bias=0.05, gamma_star=1.5, rng=None):
        self.bounds     = np.asarray(bounds, dtype=float)
        self.dim        = self.bounds.shape[0]
        self.max_iter   = max_iter
        self.step_size  = step_size
        self.goal_tol   = goal_tol
        self.goal_bias  = goal_bias
        self.gamma_star = gamma_star
        self.rng        = np.random.default_rng(rng)
        
    def _prune_path(self, path: np.ndarray, obstacles) -> np.ndarray:
        pruned = [path[0]]
        q_temp = path[0]
        for i in range(2, len(path)):
            if self._segment_collision_free(q_temp, path[i], obstacles):
                continue
            pruned.append(path[i - 1])
            q_temp = path[i - 1]
        pruned.append(path[-1])
        return np.asarray(pruned)

    def _smooth_path(self, path: np.ndarray, obstacles, interp_points: int) -> np.ndarray:
        if len(path) >= 2:
            try:
                k = min(3, len(path) - 1)
                tck, _ = splprep(path.T, s=0, k=k)
                smoothed = np.stack(splev(np.linspace(0, 1, interp_points), tck), axis=1)
            except Exception:
                smoothed = self._linear_interp(path, interp_points)
        else:
            smoothed = self._linear_interp(path, interp_points)

        for a, b in zip(smoothed[:-1], smoothed[1:]):
            if not self._segment_collision_free(a, b, obstacles):
                result = self._linear_interp(path, interp_points)
                result[0]  = path[0]
                result[-1] = path[-1]
                return result
        smoothed = np.clip(smoothed, self.bounds[:, 0], self.bounds[:, 1])
        smoothed[0]  = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    def _linear_interp(self, pts: np.ndarray, n_total: int):

        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.insert(np.cumsum(seg_lens), 0, 0.0)
        if cum[-1] == 0:                     
            return np.repeat(pts[:1], n_total, axis=0)

        u_all = cum / cum[-1]               
        u_new = np.linspace(0, 1, n_total)
        smoothed = np.empty((n_total, self.dim))
        for d in range(self.dim):
            smoothed[:, d] = np.interp(u_new, u_all, pts[:, d])
        return smoothed


    def plan(self, start, goal, obstacles, prune: bool = True, smooth: bool = True, interp_points: int = 30):
        raw_path = self._plan_raw(start, goal, obstacles)
        if raw_path is None:
            return None
        path = np.asarray(raw_path)
        if prune:
            path = self._prune_path(path, obstacles)
        if smooth:
            return self._smooth_path(path, obstacles, interp_points)
        return self._linear_interp(path, interp_points)
    
    def _plan_raw(self, start, goal, obstacles):
        """Return path as [start, …, goal] or None on failure."""
        start, goal = map(np.asarray, (start, goal))
        assert start.shape == goal.shape == (self.dim,)
        if self._in_collision(start, obstacles) or self._in_collision(goal, obstacles):
            raise ValueError("Start or goal inside an obstacle.")
        
        nodes    = [self._Node(start)]
        best_goal_node = None
        for it in range(1, self.max_iter + 1):

            if self.rng.random() < self.goal_bias:
                x_rand = goal.copy()
            else:
                x_rand = self.rng.uniform(self.bounds[:,0], self.bounds[:,1])
            
            # Build KD-tree for fast nearest neighbor search
            positions = np.array([n.x for n in nodes])
            tree = KDTree(positions)
            
            # Find nearest node using KD-tree
            _, nearest_idx = tree.query(x_rand)
            node_near = nodes[nearest_idx]
            x_new = self._steer(node_near.x, x_rand)
            
            if not self._segment_collision_free(node_near.x, x_new, obstacles):
                continue
            
            # Find neighbors within the radius using KD-tree
            r_n = min(self.gamma_star * (log(it) / it)**(1/self.dim), self.step_size * 2)
            neighbor_indices = tree.query_ball_point(x_new, r_n)
            
            # Filter neighbors by collision check
            neighbor_ids = [
                idx for idx in neighbor_indices
                if self._segment_collision_free(nodes[idx].x, x_new, obstacles)
            ]
            
            # Select parent with minimum cost
            parent_id = min(
                neighbor_ids or [nearest_idx],
                key=lambda idx: nodes[idx].cost + np.linalg.norm(nodes[idx].x - x_new)
            )
            parent_node = nodes[parent_id]
            new_cost = parent_node.cost + np.linalg.norm(parent_node.x - x_new)
            new_node = self._Node(x_new, parent=parent_node, cost=new_cost)
            nodes.append(new_node)
            
            # Rewire
            for idx in neighbor_ids:
                nbr = nodes[idx]
                potential_cost = new_node.cost + np.linalg.norm(nbr.x - x_new)
                if potential_cost < nbr.cost and \
                   self._segment_collision_free(nbr.x, x_new, obstacles):
                    nbr.parent = new_node
                    nbr.cost   = potential_cost
            
            # Goal
            if np.linalg.norm(x_new - goal) <= self.goal_tol and \
               self._segment_collision_free(x_new, goal, obstacles):
                goal_cost = new_node.cost + np.linalg.norm(x_new - goal)
                if best_goal_node is None or goal_cost < best_goal_node.cost:
                    best_goal_node = self._Node(goal, parent=new_node, cost=goal_cost)
        
        # Return path if found
        if best_goal_node is None:
            return None  # Failure
        path = []
        node = best_goal_node
        while node is not None:
            path.append(node.x.copy())
            node = node.parent
        return path[::-1]  # start → goal
    
   # Steer
    def _steer(self, x_from, x_to):
        vec = x_to - x_from
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            x_new = x_to.copy()
        else:
            x_new = x_from + (vec / dist) * self.step_size
        return np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
    
    # Collision check (delegated to utils.dataset_utils)
    def _in_collision(self, point, obstacles):
        return circle_point_in_collision(point, obstacles)

    def _segment_collision_free(self, a, b, obstacles):
        return all(circle_segment_dist(a, b, np.asarray(c)) > r for c, r in obstacles)
    
    def visualize_path(self, bounds, obstacles, path=None, raw_path=None, start=None, goal=None):
        _, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])


        for center, radius in obstacles:
            circle = plt.Circle(center, radius, color='gray', alpha=0.5)
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

def main():
    bounds = [(0, 10), (0, 10)]   
    rrt = RRTStar(bounds, max_iter=2000, step_size=1.0, goal_tol=0.3)

    start = [1.0, 1.0]
    goal  = [9.0, 9.0]

    obstacles = [
        (np.array([5.0, 5.0]), 1.5),
        (np.array([6.0, 6.0]), 1.0),
    ]


    raw = rrt._plan_raw(start, goal, obstacles)
    if raw is None:
        print("No path found in planning.")
        return

    optimized = rrt.plan(start, goal, obstacles, prune=True, smooth=True, interp_points=50)
    if optimized is None:
        print("Planning failed.")
        return

    print("Final optimized path length:", len(optimized))

    rrt.visualize_path(bounds, obstacles, path=optimized, raw_path=raw, start=start, goal=goal)

if __name__ == "__main__":
    main()