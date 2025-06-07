import numpy as np
from math import log
import matplotlib.pyplot as plt

class RRTStar:
    class _Node:
        __slots__ = ("x", "parent", "cost")
        def __init__(self, x, parent=None, cost=0.0):
            self.x      = x            
            self.parent = parent      
            self.cost   = cost         
    
    def __init__(self, bounds, max_iter=5000, step_size=0.1, goal_tol=0.5, goal_bias=0.05, gamma_star=1.5, rng=None):
        self.bounds     = np.asarray(bounds, dtype=float)
        self.dim        = self.bounds.shape[0]
        self.max_iter   = max_iter
        self.step_size  = step_size
        self.goal_tol   = goal_tol
        self.goal_bias  = goal_bias
        self.gamma_star = gamma_star
        self.rng        = np.random.default_rng(rng)
        
    def optimize_path(self, path, obstacles, interp_points: int = 30):

        if path is None or len(path) < 2:
            return False   

        pruned = [path[0]]
        q_temp = path[0]
        for i in range(2, len(path)):
            if self._segment_collision_free(q_temp, path[i], obstacles):
                continue
            pruned.append(path[i - 1])
            q_temp = path[i - 1]
        pruned.append(path[-1])
        pruned = np.asarray(pruned)

        if len(pruned) >= 4:
            try:
                from scipy.interpolate import splprep, splev

                k = min(3, len(pruned) - 1)
                # pruned.T 形状 (d, n)
                tck, _ = splprep(pruned.T, s=0, k=k)
                u_new = np.linspace(0, 1, interp_points)
                coords = splev(u_new, tck)
                smoothed = np.stack(coords, axis=1)
            except (ImportError, Exception):
                smoothed = self._linear_interp(pruned, interp_points)
        else:
            smoothed = self._linear_interp(pruned, interp_points)


        for a, b in zip(smoothed[:-1], smoothed[1:]):
            if not self._segment_collision_free(a, b, obstacles):
                return False                 
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


    def plan(self, start, goal, obstacles, optimize: bool = False, interp_points: int = 30):
        raw_path = self._plan_raw(start, goal, obstacles) 
        if raw_path is None:
            return None
        if not optimize:
            return raw_path
        optimized = self.optimize_path(raw_path, obstacles, interp_points)
        return optimized if optimized is not False else None
    
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
            
            node_near = min(nodes, key=lambda n: np.linalg.norm(n.x - x_rand))
            x_new     = self._steer(node_near.x, x_rand)
            
            if not self._segment_collision_free(node_near.x, x_new, obstacles):
                continue
            
            # Find neighbors within the radius
            r_n = min(self.gamma_star * (log(it) / it)**(1/self.dim), self.step_size * 2)
            neighbor_ids = [
                idx for idx, nd in enumerate(nodes)
                if np.linalg.norm(nd.x - x_new) <= r_n
                and self._segment_collision_free(nd.x, x_new, obstacles)
            ]
            # No neighbors found
            parent_id = min(
                neighbor_ids or [nodes.index(node_near)],
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
            return x_to.copy()
        return x_from + (vec / dist) * self.step_size
    
    # Collision check
    @staticmethod
    def _segment_to_sphere_dist(a, b, center):
        ab = b - a
        denom = np.dot(ab, ab)
        if denom == 0.0:
            return np.linalg.norm(a - center)
        t  = np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0)
        closest = a + t * ab
        return np.linalg.norm(closest - center)
    
    def _in_collision(self, point, obstacles):
        for c, r in obstacles:
            if np.linalg.norm(point - c) <= r:
                return True
        return False
    
    def _segment_collision_free(self, a, b, obstacles):
        for c, r in obstacles:
            if self._segment_to_sphere_dist(a, b, c) <= r:
                return False
        return True
    
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
    rrt = RRTStar(bounds, max_iter=8000, step_size=0.5, goal_tol=0.3)

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

    optimized = rrt.optimize_path(raw, obstacles, interp_points=50)
    if optimized is False:
        print("Optimization failed due to collision.")
        return

    print("Final optimized path length:", len(optimized))
        
    rrt.visualize_path(bounds, obstacles, path=optimized, raw_path=raw, start=start, goal=goal)

if __name__ == "__main__":
    main()