import numpy as np
from data_generator.RRT_star import RRTStar
from pathlib import Path
from typing import Union


class DataGenerator2D:
    """
    Generate 2D path-planning datasets with circular obstacles in continuous space.

    Each sample contains:
        - start / goal  : 2D positions  (float32)
        - path          : interpolated trajectory  (interp_points, 2)  float32
        - map           : binary occupancy image   (map_resolution, map_resolution)  float32

    Output is saved as a single .npy file (dict) compatible with PlanePlanningDataSets.
    Keys: 'start', 'goal', 'paths', 'map'
    """

    def __init__(
        self,
        bounds,
        num_samples: int,
        max_obstacles: int = 5,
        max_iter_per_sample: int = 100,
        map_resolution: int = 64,
        interp_points: int = 50,
        outfile: Union[str, Path] = "rrt_dataset.npy",
    ):
        self.bounds           = np.asarray(bounds, dtype=float)  # (2, 2)
        self.num_samples      = num_samples
        self.max_obstacles    = max_obstacles
        self.max_iter_per_sample = max_iter_per_sample
        self.map_resolution   = map_resolution
        self.interp_points    = interp_points
        self.rrt_star         = RRTStar(bounds=bounds)
        self.outfile          = Path(outfile)

        self.starts, self.goals = [], []
        self.paths, self.maps   = [], []

    # ------------------------------------------------------------------
    # sampling helpers
    # ------------------------------------------------------------------

    def _random_start_goal_obs(self):
        """Sample a valid (start, goal, obstacles) triple."""
        for _ in range(self.max_iter_per_sample):
            start = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            goal  = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            if np.linalg.norm(start - goal) < 1e-3:
                continue

            obstacles = []
            for _ in range(np.random.randint(1, self.max_obstacles + 1)):
                center = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                radius = np.random.uniform(0.3, 3.0)
                obstacles.append((center, radius))

            if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
                continue
            if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
                continue

            return start, goal, obstacles
        return None

    def _obstacles_to_map(self, obstacles) -> np.ndarray:
        """
        Rasterize a list of circular obstacles into a binary occupancy map.

        Each pixel is 1 if it falls inside any obstacle circle, else 0.

        Returns:
            occ : (map_resolution, map_resolution) float32
        """
        res  = self.map_resolution
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]

        # Build pixel-center coordinate grids
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        grid_x, grid_y = np.meshgrid(xs, ys)   # both (res, res)

        occ = np.zeros((res, res), dtype=np.float32)
        for (cx, cy), r in obstacles:
            dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            occ[dist <= r] = 1.0

        return occ

    # ------------------------------------------------------------------
    # generation loop
    # ------------------------------------------------------------------

    def generate(self):
        n_success = 0
        while n_success < self.num_samples:
            sample = self._random_start_goal_obs()
            if sample is None:
                continue

            start, goal, obstacles = sample
            path = self.rrt_star.plan(
                start, goal, obstacles,
                optimize=True, interp_points=self.interp_points,
            )

            if path is None:
                continue

            self.starts.append(start.astype(np.float32))
            self.goals.append(goal.astype(np.float32))
            self.paths.append(np.asarray(path, dtype=np.float32))       # (T, 2)
            self.maps.append(self._obstacles_to_map(obstacles))          # (H, W)

            n_success += 1
            if n_success % 50 == 0 or n_success == self.num_samples:
                print(f"Generated {n_success}/{self.num_samples} paths.")

        self._save_npy()
        print(f"Saved dataset to: {self.outfile.resolve()}")

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def _save_npy(self):
        """
        Save dataset as a .npy dict.

        Keys match PlanePlanningDataSets expectations:
            'start'  : (N, 2)      float32
            'goal'   : (N, 2)      float32
            'paths'  : (N, T, 2)   float32
            'map'    : (N, H, W)   float32   — raw map; dataset adds channel dim
        """
        data = {
            "start": np.stack(self.starts, axis=0),   # (N, 2)
            "goal":  np.stack(self.goals,  axis=0),   # (N, 2)
            "paths": np.stack(self.paths,  axis=0),   # (N, T, 2)
            "map":   np.stack(self.maps,   axis=0),   # (N, H, W)
        }
        np.save(self.outfile, data)


if __name__ == "__main__":
    bounds = [(0, 8), (0, 8)]
    gen = DataGenerator2D(
        bounds,
        num_samples=500,
        map_resolution=64,
        outfile="rrt_2d_dataset_500.npy",
    )
    gen.generate()
