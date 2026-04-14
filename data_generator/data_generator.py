import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union

from data_generator.RRT_star import RRTStar
from utils.scenario_utils import line_blocked as _line_blocked


def _worker(args):
    """Module-level worker for multiprocessing.Pool (must be picklable)."""
    bounds, min_obstacles, max_obstacles, max_iter, interp_points, map_resolution, radius_range, crop, smooth = args
    rng = np.random.default_rng()
    bounds_arr = np.asarray(bounds, dtype=float)
    rrt = RRTStar(bounds=bounds)

    for _ in range(max_iter):
        start = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        goal  = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])

        if np.linalg.norm(start - goal) < 2.0:
            continue

        obstacles = []
        for _ in range(rng.integers(min_obstacles, max_obstacles + 1)):
            center = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            radius = rng.uniform(*radius_range)
            obstacles.append((center, radius))

        if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
            continue
        if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
            continue

        if not _line_blocked(start, goal, obstacles):
            continue

        path = rrt.plan(start, goal, obstacles, prune=crop, smooth=smooth, interp_points=interp_points)
        if path is None:
            continue

        # build occupancy map
        res = map_resolution
        xmin, xmax = bounds_arr[0]
        ymin, ymax = bounds_arr[1]
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        grid_x, grid_y = np.meshgrid(xs, ys)
        occ = np.zeros((res, res), dtype=np.float32)
        for (cx, cy), r in obstacles:
            dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            occ[dist <= r] = 1.0

        return (
            start.astype(np.float32),
            goal.astype(np.float32),
            np.asarray(path, dtype=np.float32),
            occ,
        )
    return None


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
        min_obstacles: int = 1,
        max_obstacles: int = 5,
        radius_range: tuple = (0.3, 1.2),
        max_iter_per_sample: int = 100,
        map_resolution: int = 64,
        interp_points: int = 50,
        outfile: Union[str, Path] = "rrt_dataset.npy",
        crop: bool = True,
        smooth: bool = True,
    ):
        self.bounds           = np.asarray(bounds, dtype=float)  # (2, 2)
        self.num_samples      = num_samples
        self.min_obstacles    = min_obstacles
        self.max_obstacles    = max_obstacles
        self.radius_range     = radius_range
        self.max_iter_per_sample = max_iter_per_sample
        self.map_resolution   = map_resolution
        self.interp_points    = interp_points
        self.rrt_star         = RRTStar(bounds=bounds)
        self.outfile          = Path(outfile)
        self.crop             = crop
        self.smooth           = smooth

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

            if np.linalg.norm(start - goal) < 2.0:
                continue

            obstacles = []
            for _ in range(np.random.randint(self.min_obstacles, self.max_obstacles + 1)):
                center = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                radius = np.random.uniform(*self.radius_range)
                obstacles.append((center, radius))

            if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
                continue
            if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
                continue

            # reject samples where the straight line is not blocked — avoids trivial paths
            if not _line_blocked(start, goal, obstacles):
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
    # generation loop (single process)
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
                prune=self.crop, smooth=self.smooth, interp_points=self.interp_points,
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
    # generation loop (multiprocessing)
    # ------------------------------------------------------------------

    def generate_mp(self, num_workers=None):
        """Generate dataset using a map-reduce pattern with multiprocessing.Pool."""
        if num_workers is None:
            num_workers = cpu_count()

        print(f"Using {num_workers} workers to generate {self.num_samples} samples...", flush=True)

        task_args = (
            self.bounds.tolist(),
            self.min_obstacles,
            self.max_obstacles,
            self.max_iter_per_sample,
            self.interp_points,
            self.map_resolution,
            self.radius_range,
            self.crop,
            self.smooth,
        )

        # Submit num_workers * 10 tasks per round so the OS scheduler keeps all
        # cores busy without manual load balancing.
        batch_size    = num_workers * 10
        collected     = []
        last_reported = 0
        report_every  = 100

        with Pool(num_workers) as pool:
            while len(collected) < self.num_samples:
                needed = self.num_samples - len(collected)

                # -- Map: run batch_size workers in parallel --
                results = pool.map(_worker, [task_args] * batch_size)

                # -- Reduce: filter valid results --
                valid = [r for r in results if r is not None]
                collected.extend(valid[:needed])

                if len(collected) - last_reported >= report_every or len(collected) >= self.num_samples:
                    print(f"Generated {len(collected)}/{self.num_samples} paths.", flush=True)
                    last_reported = len(collected)

        self.starts = [r[0] for r in collected]
        self.goals  = [r[1] for r in collected]
        self.paths  = [r[2] for r in collected]
        self.maps   = [r[3] for r in collected]

        self._save_npy()
        print(f"Saved dataset to: {self.outfile.resolve()}", flush=True)

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
            "start": np.stack(self.starts[:self.num_samples], axis=0),
            "goal":  np.stack(self.goals[:self.num_samples],  axis=0),
            "paths": np.stack(self.paths[:self.num_samples],  axis=0),
            "map":   np.stack(self.maps[:self.num_samples],   axis=0),
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
