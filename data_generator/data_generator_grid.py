from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.dataset_utils import random_rectangles, sample_start_goal, rectangles_to_grid

import numpy as np
from tqdm import tqdm

from .RRT_star_grid import RRTStarGrid


class DataGeneratorGrid:
    """
    Generates a navigation dataset by sampling random occupancy grids,
    start/goal pairs, and solving each with RRT*.

    Output format is compatible with PlanePlanningDataSets:
        { 'start': (N,2), 'goal': (N,2), 'paths': (N,T,2), 'map': (N,H,W) }
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]] | np.ndarray,
        num_samples: int,
        *,
        resolution: float = 1.0,
        max_rectangles: Tuple[int, int] = (2, 6),
        step_size: float = 0.5,
        max_iter_rrt: int = 2000,
        goal_tol: float = 0.3,
        rng: Optional[int | np.random.Generator] = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.num_samples = int(num_samples)
        self.resolution = float(resolution)
        self.max_rectangles = max_rectangles
        self.step_size = step_size
        self.max_iter_rrt = max_iter_rrt
        self.goal_tol = goal_tol
        self.rng = np.random.default_rng(rng)
        self.origin = self.bounds[:, 0]
        self.nx, self.ny = (
            int((hi - lo) / self.resolution) for lo, hi in self.bounds
        )
        self._data: Dict[str, list] = {"start": [], "goal": [], "paths": [], "map": []}

    def generate(self, smooth: bool = True, interp: int = 100) -> Dict[str, np.ndarray]:
        """
        Run RRT* to collect `num_samples` successful trajectories.

        Returns a dict ready for PlanePlanningDataSets:
            { 'start': (N,2), 'goal': (N,2), 'paths': (N,T,2), 'map': (N,H,W) }
        """
        self._data = {"start": [], "goal": [], "paths": [], "map": []}
        max_attempts = self.num_samples * 200
        attempts = 0

        with tqdm(total=self.num_samples, desc="Generating samples") as pbar:
            while len(self._data["start"]) < self.num_samples:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Reached {max_attempts} attempts but only collected "
                        f"{len(self._data['start'])}/{self.num_samples} samples. "
                        "Consider relaxing obstacle density or increasing max_attempts."
                    )
                attempts += 1

                rects = random_rectangles(self.max_rectangles, self.bounds, self.rng)
                grid = rectangles_to_grid(
                    self.nx, self.ny, self.bounds, self.resolution, rects
                )
                start, goal = sample_start_goal(
                    self.bounds, grid, self.resolution, self.origin, self.rng
                )

                planner = RRTStarGrid(
                    self.bounds,
                    grid,
                    self.resolution,
                    max_iter=self.max_iter_rrt,
                    step_size=self.step_size,
                    goal_tol=self.goal_tol,
                    rng=self.rng,
                )
                path = planner.plan(
                    start, goal, prune=True, optimize=smooth, interp_points=interp
                )

                if path is None:
                    continue

                self._data["start"].append(start)
                self._data["goal"].append(goal)
                self._data["map"].append(grid)
                self._data["paths"].append(path)
                pbar.update(1)
                pbar.set_postfix(attempts=attempts)

        return {
            "start": np.array(self._data["start"], dtype=np.float32),
            "goal":  np.array(self._data["goal"],  dtype=np.float32),
            "paths": np.array(self._data["paths"], dtype=np.float32),
            "map":   np.array(self._data["map"],   dtype=np.float32),
        }

    def save(self, dataset: Dict[str, np.ndarray], outfile: str | Path) -> None:
        """
        Save dataset to a .npy file.
        Load with: np.load(outfile, allow_pickle=True).item()
        """
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        np.save(outfile, dataset)
        print(f"Dataset saved → {outfile.resolve()}  ({len(dataset['paths'])} samples)")


if __name__ == "__main__":
    gen = DataGeneratorGrid(bounds=[(0, 8), (0, 8)], num_samples=10, rng=30)
    dataset = gen.generate()
    gen.save(dataset, "dataset/train_data_set.npy")

