from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
from utils.dataset_utils import random_rectangles, sample_start_goal, rectangles_to_grid

import numpy as np

from .RRT_star_grid import RRTStarGrid

import copy



@dataclass
class Sample2D:
    """One navigation instance."""

    grid: np.ndarray  # occupancy grid (bool)
    cell_size: float
    bounds: np.ndarray  # shape (2, 2)
    obstacles: List[Tuple[float, float, float, float]]  # (x_min, y_min, w, h)
    start: np.ndarray  # shape (2,)
    goal: np.ndarray  # shape (2,)
    path: np.ndarray  # (N, 2) smoothed/pruned path
    raw_path: np.ndarray  # (M, 2) 

    def to_dict(self): 
        d = asdict(self)
        d["obstacles"] = np.asarray(self.obstacles, dtype=float)
        return d



class DataGeneratorGrid:
    def __init__(
        self,
        bounds: List[Tuple[float, float]] | np.ndarray,
        num_samples: int,
        *,
        collection_data: bool = True,
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
        self.collection_data = collection_data
        self.current_sample = 0
        self.training_data_set = {
            "start": [],
            "goal": [],
            "paths": [],
            "map": [],
            
        }

        # grid size
        self.nx, self.ny = (
            int((hi - lo) / self.resolution) for lo, hi in self.bounds
        )

    def generate_dataset(self, smooth: bool = True, interp: int = 100) -> List[Sample2D]:
        samples: List[Sample2D] = []
        attempts = 0
        while len(samples) < self.num_samples and attempts < self.num_samples * 200:
            print(f"Generating sample {len(samples) + 1}/{self.num_samples} (attempts: {attempts})")
            print(f"current training data set length: {len(self.training_data_set['start'])}")
            attempts += 1
            rects = random_rectangles(self.max_rectangles, self.bounds, self.rng)
            grid = rectangles_to_grid(self.nx, self.ny, self.bounds, self.resolution, rects)
            start, goal = sample_start_goal(self.bounds, grid, self.resolution,self.origin, self.rng)

            planner = RRTStarGrid(
                self.bounds,
                grid,
                self.resolution,
                collect_training_data=self.collection_data,
                max_iter=self.max_iter_rrt,
                step_size=self.step_size,
                goal_tol=self.goal_tol,
                rng=self.rng,
            )
            path = planner.plan(
                start,
                goal,
                prune=True,
                optimize=smooth,
                interp_points=interp,
            )
            

            if path is None:
                continue
            
            self.training_data_set["start"].append(copy.deepcopy(start))
            self.training_data_set["goal"].append(copy.deepcopy(goal))
            self.training_data_set["map"].append(copy.deepcopy(grid))
            self.training_data_set["paths"].append(copy.deepcopy(path))
            

            raw_path = planner._plan_raw(start, goal)  # noqa: SLF001 – keep raw
            samples.append(
                Sample2D(
                    grid=grid,
                    cell_size=self.resolution,
                    bounds=self.bounds,
                    obstacles=rects,
                    start=start,
                    goal=goal,
                    path=path,
                    raw_path=np.asarray(raw_path),
                )
            )
        if len(samples) < self.num_samples:
            raise RuntimeError("Could not create the requested number of samples.")
        return samples, self.training_data_set

    def save_npz(self, samples: List[Sample2D], outfile: str | Path) -> None:
        arr = {f"sample_{i}": s.to_dict() for i, s in enumerate(samples)}
        np.savez_compressed(outfile, **arr)
    
    # def in_collision(self, point: np.ndarray, grid) -> bool:
    #     idx = self._to_index(point)
    #     if not self._index_in_bounds(idx, grid):
    #         return True  
        
    #   
    #     for di in [-1, 0, 1]:
    #         for dj in [-1, 0, 1]:
    #             check_idx = idx + np.array([di, dj])
    #             if self._index_in_bounds(check_idx, grid):
    #                 if grid[tuple(check_idx)]:
    # 
    #                     cell_center = (check_idx + 0.5) * self.resolution + self.origin
    #                     if np.linalg.norm(point - cell_center) < self.resolution * 0.5:
    #                         return True
    #     return False
    

    def save_train_data(self, out_file: str | Path):
        np.save(out_file, self.training_data_set)


if __name__ == "__main__":
    gen = DataGeneratorGrid(bounds=[(0, 8), (0, 8)], num_samples=10, rng=30)
    ds, train_data_set = gen.generate_dataset()
    gen.save_train_data("train_data_set.npy")
    train_data_set = np.load("train_data_set.npy", allow_pickle=True).item()

