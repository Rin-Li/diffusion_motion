import numpy as np
from RRT_star import RRTStar
from pathlib import Path

class DataGenerator2D:
    def __init__(self, bounds, num_samples: int, max_obstacles: int = 5, max_iter_per_sample: int = 100, outfile: str | Path = "rrt_dataset.npz"):
        self.bounds = np.asarray(bounds, dtype=float)
        self.num_samples = num_samples
        self.max_obstacles = max_obstacles
        self.max_iter_per_sample = max_iter_per_sample
        self.rrt_star = RRTStar(bounds=bounds)
        self.outfile = Path(outfile)

        self.starts, self.goals = [], []
        self.paths, self.obstacles_all = [], []

    def _random_start_goal_obs(self):

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

    def generate(self):
        n_success = 0
        while n_success < self.num_samples:
            sample = self._random_start_goal_obs()
            if sample is None:
                continue

            start, goal, obstacles = sample
            path = self.rrt_star.plan(
                start, goal, obstacles, optimize=True, interp_points=50
            )

            if path is None:
                continue

            self.starts.append(start.astype(np.float32))
            self.goals.append(goal.astype(np.float32))
            self.paths.append(np.asarray(path, dtype=np.float32))         
            self.obstacles_all.append(np.asarray(
                [(c[0], c[1], r) for c, r in obstacles], dtype=np.float32
            ))                                                        
            n_success += 1
            if n_success % 50 == 0 or n_success == self.num_samples:
                print(f"Generate for {n_success}/{self.num_samples} paths.")

        self._save_npz()
        print(f"Save data set to: {self.outfile.resolve()}")


    def _save_npz(self):

        np.savez_compressed(
            self.outfile,
            starts=np.array(self.starts, dtype=object),
            goals=np.array(self.goals, dtype=object),
            paths=np.array(self.paths, dtype=object),
            obstacles=np.array(self.obstacles_all, dtype=object),
        )


if __name__ == "__main__":
    bounds = [(0, 8), (0, 8)]    # x, y 范围
    gen_1 = DataGenerator2D(bounds, num_samples=500, outfile="rrt_2d_dataset_500.npz")
    gen_1.generate()
    gen_2 = DataGenerator2D(bounds, num_samples=1000, outfile="rrt_2d_dataset_1000.npz")
    gen_2.generate()
