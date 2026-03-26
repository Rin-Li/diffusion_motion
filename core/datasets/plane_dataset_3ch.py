"""
3-channel map dataset for continuous path planning.

Map channels:
    ch0 : binary obstacle occupancy  (same as original)
    ch1 : start position gaussian blob
    ch2 : goal  position gaussian blob

Start and goal are NOT returned as a separate 'env' key — they are fully
encoded in the map image. The 'env' key is kept only for external use
(e.g. hard-clamping trajectory endpoints after inference).
"""

import torch
import numpy as np

from core.datasets.plane_dataset_embeed import get_data_stats, normalize_data


def _gaussian_blob(center_xy, bounds, shape=(64, 64), sigma=2.0):
    """
    Rasterize a 2-D gaussian blob centred at `center_xy` (in world coords).

    Args:
        center_xy : (2,) array-like  [x, y] in world space
        bounds    : [(xmin, xmax), (ymin, ymax)]
        shape     : (H, W) output grid size
        sigma     : std-dev in pixels
    Returns:
        blob : (H, W) float32, peak value 1.0
    """
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    H, W = shape

    cx = (center_xy[0] - xmin) / (xmax - xmin) * (W - 1)
    cy = (center_xy[1] - ymin) / (ymax - ymin) * (H - 1)

    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    blob = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return blob.astype(np.float32)


class PlanePlanningDataSets3Ch(torch.utils.data.Dataset):
    """
    Drop-in replacement for PlanePlanningDataSets that uses a 3-channel map.

    Returned batch keys:
        'sample' : (T, 2)       normalized action trajectory
        'map'    : (3, H, W)    3-channel map (obstacles, start blob, goal blob)
        'env'    : (4,)         [start_x, start_y, goal_x, goal_y]  — for hard clamp only
    """

    def __init__(self, dataset_path: str, bounds=((0, 8), (0, 8)), sigma: float = 2.0):
        data = np.load(dataset_path, allow_pickle=True).item()

        self.paths = torch.tensor(data['paths'], dtype=torch.float32)   # (N, T, 2)
        self.start = torch.tensor(data['start'], dtype=torch.float32)   # (N, 2)
        self.goal  = torch.tensor(data['goal'],  dtype=torch.float32)   # (N, 2)
        raw_map    = data['map']                                         # (N, H, W) float32

        N, H, W = raw_map.shape

        # normalize trajectories using fixed bounds [0, 8] — must match policy config["normalizer"]
        stats = {
            'min': torch.tensor([0.0, 0.0]),
            'max': torch.tensor([8.0, 8.0]),
        }
        self.paths = normalize_data(self.paths, stats)

        # build 3-channel maps: (N, 3, H, W)
        ch1 = np.stack([
            _gaussian_blob(data['start'][i], bounds, shape=(H, W), sigma=sigma)
            for i in range(N)
        ])  # (N, H, W)
        ch2 = np.stack([
            _gaussian_blob(data['goal'][i], bounds, shape=(H, W), sigma=sigma)
            for i in range(N)
        ])  # (N, H, W)

        map3 = np.stack([raw_map, ch1, ch2], axis=1)   # (N, 3, H, W)
        self.map = torch.tensor(map3, dtype=torch.float32)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        return {
            'sample': self.paths[idx],                                          # (T, 2)
            'map':    self.map[idx],                                            # (3, H, W)
            'env':    torch.cat([self.start[idx], self.goal[idx]], dim=-1),    # (4,) — clamp only
        }
