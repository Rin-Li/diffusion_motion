import numpy as np
from typing import Iterable, List, Tuple, Optional


Rect = Tuple[float, float, float, float]  # (x0, y0, w, h)


def _rect_to_bounds(rect: Rect) -> Tuple[float, float, float, float]:
    x0, y0, w, h = rect
    return x0, y0, x0 + w, y0 + h


def sample_rect_edge_points(rect: Rect, n: int) -> np.ndarray:
    xmin, ymin, xmax, ymax = _rect_to_bounds(rect)
    pts = []
    for _ in range(n):
        t = np.random.rand()
        edge = np.random.randint(0, 4)
        if edge == 0:
            pts.append((xmin + t * (xmax - xmin), ymin))
        elif edge == 1:
            pts.append((xmax, ymin + t * (ymax - ymin)))
        elif edge == 2:
            pts.append((xmax - t * (xmax - xmin), ymax))
        else:
            pts.append((xmin, ymax - t * (ymax - ymin)))
    return np.array(pts, dtype=np.float32)


def sample_point_cloud_from_rects(
    rects: Iterable[Rect],
    total_points: int,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> np.ndarray:
    rects_list: List[Rect] = list(rects)
    if not rects_list:
        return np.zeros((total_points, 2), dtype=np.float32)

    per = max(1, total_points // len(rects_list))
    pts = []
    for r in rects_list:
        pts.append(sample_rect_edge_points(r, per))
    cloud = np.concatenate(pts, axis=0)

    # pad or trim
    if cloud.shape[0] < total_points:
        pad = total_points - cloud.shape[0]
        if bounds is None:
            low = np.array([0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([bounds[0][0], bounds[1][0]], dtype=np.float32)
            high = np.array([bounds[0][1], bounds[1][1]], dtype=np.float32)
        extra = np.random.rand(pad, 2).astype(np.float32)
        extra = low + (high - low) * extra
        cloud = np.concatenate([cloud, extra], axis=0)
    if cloud.shape[0] > total_points:
        idx = np.random.choice(cloud.shape[0], total_points, replace=False)
        cloud = cloud[idx]
    return cloud


def sample_point_cloud_from_grid(
    grid: np.ndarray,
    total_points: int,
    *,
    cell_size: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim == 4:  # [B, C, H, W] -> take first
        grid = grid[0, 0]
    elif grid.ndim == 3:  # [C, H, W]
        grid = grid[0]

    idxs = np.argwhere(grid > 0)
    if idxs.size == 0:
        return np.zeros((total_points, 2), dtype=np.float32)

    # convert grid indices to world coordinates (cell centers)
    pts = []
    for ix, iy in idxs:
        x = origin[0] + (ix + 0.5) * cell_size
        y = origin[1] + (iy + 0.5) * cell_size
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    # sample or pad to total_points
    if pts.shape[0] >= total_points:
        idx = np.random.choice(pts.shape[0], total_points, replace=False)
        return pts[idx]

    pad = total_points - pts.shape[0]
    h, w = grid.shape
    low = np.array([origin[0], origin[1]], dtype=np.float32)
    high = np.array([origin[0] + h * cell_size, origin[1] + w * cell_size], dtype=np.float32)
    extra = np.random.rand(pad, 2).astype(np.float32)
    extra = low + (high - low) * extra
    return np.concatenate([pts, extra], axis=0)
