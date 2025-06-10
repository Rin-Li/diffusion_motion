import torch
import numpy as np
from collections import namedtuple

# Data : [n, 30, 2]
def get_data_stats(data: torch.Tensor):
    _, _, d = data.shape  # n: batch size, T: sequence length, d: dimension
    flat = data.reshape(-1, d)  # Flatten to [n*T, d]

    data_min = flat.min(dim=0).values  # shape: (d,)
    data_max = flat.max(dim=0).values  # shape: (d,)

    stats = {
        "min": data_min,
        "max": data_max,
    }
    return stats
def normalize_data(data: torch.Tensor, stats):
    n, T, d = data.shape  
    flat = data.reshape(-1, d) 


    eps = 1e-8
    norm = (flat - stats['min']) / (stats['max'] - stats['min'] + eps)  # → [0, 1]
    norm = norm * 2 - 1  # → [-1, 1]

    norm = norm.reshape(n, T, d)
    return norm

def denormalize_data(data: torch.Tensor, stats):
    n, T, d = data.shape  
    flat = data.reshape(-1, d)  

    eps = 1e-8
    denorm = (flat + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']  # → [min, max]

    denorm = denorm.reshape(n, T, d)
    return denorm

Batch = namedtuple('Batch', 'start goal path map')
class PlanePlanningDataSets(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str):
        self.data = np.load(dataset_path, allow_pickle=True).item()
        self.paths = torch.tensor(self.data['paths'], dtype=torch.float32)
        self.start = torch.tensor(self.data['start'], dtype=torch.float32)
        self.goal = torch.tensor(self.data['goal'], dtype= torch.float32)
        self.obstacles = torch.tensor(self.data['map'], dtype=torch.float32)
        stats = get_data_stats(self.paths)
        self.paths = normalize_data(self.paths, stats)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bacth = Batch(self.start[idx],
                      self.goal[idx],
                      self.paths[idx],
                      self.obstacles[idx])
        return bacth

