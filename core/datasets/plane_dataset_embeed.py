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

class PlanePlanningDataSets(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str):
        self.data = np.load(dataset_path, allow_pickle=True).item()

        self.paths = torch.tensor(self.data['paths'], dtype=torch.float32)           # (N, T, action_dim)
        self.start = torch.tensor(self.data['start'], dtype=torch.float32)           # (N, obs_dim)
        self.goal = torch.tensor(self.data['goal'], dtype=torch.float32)             # (N, obs_dim)
        self.map = torch.tensor(self.data['map'], dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

        stats = get_data_stats(self.paths)
        self.paths = normalize_data(self.paths, stats)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        sample = self.paths[idx]  # (T, action_dim)
        map_cond = self.map[idx]  # (1, H, W)
        env_cond = torch.cat([self.start[idx], self.goal[idx]], dim=-1)  # (2 * obs_dim)
        return {
            "sample": sample,     # 动作轨迹
            "map": map_cond,      # 地图条件
            "env": env_cond,      # start+goal
        }



def main():
    dataset = np.load('dataset/train_data_set.npy', allow_pickle=True).item()
    stats = get_data_stats(torch.tensor(dataset['paths'], dtype=torch.float32))
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()
