import torch
from core.datasets.plane_dataset import PlanePlanningDataSets

data_path = '/Users/yulinli/Desktop/Exp/diffusion_policy/test/train_data_set_flatten.npy'
PlanePlanningDataSets = PlanePlanningDataSets(data_path)


dataloader = torch.utils.data.DataLoader(
    PlanePlanningDataSets,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
)

batch = next(iter(dataloader))
print("Batch start shape:", batch.start.shape)
print("Batch goal shape:", batch.goal.shape)
print("Batch path shape:", batch.path.shape)
print("Batch map shape:", batch.map.shape)
