import torch
from core.datasets.plane_dataset_embeed import PlanePlanningDataSets

data_path = '/Users/yulinli/diffusion_motion/test/train_data_set_flatten.npy'
PlanePlanningDataSets = PlanePlanningDataSets(data_path)


dataloader = torch.utils.data.DataLoader(
    PlanePlanningDataSets,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
)

batch = next(iter(dataloader))
print("Batch sample:", batch["sample"].shape)
print("Batch map:", batch["map"].shape)
print("Batch env:", batch["env"].shape)
