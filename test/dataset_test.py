import torch
from core.datasets.plane_dataset_embed import PlanePlanningDataSets


def main():
    data_path = "/Users/yulinli/diffusion_motion/test/train_data_set_flatten.npy"
    dataset = PlanePlanningDataSets(data_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )

    batch = next(iter(dataloader))
    print("Batch sample:", batch["sample"].shape)
    print("Batch map:", batch["map"].shape)
    print("Batch env:", batch["env"].shape)


if __name__ == "__main__":
    main()
