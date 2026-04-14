from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, Optional

from core.diffusion.diffusion import build_noise_scheduler_from_config
from core.datasets.plane_dataset_embed import PlanePlanningDataSets
from utils.xcloud_utils import sample_point_cloud_from_grid


def build_dataloader_from_dataset_and_config(config: Dict, dataset: torch.utils.data.Dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )


class PlaneDiffusionTrainer:
    def __init__(
        self,
        net: nn.Module,
        dataset: PlanePlanningDataSets,
        config: Dict,
        device: Optional[str] = None,
        wandb_run=None,
    ):
        self.net = net
        self.config = config.to_dict()
        self.noise_scheduler = build_noise_scheduler_from_config(self.config)
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() and device is None else device
        self.wandb_run = wandb_run

        self.net.to(self.device)

        # build optimizer
        if self.config["trainer"]["optimizer"]["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.net.parameters(),
                lr=self.config["trainer"]["optimizer"]["learning_rate"],
                weight_decay=self.config["trainer"]["optimizer"]["weight_decay"],
            )
        else:
            raise NotImplementedError

        # build dataset
        self.dataloader = build_dataloader_from_dataset_and_config(self.config, dataset)

        # set EMA
        self.use_ema = self.config["trainer"]["use_ema"]
        self.ema = EMAModel(parameters=self.net.parameters(), power=0.75) if self.use_ema else None

        # mixed precision — enabled only on CUDA, gracefully disabled elsewhere
        self._use_amp = self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

    def prepare_inputs(self, batch):
        """
        - sample: target action trajectory (B, T, action_dim)
        - map: map condition (B, C, H, W)  — 1-channel or 3-channel depending on dataset
        """
        action   = batch["sample"].to(self.device, dtype=torch.float32)   # target output
        map_cond = batch["map"].to(self.device, dtype=torch.float32)       # condition: map image
        batch_size = action.shape[0]

        env_cond = batch.get("env", None)
        if env_cond is not None:
            env_cond = env_cond.to(self.device, dtype=torch.float32)

        xcloud = None
        networks = self.config.get("network_config", self.config.get("networks", {}))
        use_xcloud = bool(networks.get("use_xcloud", False))
        if use_xcloud:
            if "xcloud" in batch:
                xcloud = batch["xcloud"].to(self.device, dtype=torch.float32)
            else:
                total_points = networks.get("xcloud_encoder", {}).get("total_points", 128)
                # build xcloud from occupancy grid
                xcloud_list = []
                map_np = map_cond.detach().cpu().numpy()
                for i in range(map_np.shape[0]):
                    xcloud_np = sample_point_cloud_from_grid(map_np[i], total_points)
                    xcloud_list.append(xcloud_np)
                xcloud = torch.from_numpy(np.stack(xcloud_list, axis=0)).to(self.device, dtype=torch.float32)

        return map_cond, env_cond, action, batch_size, xcloud

    def optimization_step(self, action, map_cond, env_cond, batch_size, xcloud=None):
        # sample noise to add to actions
        noise = torch.randn(action.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        # forward pass under mixed precision
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            noise_pred = self.net(noisy_actions, timesteps, map_cond, env_cond, xcloud)
            loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize with gradient scaler (no-op when AMP disabled)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        if self.use_ema:
            self.ema.step(self.net.parameters())

        return loss

    def train(self, num_epochs: int, save_ckpt_epoch: int = None,
              eval_fn=None, eval_every: int = 100):
        if save_ckpt_epoch is None:
            save_ckpt_epoch = num_epochs

        # set learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name=self.config["trainer"]["lr_scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["trainer"]["lr_scheduler"]["num_warmup_steps"],
            num_training_steps=len(self.dataloader) * num_epochs,
        )

        # training loop
        trn_loss = []
        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc="Batch", leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        map_cond, env_cond, action, B, xcloud = self.prepare_inputs(nbatch)
                        loss = self.optimization_step(action, map_cond, env_cond, B, xcloud)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                        if self.wandb_run is not None:
                            lr = self.optimizer.param_groups[0]["lr"]
                            global_step = epoch_idx * len(self.dataloader) + batch_idx
                            self.wandb_run.log(
                                {"train/loss": loss_cpu, "train/lr": lr},
                                step=global_step,
                            )

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                trn_loss.append(np.mean(epoch_loss))
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"train/epoch_loss": trn_loss[-1]},
                        step=(epoch_idx + 1) * len(self.dataloader),
                    )

                # save intermediate ckpt
                if (epoch_idx + 1) % save_ckpt_epoch == 0:
                    self.save_checkpoint(path=f"ckpt/ckpt_ep{epoch_idx + 1}.ckpt")

                # eval callback
                if eval_fn is not None and (epoch_idx + 1) % eval_every == 0:
                    self.net.eval()
                    with torch.no_grad():
                        eval_fn(self.net, epoch_idx + 1)
                    self.net.train()

        return trn_loss

    def save_checkpoint(self, path: str):
        save_model = self.net
        if self.config["trainer"]["use_ema"]:
            self.ema.copy_to(save_model.parameters())
        torch.save(save_model.state_dict(), path)
