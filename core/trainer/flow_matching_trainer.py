import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from typing import Optional

from core.flow_matching.fm import FlowMatching
from utils.xcloud_utils import sample_point_cloud_from_grid


class FlowMatchingTrainer:
    def __init__(
        self,
        net: nn.Module,
        dataset: torch.utils.data.Dataset,
        config,
        device: Optional[str] = None,
        wandb_run=None,
    ):
        self.net = net
        self.config = config.to_dict()
        self.fm = FlowMatching(
            num_infer_steps=self.config['flow_matching']['num_infer_steps']
        )
        self.device = 'cuda' if torch.cuda.is_available() and device is None else device
        self.wandb_run = wandb_run

        self.net.to(self.device)

        opt_cfg = self.config['trainer']['optimizer']
        self.optimizer = torch.optim.AdamW(
            params=self.net.parameters(),
            lr=opt_cfg['learning_rate'],
            weight_decay=opt_cfg['weight_decay'],
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['trainer']['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )

        self.use_ema = self.config['trainer']['use_ema']
        self.ema = EMAModel(parameters=self.net.parameters(), power=0.75) if self.use_ema else None

        self._use_amp = self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

    def prepare_inputs(self, batch):
        # action: (B, T, D)   map_cond: (B, C, H, W)
        action   = batch['sample'].to(self.device, dtype=torch.float32)
        map_cond = batch['map'].to(self.device, dtype=torch.float32)
        batch_size = action.shape[0]

        xcloud = None
        networks = self.config.get("network_config", self.config.get("networks", {}))
        use_xcloud = bool(networks.get("use_xcloud", False))
        if use_xcloud:
            if "xcloud" in batch:
                xcloud = batch["xcloud"].to(self.device, dtype=torch.float32)
            else:
                total_points = networks.get("xcloud_encoder", {}).get("total_points", 128)
                xcloud_list = []
                map_np = map_cond.detach().cpu().numpy()
                for i in range(map_np.shape[0]):
                    xcloud_np = sample_point_cloud_from_grid(map_np[i], total_points)
                    xcloud_list.append(xcloud_np)
                xcloud = torch.from_numpy(np.stack(xcloud_list, axis=0)).to(self.device, dtype=torch.float32)

        return map_cond, action, batch_size, xcloud

    def optimization_step(self, action, map_cond, batch_size, xcloud=None):
        self.optimizer.zero_grad()

        x1 = action                                                   # (B, T, D)
        x0 = torch.randn_like(x1)                                     # (B, T, D)
        t  = self.fm.sample_timesteps(batch_size, self.device)        # (B,)
        t_exp = t[:, None, None]                                      # (B, 1, 1)

        xt       = self.fm.interpolate(x0, x1, t_exp)                # (B, T, D)
        v_target = self.fm.get_velocity_target(x0, x1)               # (B, T, D)

        with torch.amp.autocast('cuda', enabled=self._use_amp):
            v_pred = self.net(sample=xt, timestep=t, map_cond=map_cond, xcloud=xcloud)
            loss = F.mse_loss(v_pred, v_target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        if self.use_ema:
            self.ema.step(self.net.parameters())

        return loss

    def train(self, num_epochs: int, save_ckpt_epoch: int = None,
              eval_fn=None, eval_every: int = 100):
        if save_ckpt_epoch is None:
            save_ckpt_epoch = num_epochs

        lr_cfg = self.config['trainer']['lr_scheduler']
        self.lr_scheduler = get_scheduler(
            name=lr_cfg['name'],
            optimizer=self.optimizer,
            num_warmup_steps=lr_cfg['num_warmup_steps'],
            num_training_steps=len(self.dataloader) * num_epochs,
        )

        trn_loss = []
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = []
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        map_cond, action, B, xcloud = self.prepare_inputs(nbatch)
                        loss = self.optimization_step(action, map_cond, B, xcloud)
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

                mean_loss = np.mean(epoch_loss)
                tglobal.set_postfix(loss=mean_loss)
                trn_loss.append(mean_loss)
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"train/epoch_loss": mean_loss},
                        step=(epoch_idx + 1) * len(self.dataloader),
                    )

                if (epoch_idx + 1) % save_ckpt_epoch == 0:
                    self.save_checkpoint(f'ckpt/fm_ep{epoch_idx + 1}.ckpt')

                if eval_fn is not None and (epoch_idx + 1) % eval_every == 0:
                    self.net.eval()
                    with torch.no_grad():
                        eval_fn(self.net, epoch_idx + 1)
                    self.net.train()

        return trn_loss

    def save_checkpoint(self, path: str):
        save_model = self.net
        if self.use_ema:
            self.ema.copy_to(save_model.parameters())
        torch.save(save_model.state_dict(), path)
