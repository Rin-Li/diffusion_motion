import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from collections import defaultdict
from config import PlaneTestConfig

from core.networks.diffusion import GaussianDiffusionPB
from core.datasets.plane_dataset import PlanePlanningDataSets
from core.networks.unet import TemporalUnet_WCond

class PlaneDiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize dataset
        self.dataset = PlanePlanningDataSets(
            dataset_path=config.dataset_path,
            data_config=config.data_config
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Initialize model components
        self.unet = TemporalUnet_WCond(
            horizon=config.horizon,
            transition_dim=config.observation_dim,  # Only state, no action
            cond_dim=config.observation_dim,  # Start/goal conditioning
            dim=config.unet_dim,
            dim_mults=config.dim_mults,
            wall_embed_dim=config.wall_embed_dim,
            network_config=config.network_config
        ).to(self.device)
        
        self.diffusion = GaussianDiffusionPB(
            model=self.unet,
            horizon=config.horizon,
            observation_dim=config.observation_dim,
            action_dim=0,  # No actions in this planning task
            n_timesteps=config.n_timesteps,
            loss_type=config.loss_type,
            clip_denoised=config.clip_denoised,
            predict_epsilon=config.predict_epsilon,
            loss_discount=config.loss_discount,
            condition_guidance_w=config.condition_guidance_w,
            diff_config=config.diff_config
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        # Logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=config.run_name
            )
    
    def prepare_batch(self, batch):
        """Prepare batch for training"""
        start = batch.start.to(self.device)  # [B, 2]
        goal = batch.goal.to(self.device)    # [B, 2]
        paths = batch.path.to(self.device)   # [B, horizon, 2]
        obstacles = batch.obstacles.to(self.device)  # [B, grid_size, grid_size]
        
        # Prepare conditioning - start and goal positions
        cond = {
            0: start,           # Condition at timestep 0 (start)
            self.config.horizon - 1: goal  # Condition at final timestep (goal)
        }
        
        # Prepare wall locations (obstacles)
        # Assuming obstacles is a 2D grid, reshape for wall encoder
        batch_size = obstacles.shape[0]
        wall_locations = obstacles.view(batch_size, 1, *obstacles.shape[1:])
        
        return paths, cond, wall_locations
    
    def train_step(self, batch):
        """Single training step"""
        self.diffusion.train()
        
        # Prepare data
        trajectories, cond, wall_locations = self.prepare_batch(batch)
        
        # Forward pass
        loss, info = self.diffusion.loss(
            x=trajectories,  # Ground truth trajectories
            cond=cond,       # Start/goal conditioning
            wall_locations=wall_locations  # Obstacle information
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion.parameters(),
                self.config.grad_clip_norm
            )
        
        self.optimizer.step()
        
        return loss.item(), info
    
    def validate(self):
        """Validation step"""
        self.diffusion.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Validating"):
                trajectories, cond, wall_locations = self.prepare_batch(batch)
                loss, _ = self.diffusion.loss(trajectories, cond, wall_locations)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def sample_trajectories(self, num_samples=4):
        """Sample trajectories for visualization"""
        self.diffusion.eval()
        
        with torch.no_grad():
            # Get a batch for conditioning
            batch = next(iter(self.dataloader))
            trajectories, cond, wall_locations = self.prepare_batch(batch)
            
            # Take only first few samples
            cond = {k: v[:num_samples] for k, v in cond.items()}
            wall_locations = wall_locations[:num_samples]
            
            # Sample trajectories
            samples = self.diffusion.conditional_sample(
                cond=cond,
                walls_loc=wall_locations,
                use_ddim=True,
                verbose=False
            )
            
            return samples, trajectories[:num_samples], cond, wall_locations
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.diffusion.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                loss, info = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{np.mean(epoch_losses):.4f}"
                })
                
                # Log to wandb
                if self.config.use_wandb and self.step % self.config.log_freq == 0:
                    log_dict = {
                        'train/loss': loss,
                        'train/epoch': epoch,
                        'train/step': self.step,
                        'train/lr': self.optimizer.param_groups[0]['lr']
                    }
                    if 'diffuse_loss' in info:
                        log_dict['train/diffuse_loss'] = info['diffuse_loss'].item()
                    wandb.log(log_dict, step=self.step)
            
            # Validation
            if epoch % self.config.val_freq == 0:
                val_loss = self.validate()
                print(f"Validation loss: {val_loss:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/epoch': epoch
                    }, step=self.step)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config.save_dir, 'best_model.pt')
                    )
            
            # Sample trajectories for visualization
            if epoch % self.config.sample_freq == 0:
                try:
                    samples, gt_trajs, cond, walls = self.sample_trajectories()
                    print(f"Sampled trajectories shape: {samples.shape}")
                    # Here you could add visualization code
                except Exception as e:
                    print(f"Sampling failed: {e}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.config.save_dir, 'final_model.pt')
        )




# Usage example
if __name__ == "__main__":
    config = PlaneTestConfig()
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = PlaneDiffusionTrainer(config)
    
    # Start training
    trainer.train()