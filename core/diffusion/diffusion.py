from typing import Dict, List

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

from core.networks.embeddUnet import ConditionalUnet1D
from utils.normalizer import LinearNormalizer


def build_networks_from_config(config: Dict):
    action_dim = config["networks"]["unet_config"]["action_dim"]
    obs_dim = config["networks"]["unet_config"]["action_horizon"]
    obstacle_encode_dim = config["networks"]["vit_config"]["num_classes"]
    env_encode_dim = config["networks"]["mlp_config"]["embed_dim"]
    network_config = config["networks"]
    is_cnn = config["networks"]["is_cnn"]
    return ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * action_dim + obstacle_encode_dim + env_encode_dim, network_config=network_config, is_cnn=is_cnn)


def build_noise_scheduler_from_config(config: Dict):
    type_noise_scheduler = config["noise_scheduler"]["type"]
    if type_noise_scheduler.lower() == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddpm"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddpm"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddpm"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddpm"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddim"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddim"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddim"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddim"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "dpmsolver":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["noise_scheduler"]["dpmsolver"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["dpmsolver"]["beta_schedule"],
            prediction_type=config["noise_scheduler"]["dpmsolver"]["prediction_type"],
            use_karras_sigmas=config["noise_scheduler"]["dpmsolver"]["use_karras_sigmas"],
        )
    else:
        raise NotImplementedError


class PlaneDiffusionPolicy:
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler,
        config,
        device,
    ):
        self.device = device
        self.net = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.norm_stats = config["normalizer"]
        self.net.to(self.device)
        self.config = config
        self.use_single_step_inference = False


    def predict_action(self, obs_dict: dict, initial_action):
        """
        obs_dict 结构：
        - "sample": [obs_horizon, obs_dim]
        - "env": [2 * obs_dim] 的拼接向量
        - "map": [1, 8, 8]
        """

        # 1. 动作
        obs_seq = obs_dict["sample"]  # shape: [obs_horizon, obs_dim]
        nobs = self.normalizer.normalize_data(obs_seq, stats=self.norm_stats)
        nobs = nobs.flatten()
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

        # 2. env 
        env_cond = torch.from_numpy(obs_dict["env"]).to(self.device, dtype=torch.float32).unsqueeze(0)  # [1, 2*obs_dim]

        # 3. map 
        map_cond = torch.from_numpy(obs_dict["map"]).to(self.device, dtype=torch.float32).unsqueeze(0)  # [1, 1, 8, 8]

        # 4. 初始化 action 序列为高斯噪声
        noisy_action = initial_action
        naction = noisy_action

        # 5. 调度器时间步设置
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = (
            self.noise_scheduler.timesteps[:1] if self.use_single_step_inference else self.noise_scheduler.timesteps
        )
        
        
        action_all = [naction.detach().cpu().numpy()[0]]  
        # 6. 执行反向去噪
        for t in timesteps:
            noise_pred = self.net(
                sample=naction,
                timestep=t,
                map_cond=map_cond,
                env_cond=env_cond,
            )
            if self.use_single_step_inference:
                naction = noisy_action - noise_pred
            else:
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=naction
                ).prev_sample
                
            action_all.append(naction.detach().cpu().numpy()[0].copy())

        # 7. 反归一化
        action_pred = naction.detach().cpu().numpy()[0]
        action_pred = self.normalizer.unnormalize_data(action_pred, stats=self.norm_stats)
        
            # 反归一化所有轨迹步骤
        action_all_unnorm = []
        for action_step in action_all:
            action_step_unnorm = self.normalizer.unnormalize_data(action_step, stats=self.norm_stats)
            action_all_unnorm.append(action_step_unnorm)

        return action_pred, action_all_unnorm  # shape: [pred_horizon, action_dim]


    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)


