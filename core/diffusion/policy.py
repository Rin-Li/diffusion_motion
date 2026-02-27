import numpy as np
import torch

from utils.normalizer import LinearNormalizer


class PlaneDiffusionPolicy:
    """
    Wraps a ConditionalUnet1D + noise scheduler for inference.

    Conditions:
        - map_cond : obstacle map  (B, 1, H, W)  — encoded by ViT/CNN inside the network
        - env_cond : [start, goal] (B, 2*obs_dim) — encoded by MLP inside the network
    """

    def __init__(self, model, noise_scheduler, config, device):
        self.device = device
        self.net = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.norm_stats = config["normalizer"]
        self.config = config
        self.use_single_step_inference = False
        self.net.to(self.device)

    def predict_action(self, obs_dict: dict, initial_action: torch.Tensor):
        """
        Run reverse diffusion to predict a trajectory.

        Args:
            obs_dict: {
                "env": np.ndarray [2 * obs_dim]  — concatenated [start, goal],
                "map": np.ndarray [1, H, W]      — obstacle map,
            }
            initial_action: torch.Tensor [1, pred_horizon, action_dim] — Gaussian noise seed.

        Returns:
            action_pred      : np.ndarray [pred_horizon, action_dim]  (denormalized)
            action_all_unnorm: list of np.ndarray, one per diffusion step (denormalized)
        """
        # 1. env (start + goal)
        env_cond = (
            torch.from_numpy(obs_dict["env"])
            .to(self.device, dtype=torch.float32)
            .unsqueeze(0)
        )  # [1, 2*obs_dim]

        # 2. map
        map_cond = (
            torch.from_numpy(obs_dict["map"])
            .to(self.device, dtype=torch.float32)
            .unsqueeze(0)
        )  # [1, 1, H, W]

        # 3. initialize action sequence from provided noise
        naction = initial_action

        # 4. set scheduler timesteps
        self.noise_scheduler.set_timesteps(
            self.noise_scheduler.config.num_train_timesteps
        )
        timesteps = (
            self.noise_scheduler.timesteps[:1]
            if self.use_single_step_inference
            else self.noise_scheduler.timesteps
        )

        action_all = [naction.detach().cpu().numpy()[0]]

        # 5. reverse denoising loop
        for t in timesteps:
            noise_pred = self.net(
                sample=naction,
                timestep=t,
                map_cond=map_cond,
                env_cond=env_cond,
            )
            if self.use_single_step_inference:
                naction = initial_action - noise_pred
            else:
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=naction,
                ).prev_sample
            action_all.append(naction.detach().cpu().numpy()[0].copy())

        # 6. denormalize final prediction
        action_pred = naction.detach().cpu().numpy()[0]
        action_pred = self.normalizer.unnormalize_data(
            action_pred, stats=self.norm_stats
        )

        # 6b. denormalize all intermediate steps
        action_all_unnorm = [
            self.normalizer.unnormalize_data(step, stats=self.norm_stats)
            for step in action_all
        ]

        return action_pred, action_all_unnorm

    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
