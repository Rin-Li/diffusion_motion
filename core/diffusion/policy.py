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
                "xcloud": np.ndarray [N, 2]      — optional point cloud,
            }
            initial_action: torch.Tensor [1, pred_horizon, action_dim] — Gaussian noise seed.

        Returns:
            action_pred      : np.ndarray [pred_horizon, action_dim]  (denormalized)
            action_all_unnorm: list of np.ndarray, one per diffusion step (denormalized)
        """
        # 1. env (start + goal) — optional; used only for hard endpoint clamp
        env_np = obs_dict.get("env", None)
        if env_np is not None:
            env_cond = (
                torch.from_numpy(np.asarray(env_np, dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )  # [1, 2*obs_dim]
        else:
            env_cond = None

        networks = self.config.get("network_config", self.config.get("networks", {}))
        use_xcloud = bool(networks.get("use_xcloud", False))

        # 2. map (1-channel or 3-channel depending on mode)
        map_cond = None
        if "map" in obs_dict and obs_dict["map"] is not None:
            map_cond = (
                torch.from_numpy(np.asarray(obs_dict["map"], dtype=np.float32))
                .to(self.device)
                .unsqueeze(0)
            )  # [1, C, H, W]

        # 3. xcloud (preferred when enabled)
        xcloud = None
        if use_xcloud:
            if "xcloud" in obs_dict:
                xcloud = (
                    torch.from_numpy(np.asarray(obs_dict["xcloud"], dtype=np.float32))
                    .to(self.device)
                    .unsqueeze(0)
                )
            elif map_cond is not None:
                from utils.xcloud_utils import sample_point_cloud_from_grid
                total_points = networks.get("xcloud_encoder", {}).get("total_points", 128)
                xcloud_np = sample_point_cloud_from_grid(obs_dict["map"], total_points)
                xcloud = torch.from_numpy(xcloud_np).to(self.device).unsqueeze(0)
            else:
                raise ValueError("use_xcloud is enabled but no xcloud/map provided in obs_dict.")

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
                xcloud=xcloud,
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

        # 6b. hard-clamp endpoints to exactly match start and goal (when env is available)
        if env_np is not None:
            obs_dim = len(env_np) // 2
            action_pred[0]  = env_np[:obs_dim]   # first point = start
            action_pred[-1] = env_np[obs_dim:]   # last point  = goal

        # 6c. denormalize all intermediate steps
        action_all_unnorm = [
            self.normalizer.unnormalize_data(step, stats=self.norm_stats)
            for step in action_all
        ]

        return action_pred, action_all_unnorm

    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
