import numpy as np
import torch

from utils.normalizer import LinearNormalizer


class FlowMatchingPolicy:

    def __init__(self, model, flow_matching, config, device):
        self.device = device
        self.net = model
        self.fm = flow_matching
        self.normalizer = LinearNormalizer()
        self.norm_stats = config["normalizer"]
        self.config = config
        self.net.to(self.device)

    def predict_action(self, obs_dict: dict, x0: torch.Tensor):
        env_cond = self._prep_env(obs_dict)
        map_cond = self._prep_map(obs_dict)

        xt = x0.to(self.device)
        timesteps = self.fm.get_inference_timesteps()
        dt = 1.0 / self.fm.num_infer_steps

        for t in timesteps[:-1]:
            t_batch = t.expand(xt.shape[0]).to(self.device)  # (B,)
            with torch.no_grad():
                v_pred = self.net(
                    sample=xt,
                    timestep=t_batch,
                    map_cond=map_cond,
                    env_cond=env_cond,
                )
            xt = self.fm.euler_step(xt, v_pred, dt)

        action_pred = xt.detach().cpu().numpy()[0]
        action_pred = self.normalizer.unnormalize_data(action_pred, stats=self.norm_stats)

        env_np = obs_dict.get("env")
        if env_np is not None:
            obs_dim = len(env_np) // 2
            action_pred[0]  = env_np[:obs_dim]
            action_pred[-1] = env_np[obs_dim:]

        return action_pred

    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def _prep_env(self, obs_dict) -> torch.Tensor | None:
        env_np = obs_dict.get("env")
        if env_np is None:
            return None
        return torch.from_numpy(np.asarray(env_np, dtype=np.float32)).to(self.device).unsqueeze(0)  # (1, 2*obs_dim)

    def _prep_map(self, obs_dict) -> torch.Tensor:
        return torch.from_numpy(np.asarray(obs_dict["map"], dtype=np.float32)).to(self.device).unsqueeze(0)  # (1, C, H, W)
