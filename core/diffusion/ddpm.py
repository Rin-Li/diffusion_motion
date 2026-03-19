"""
Minimal DDPMScheduler extracted from diffusers for easier debugging.

Only keeps what this project actually uses:
  - beta_schedule : squaredcos_cap_v2  (cosine)
  - prediction_type : epsilon
  - variance_type  : fixed_small  (posterior variance, no noise at t=0)
  - clip_sample    : True / False

Reference: https://arxiv.org/pdf/2006.11239.pdf
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------

def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """
    Cosine schedule (squaredcos_cap_v2).

    Builds beta_t such that alpha_bar_t follows a cosine curve.
    See Nichol & Dhariwal 2021, Eq. (17).
    """
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1.0 - alpha_bar_t2 / alpha_bar_t1, max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Output / config containers
# ---------------------------------------------------------------------------

@dataclass
class DDPMSchedulerOutput:
    prev_sample: torch.Tensor           # x_{t-1}
    pred_original_sample: torch.Tensor  # predicted x_0


@dataclass
class _Config:
    """Mimics diffusers FrozenDict so scheduler.config.xxx still works."""
    num_train_timesteps: int
    beta_schedule: str
    clip_sample: bool
    prediction_type: str


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """
    DDPM reverse-diffusion scheduler (Ho et al. 2020).

    Args:
        num_train_timesteps : total diffusion steps used during training
        beta_schedule       : only "squaredcos_cap_v2" is supported here
        clip_sample         : whether to clamp pred_x0 to [-1, 1]
        prediction_type     : only "epsilon" is supported here
        beta_start / beta_end : ignored for cosine schedule, kept for API compat
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        beta_start: float = 0.0001,  # unused for cosine schedule
        beta_end: float = 0.02,      # unused for cosine schedule
    ):
        self.config = _Config(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )

        # --- build beta / alpha tables ---
        if beta_schedule == "squaredcos_cap_v2":
            self.betas = _betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError(f"beta_schedule '{beta_schedule}' not supported.")

        self.alphas = 1.0 - self.betas                           # alpha_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_bar_t

        # Default timestep sequence (overwritten by set_timesteps)
        self.num_inference_steps: Optional[int] = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy()
        )

    # -----------------------------------------------------------------------
    # set_timesteps  –  called once before the inference loop
    # -----------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int):
        """
        Build the descending timestep sequence for inference.

        Uses "leading" spacing (same default as diffusers):
          step_ratio = T // num_inference_steps
          timesteps  = [0, step_ratio, 2*step_ratio, ...] reversed
        """
        self.num_inference_steps = num_inference_steps
        T = self.config.num_train_timesteps
        step_ratio = T // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    # -----------------------------------------------------------------------
    # previous_timestep  –  t-1 in the inference sequence
    # -----------------------------------------------------------------------

    def _previous_timestep(self, t: int) -> int:
        """Return the previous timestep in self.timesteps, or -1 at the end."""
        idx = (self.timesteps == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0 or idx[0] == len(self.timesteps) - 1:
            return -1
        return int(self.timesteps[idx[0] + 1])

    # -----------------------------------------------------------------------
    # add_noise  –  forward process  q(x_t | x_0)
    # -----------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.Tensor,  # x_0   (B, T, D)
        noise: torch.Tensor,              # eps   (B, T, D)
        timesteps: torch.Tensor,          # t     (B,)
    ) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)

        sqrt_alpha_bar = alphas_cumprod[timesteps] ** 0.5          # (B,)
        sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5    # (B,)

        # broadcast over (T, D) dims
        while sqrt_alpha_bar.dim() < original_samples.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha_bar * original_samples + sqrt_one_minus * noise

    # -----------------------------------------------------------------------
    # step  –  one reverse step  p(x_{t-1} | x_t)
    # -----------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,  # predicted noise eps_theta(x_t, t)
        timestep: int,               # current t
        sample: torch.Tensor,        # x_t
    ) -> DDPMSchedulerOutput:
        """
        DDPM reverse step (fixed_small variance, epsilon prediction).

        1. Predict x_0 from eps prediction.
        2. Compute posterior mean  mu_t(x_t, x_0).
        3. Add posterior noise (zero at t=0).
        """
        t = int(timestep)
        prev_t = self._previous_timestep(t)

        # alpha quantities for t and t-1
        alpha_bar_t      = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        beta_t      = 1.0 - alpha_bar_t
        beta_t_prev = 1.0 - alpha_bar_t_prev
        alpha_t     = alpha_bar_t / alpha_bar_t_prev   # = 1 - current_beta_t
        current_beta_t = 1.0 - alpha_t

        # --- step 1: reconstruct x_0 from predicted noise ---
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        if self.config.prediction_type == "epsilon":
            pred_x0 = (sample - beta_t**0.5 * model_output) / alpha_bar_t**0.5
        else:
            raise NotImplementedError(f"prediction_type '{self.config.prediction_type}' not supported.")

        if self.config.clip_sample:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # --- step 2: posterior mean  mu_t  (formula 7 in DDPM paper) ---
        # mu_t = sqrt(alpha_bar_{t-1}) * current_beta_t / beta_t * x_0
        #       + sqrt(alpha_t)        * beta_t_prev    / beta_t * x_t
        coeff_x0 = alpha_bar_t_prev**0.5 * current_beta_t / beta_t
        coeff_xt = alpha_t**0.5          * beta_t_prev    / beta_t
        mu_t = coeff_x0 * pred_x0 + coeff_xt * sample

        # --- step 3: posterior variance  sigma_t^2  (fixed_small) ---
        # sigma_t^2 = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * current_beta_t
        variance = (beta_t_prev / beta_t) * current_beta_t
        variance = variance.clamp(min=1e-20)

        # sample: x_{t-1} = mu_t  +  sigma_t * eps   (no noise at t=0)
        noise = torch.randn_like(sample) if t > 0 else torch.zeros_like(sample)
        prev_sample = mu_t + variance**0.5 * noise

        return DDPMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
