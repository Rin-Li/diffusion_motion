from typing import Dict

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from core.networks.embeddUnet import ConditionalUnet1D


def build_networks_from_config(config: Dict) -> ConditionalUnet1D:
    """
    Build ConditionalUnet1D from a config dict.

    global_cond_dim = obstacle_encode_dim (ViT/CNN output)
                    + env_encode_dim      (MLP output)
    Both encoders live *inside* ConditionalUnet1D, so obs_dim is NOT added here.
    """
    action_dim = config["networks"]["unet_config"]["action_dim"]
    obstacle_encode_dim = config["networks"]["vit_config"]["num_classes"]
    env_encode_dim = config["networks"]["mlp_config"]["embed_dim"]
    network_config = config["networks"]
    # Support both key spellings: "is_CNN" (top-level) and "is_cnn" (nested)
    is_cnn = config.get("is_CNN", config.get("is_cnn", False))
    return ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obstacle_encode_dim + env_encode_dim,
        network_config=network_config,
        is_cnn=is_cnn,
    )


def build_noise_scheduler_from_config(config: Dict):
    """Build a diffusers noise scheduler from a config dict."""
    scheduler_type = config["noise_scheduler"]["type"].lower()
    if scheduler_type == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddpm"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddpm"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddpm"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddpm"]["prediction_type"],
        )
    elif scheduler_type == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddim"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddim"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddim"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddim"]["prediction_type"],
        )
    elif scheduler_type == "dpmsolver":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["noise_scheduler"]["dpmsolver"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["dpmsolver"]["beta_schedule"],
            prediction_type=config["noise_scheduler"]["dpmsolver"]["prediction_type"],
            use_karras_sigmas=config["noise_scheduler"]["dpmsolver"]["use_karras_sigmas"],
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")
