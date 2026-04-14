from typing import Dict

from core.diffusion.ddpm import DDPMScheduler

from core.networks.embedUnet import ConditionalUnet1D


def build_networks_from_config(config: Dict) -> ConditionalUnet1D:
    """
    Build ConditionalUnet1D from a config dict.

    global_cond_dim = obstacle_encode_dim (ViT/CNN output)
                    + env_encode_dim      (MLP output)
    Both encoders live *inside* ConditionalUnet1D, so obs_dim is NOT added here.
    """
    action_dim = config["networks"]["unet_config"]["action_dim"]
    network_config = config["networks"]
    use_xcloud = bool(network_config.get("use_xcloud", False))
    if use_xcloud:
        obstacle_encode_dim = network_config["xcloud_encoder"]["embed_dim"]
    else:
        obstacle_encode_dim = network_config["vit_config"]["num_classes"]
    env_encode_dim = network_config["mlp_config"]["embed_dim"]
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
    else:
        raise NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")
