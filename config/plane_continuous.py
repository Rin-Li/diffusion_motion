class PlaneContinuousConfig:
    """
    Config for continuous-space 2D path planning with diffusion.

    Key differences from PlaneTestEmbedConfig (grid version):
        - map encoder : CNN (64×64 occupancy image → 128-dim latent)
        - global_cond_dim = 128 (CNN) + 64 (MLP) = 192
        - normalizer stats match continuous bounds [0, 8]
    """

    def __init__(self):
        self.horizon    = 50        # number of trajectory waypoints (= interp_points)
        self.action_dim = 2         # (x, y)

        self.network_config = {
            # --- UNet backbone ---
            'unet_config': {
                'action_dim':     2,
                'action_horizon': 50,
            },
            # --- CNN map encoder: (B, 1, 64, 64) -> (B, 128) ---
            'cnn_config': {
                'image_size': 64,
                'output_dim': 128,
            },
            # --- MLP env encoder: [start, goal] -> (B, 64) ---
            'mlp_config': {
                'obs_dim':   2,
                'embed_dim': 64,
            },
        }

        self.noise_scheduler = {
            "type": "ddpm",
            "ddpm": {
                "num_train_timesteps": 100,
                "beta_schedule":       "squaredcos_cap_v2",
                "clip_sample":         True,
                "prediction_type":     "epsilon",
            },
        }

        # Normalizer stats for action trajectories in bounds [0, 8]
        self.normalizer = {
            'min': [0.0, 0.0],
            'max': [8.0, 8.0],
        }

        self.trainer = {
            'use_ema':    True,
            'batch_size': 256,
            'optimizer': {
                'name':          'adamw',
                'learning_rate': 1.0e-4,
                'weight_decay':  1.0e-6,
            },
            'lr_scheduler': {
                'name':             'cosine',
                'num_warmup_steps': 500,
            },
        }

        self.is_CNN = True

    def to_dict(self):
        return {
            "network_config":  self.network_config,
            "noise_scheduler": self.noise_scheduler,
            "normalizer":      self.normalizer,
            "trainer":         self.trainer,
            "horizon":         self.horizon,
            "action_dim":      self.action_dim,
            "is_CNN":          self.is_CNN,
        }
