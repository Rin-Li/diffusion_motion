class PlaneContinuousConfig:
    """
    Config for continuous-space 2D path planning with diffusion.

    Key differences from PlaneTestEmbedConfig (grid version):
        - map encoder : CNN (64×64 3-channel map → 128-dim latent)
          ch0=obstacles, ch1=start gaussian blob, ch2=goal gaussian blob
        - global_cond_dim = 128 (CNN only, no MLP)
        - normalizer stats match continuous bounds [0, 8]
    """

    def __init__(self):
        self.horizon    = 48        # number of trajectory waypoints (= interp_points); must be divisible by 4
        self.action_dim = 2         # (x, y)

        self.network_config = {
            # --- UNet backbone ---
            'unet_config': {
                'action_dim':     2,
                'action_horizon': 48,
            },
            # --- CNN map encoder: (B, 3, 64, 64) -> (B, 128) ---
            # ch0=obstacles, ch1=start blob, ch2=goal blob
            'cnn_config': {
                'image_size':  64,
                'output_dim':  128,
                'in_channels': 3,
            },
            # no mlp_config: start/goal encoded directly in map channels
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

        self.data_generator = {
            'crop':   True,   # RRT path shortcutting
            'smooth': True,   # spline smoothing after shortcutting
        }

    def to_dict(self):
        return {
            "network_config":  self.network_config,
            "noise_scheduler": self.noise_scheduler,
            "normalizer":      self.normalizer,
            "trainer":         self.trainer,
            "horizon":         self.horizon,
            "action_dim":      self.action_dim,
            "is_CNN":          self.is_CNN,
            "data_generator":  self.data_generator,
        }
