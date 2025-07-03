class PlaneTestEmbedConfig:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'unet_config': {
                'action_dim': 2,
                'action_horizon': 32,
            },
            'mlp_config': {
                'obs_dim': 2,
                'embed_dim': 64
            },
            'vit_config': {
                'image_size': 8,
                'patch_size': 4,
                'channels': 1,
                'num_classes': 32,
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'dropout': 0.1,
                'emb_dropout': 0.1
            }
        }

        self.noise_scheduler = {
            "type": "ddpm",
            "ddpm": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon",
            },
            "ddim": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon",
            },
            "dpmsolver": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "prediction_type": "epsilon",
                "use_karras_sigmas": True,
            },
        }

        self.normalizer = {
            'min': [0.0015, 0.0014],
            'max': [7.9972, 7.9981]
        }

        self.trainer = {
            'use_ema': True,
            'batch_size': 256,
            'optimizer': {
                'name': "adamw",
                'learning_rate': 1.0e-4,
                'weight_decay': 1.0e-6
            },
            'lr_scheduler': {
                'name': "cosine",
                'num_warmup_steps': 500
            }
        }
        
        self.is_CNN = True

    def to_dict(self):
        return {
            "network_config": self.network_config,
            "noise_scheduler": self.noise_scheduler,
            "normalizer": self.normalizer,
            "trainer": self.trainer,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "is_CNN": self.is_CNN
        }
