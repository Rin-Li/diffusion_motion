class PlaneFMConfig:
    def __init__(self):
        self.horizon    = 48
        self.action_dim = 2

        self.network_config = {
            'unet_config': {
                'action_dim':     2,
                'action_horizon': 48,
            },
            'cnn_config': {
                'image_size':  64,
                'output_dim':  128,
                'in_channels': 3,
            },
            # Point-cloud encoder is optional. If use_xcloud=True, map encoders (CNN/ViT) are ignored.
            'xcloud_encoder': {
                'point_dim': 2,
                'hidden_dims': [64, 128],
                'embed_dim': 64,
                'total_points': 128,
            },
            'use_xcloud': False,
        }

        self.flow_matching = {
            'num_infer_steps': 10,
        }

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
            'smooth': False,   # spline smoothing after shortcutting
        }

    def to_dict(self):
        return {
            'network_config':  self.network_config,
            'flow_matching':   self.flow_matching,
            'normalizer':      self.normalizer,
            'trainer':         self.trainer,
            'horizon':         self.horizon,
            'action_dim':      self.action_dim,
            'is_CNN':          self.is_CNN,
            'data_generator':  self.data_generator,
        }
