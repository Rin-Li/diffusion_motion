class PlaneTestConfig:
    def __init__(self):
        # Dataset
        self.dataset_path = "dataset/train_data_set.npy"
        
        # Model architecture
        self.horizon = 32
        self.transition_dim = 2  # x, y coordinates
        self.unet_dim = 64
        self.dim_mults = (1, 2, 4, 8)
        self.wall_embed_dim = 32
        
        # Diffusion parameters
        self.n_timesteps = 100
        self.loss_type = 'l2'
        self.clip_denoised = True
        self.predict_epsilon = True
        self.loss_discount = 1.0
        self.condition_guidance_w = 2.0
        self.observation_dim = 2
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        self.min_lr = 1e-6
        self.grad_clip_norm = 1.0
        
        # Logging and saving
        self.use_wandb = True
        self.wandb_project = "plane_diffusion"
        self.run_name = "plane_planning_v1"
        self.log_freq = 10
        self.val_freq = 50
        self.sample_freq = 100
        self.save_freq = 200
        self.save_dir = "./checkpoints"
        
        self.action_dim = 0
        
        # System
        self.device = "cuda"
        self.num_workers = 4
        
        # Network specific configs
        self.network_config = {
            'cat_t_w': False,
            'resblock_ksize': 5,
            'use_downup_sample': True,
            'energy_mode': False,
            'concept_drop_prob': 0.1,
            'vit_config': {
                'image_size': 8,  
                'patch_size': 4,
                'channels': 1,
                'num_classes': self.wall_embed_dim,
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'dropout': 0.1,
                'emb_dropout': 0.1
            }
        }
        
        self.diff_config = {
            'ddim_steps': 10,
            'ddim_set_alpha_to_one': True,
            'is_dyn_env': False,
            'train_apply_condition': True,
            'set_cond_noise_to_0': False,
            'debug_mode': False,
            'manual_loss_weights': {}
        }