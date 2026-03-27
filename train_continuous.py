"""
Continuous-space diffusion trajectory training script.

Pipeline:
  1. Data generation   — RRT* in continuous 2D space (multiprocessing), skipped if data exists
  2. Dataset loading & normalization
  3. Model construction — ConditionalUnet1D with CNN map encoder
  4. Training           — DDPM with FiLM conditioning, with periodic visual evaluation
  5. Inference test     — shape sanity check
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import torch
from core.datasets.plane_dataset_3ch import _gaussian_blob

# Config
OUTFILE           = 'dataset/train_continuous.npy'
GENERATE_DATA     = False   # True = force regenerate; False = skip if file exists
NUM_SAMPLES       = 10000   # total samples to generate (only used when generating)
INTERP_POINTS     = 48      # must match config horizon (must be divisible by 4)
MAP_RES           = 64      # must match config cnn_config['image_size']
BOUNDS            = [(0, 8), (0, 8)]

MIN_OBSTACLES     = 5       # min number of circular obstacles per scenario
MAX_OBSTACLES     = 12       # max number of circular obstacles per scenario
RADIUS_MIN        = 0.3     # obstacle radius range (world units)
RADIUS_MAX        = 0.8

NUM_TRAIN_SAMPLES = 5000    # samples to train on; set to None to use the full dataset
NUM_EPOCHS        = 20000
SAVE_CKPT_EPOCH   = 1000
EVAL_EVERY        = 100     # generate visualization every N epochs


# Eval helpers

def make_random_scenario(bounds=BOUNDS, min_obstacles=MIN_OBSTACLES, max_obstacles=MAX_OBSTACLES,
                         radius_min=RADIUS_MIN, radius_max=RADIUS_MAX, max_attempts=100):
    """Generate a random (occ_map, start, goal, gt_path) scenario using RRT*."""
    from data_generator.RRT_star import RRTStar
    from data_generator.data_generator import _line_blocked

    bounds_arr = np.asarray(bounds, dtype=float)
    rrt = RRTStar(bounds=bounds)

    for _ in range(max_attempts):
        start = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        goal  = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        if np.linalg.norm(start - goal) < 2.0:
            continue

        obstacles = []
        for _ in range(np.random.randint(min_obstacles, max_obstacles + 1)):
            center = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            radius = np.random.uniform(radius_min, radius_max)
            obstacles.append((center, radius))

        if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
            continue
        if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
            continue

        if not _line_blocked(start, goal, obstacles):
            continue

        path = rrt.plan(start, goal, obstacles, optimize=True, interp_points=INTERP_POINTS)
        if path is None:
            continue

        # build 3-channel map: ch0=obstacles, ch1=start blob, ch2=goal blob
        res = MAP_RES
        xmin, xmax = bounds_arr[0]
        ymin, ymax = bounds_arr[1]
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        grid_x, grid_y = np.meshgrid(xs, ys)
        occ = np.zeros((res, res), dtype=np.float32)
        for (cx, cy), r in obstacles:
            occ[np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2) <= r] = 1.0

        start_f = start.astype(np.float32)
        goal_f  = goal.astype(np.float32)
        ch1 = _gaussian_blob(start_f, bounds, shape=(res, res))
        ch2 = _gaussian_blob(goal_f,  bounds, shape=(res, res))
        map3 = np.stack([occ, ch1, ch2], axis=0)   # (3, H, W)

        return map3, start_f, goal_f, np.asarray(path, dtype=np.float32)

    return None


def make_eval_fn(noise_scheduler, config_dict, config, save_dir, fixed_scenario):
    """Return an eval callback: eval_fn(net, epoch) → saves PNG to save_dir/{fixed,random}/."""
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend, safe in training loop
    import matplotlib.pyplot as plt
    from core.diffusion.policy import PlaneDiffusionPolicy

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _run_and_plot(net, scenario, tag, epoch):
        if scenario is None:
            return
        occ_map, start, goal, gt_path = scenario

        policy = PlaneDiffusionPolicy(
            model=net, noise_scheduler=noise_scheduler,
            config=config_dict, device=DEVICE,
        )
        obs_dict = {
            'env': np.concatenate([start, goal]),  # kept for hard endpoint clamp
            'map': occ_map,                        # (3, H, W) already
        }
        initial_action = torch.randn(1, config.horizon, config.action_dim, device=DEVICE)
        pred_path, _ = policy.predict_action(obs_dict, initial_action)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(occ_map[0], origin='lower', cmap='gray_r',   # ch0 = obstacles
                  extent=[BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]])
        ax.plot(gt_path[:, 0],   gt_path[:, 1],   'b--', linewidth=1.5, label='ground truth')
        ax.plot(pred_path[:, 0], pred_path[:, 1], 'r-',  linewidth=2.0, label='diffusion')
        ax.scatter(*start, c='g', s=80, zorder=5, label='start')
        ax.scatter(*goal,  c='r', s=80, zorder=5, label='goal')
        ax.set_xlim(BOUNDS[0]); ax.set_ylim(BOUNDS[1])
        ax.set_title(f'Epoch {epoch} — {tag}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(save_dir / tag / f'epoch_{epoch:05d}.png', dpi=100)
        plt.close(fig)

    def eval_fn(net, epoch):
        _run_and_plot(net, fixed_scenario,       'fixed',  epoch)
        _run_and_plot(net, make_random_scenario(), 'random', epoch)

    return eval_fn


def main():
    # 1. Data Generation
    if GENERATE_DATA or not Path(OUTFILE).exists():
        from data_generator.data_generator import DataGenerator2D
        print(f'Generating {NUM_SAMPLES} samples → {OUTFILE}')
        gen = DataGenerator2D(
            bounds=BOUNDS,
            num_samples=NUM_SAMPLES,
            min_obstacles=MIN_OBSTACLES,
            max_obstacles=MAX_OBSTACLES,
            radius_range=(RADIUS_MIN, RADIUS_MAX),
            map_resolution=MAP_RES,
            interp_points=INTERP_POINTS,
            outfile=OUTFILE,
        )
        gen.generate_mp()
    else:
        print(f'Dataset found at {OUTFILE}, skipping generation.')

    # 2. Dataset Loading
    from core.datasets.plane_dataset_3ch import PlanePlanningDataSets3Ch

    dataset = PlanePlanningDataSets3Ch(OUTFILE, bounds=BOUNDS)
    print(f'Dataset size : {len(dataset)}')

    if NUM_TRAIN_SAMPLES is not None:
        dataset = torch.utils.data.Subset(dataset, range(NUM_TRAIN_SAMPLES))
    print(f'Training on  : {len(dataset)} samples')

    sample = dataset[0]
    print('sample keys  :', list(sample.keys()))
    print('trajectory   :', sample['sample'].shape)
    print('map          :', sample['map'].shape)   # (3, 64, 64)

    # 3. Model Construction
    from config.plane_continuous import PlaneContinuousConfig
    from core.networks.embeddUnet import ConditionalUnet1D
    from core.diffusion.builder import build_noise_scheduler_from_config

    config      = PlaneContinuousConfig()
    config_dict = config.to_dict()

    cnn_output_dim = config.network_config['cnn_config']['output_dim']

    net = ConditionalUnet1D(
        input_dim       = config.action_dim,
        global_cond_dim = cnn_output_dim,   # 128; no MLP term
        network_config  = config.network_config,
        is_cnn          = config.is_CNN,
    )

    noise_scheduler = build_noise_scheduler_from_config(config_dict)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total_params:,}')

    # shape sanity check
    dummy_action = torch.randn(2, config.horizon, config.action_dim)
    dummy_map    = torch.randn(2, 3, 64, 64)   # 3-channel
    dummy_t      = torch.zeros(2).long()
    out = net(dummy_action, dummy_t, dummy_map)
    print(f'Forward pass output shape: {out.shape}')   # (2, 48, 2)

    # 4. Setup eval & output directories
    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(exist_ok=True)

    viz_dir = Path('intermediate_results')
    (viz_dir / 'fixed').mkdir(parents=True, exist_ok=True)
    (viz_dir / 'random').mkdir(parents=True, exist_ok=True)

    print('Generating fixed test scenario...')
    fixed_scenario = make_random_scenario()
    while fixed_scenario is None:
        fixed_scenario = make_random_scenario()

    eval_fn = make_eval_fn(noise_scheduler, config_dict, config, viz_dir, fixed_scenario)

    # 5. Training
    from core.trainer.plane_diffusion_trainer_embeed import PlaneDiffusionTrainer

    trainer = PlaneDiffusionTrainer(
        net     = net,
        dataset = dataset,
        config  = config,
    )

    trainer.train(
        num_epochs      = NUM_EPOCHS,
        save_ckpt_epoch = SAVE_CKPT_EPOCH,
        eval_fn         = eval_fn,
        eval_every      = EVAL_EVERY,
    )

    trainer.save_checkpoint(str(ckpt_dir / 'ckpt_continuous_final.ckpt'))
    print('Training complete.')

    # 6. Inference test
    from core.diffusion.policy import PlaneDiffusionPolicy

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy = PlaneDiffusionPolicy(
        model           = net,
        noise_scheduler = noise_scheduler,
        config          = config_dict,
        device          = DEVICE,
    )

    raw_data = np.load(OUTFILE, allow_pickle=True).item()
    test_idx = 10

    start_t = raw_data['start'][test_idx]
    goal_t  = raw_data['goal'][test_idx]
    occ_t   = raw_data['map'][test_idx]   # (H, W)
    ch1_t   = _gaussian_blob(start_t, BOUNDS, shape=occ_t.shape)
    ch2_t   = _gaussian_blob(goal_t,  BOUNDS, shape=occ_t.shape)
    map3_t  = np.stack([occ_t, ch1_t, ch2_t], axis=0)   # (3, H, W)

    obs_dict = {
        'env': np.concatenate([start_t, goal_t]),   # for hard clamp
        'map': map3_t,                               # (3, H, W)
    }

    initial_action = torch.randn(1, config.horizon, config.action_dim, device=DEVICE)
    pred_path, _ = policy.predict_action(obs_dict, initial_action)
    print('Predicted path shape:', pred_path.shape)


if __name__ == '__main__':
    main()
