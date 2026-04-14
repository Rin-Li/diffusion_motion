from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from config.plane_fm import PlaneFMConfig
from core.datasets.plane_dataset_3ch import PlanePlanningDataSets3Ch, _gaussian_blob
from core.flow_matching import FlowMatching
from core.flow_matching.policy import FlowMatchingPolicy
from core.networks.embedUnet import ConditionalUnet1D
from core.trainer.flow_matching_trainer import FlowMatchingTrainer
from data_generator.RRT_star import RRTStar
from utils.scenario_utils import line_blocked
from utils.wandb_utils import init_wandb

matplotlib.use('Agg')

OUTFILE           = 'dataset/train_continuous.npy'
GENERATE_DATA     = False
NUM_SAMPLES       = 10000
INTERP_POINTS     = 48
MAP_RES           = 64
BOUNDS            = [(0, 8), (0, 8)]

MIN_OBSTACLES     = 5
MAX_OBSTACLES     = 10
RADIUS_MIN        = 0.3
RADIUS_MAX        = 0.8

NUM_TRAIN_SAMPLES = 1000
NUM_EPOCHS        = 2000
SAVE_CKPT_EPOCH   = 10000
EVAL_EVERY        = 500


def make_random_scenario(bounds=BOUNDS, min_obstacles=MIN_OBSTACLES, max_obstacles=MAX_OBSTACLES,
                         radius_min=RADIUS_MIN, radius_max=RADIUS_MAX, max_attempts=100):
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
        if not line_blocked(start, goal, obstacles):
            continue

        path = rrt.plan(start, goal, obstacles, prune=True, smooth=True, interp_points=INTERP_POINTS)
        if path is None:
            continue

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


def make_eval_fn(fm, config_dict, config, save_dir, fixed_scenario):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _run_and_plot(net, scenario, tag, epoch):
        if scenario is None:
            return
        occ_map, start, goal, gt_path = scenario

        policy = FlowMatchingPolicy(
            model=net, flow_matching=fm, config=config_dict, device=DEVICE,
        )
        obs_dict = {
            'env': np.concatenate([start, goal]),
            'map': occ_map,
        }
        x0 = torch.randn(1, config.horizon, config.action_dim, device=DEVICE)
        pred_path = policy.predict_action(obs_dict, x0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('#F7F7F7')
        ax.imshow(occ_map[0], origin='lower', cmap='Greys', interpolation='bilinear',
                  extent=[BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1]], vmin=0, vmax=1)
        ax.plot(gt_path[:, 0],   gt_path[:, 1],   color='#5C6BC0', linestyle='--', linewidth=2.0,  label='ground truth')
        ax.plot(pred_path[:, 0], pred_path[:, 1], color='#EF6C00', linestyle='-',  linewidth=2.5,  label='flow matching')
        ax.scatter(*start, color='#43A047', marker='*', s=250, zorder=6, label='start')
        ax.scatter(*goal,  color='#E53935', marker='D', s=120, zorder=6, label='goal')
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'Epoch {epoch}  —  {tag}', fontsize=11, fontweight='bold', pad=8)
        ax.legend(fontsize=9, framealpha=0.85, loc='upper right')
        plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
        fig.savefig(save_dir / tag / f'epoch_{epoch:05d}.png', dpi=180)
        plt.close(fig)

    def eval_fn(net, epoch):
        _run_and_plot(net, fixed_scenario,        'fixed',  epoch)
        _run_and_plot(net, make_random_scenario(), 'random', epoch)

    return eval_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-notes", default=None)
    args = parser.parse_args()

    # 1. Config
    config      = PlaneFMConfig()
    config_dict = config.to_dict()

    wandb_run = init_wandb(
        args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        config=config_dict,
        group="flow-matching",
    )

    # 2. Data generation
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
            crop=config.data_generator['crop'],
            smooth=config.data_generator['smooth'],
        )
        gen.generate_mp()
    else:
        print(f'Dataset found at {OUTFILE}, skipping generation.')

    # 3. Dataset loading
    dataset = PlanePlanningDataSets3Ch(OUTFILE, bounds=BOUNDS)
    print(f'Dataset size : {len(dataset)}')

    if NUM_TRAIN_SAMPLES is not None:
        dataset = torch.utils.data.Subset(dataset, range(NUM_TRAIN_SAMPLES))
    print(f'Training on  : {len(dataset)} samples')
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "num_epochs": NUM_EPOCHS,
                "save_ckpt_epoch": SAVE_CKPT_EPOCH,
                "eval_every": EVAL_EVERY,
                "num_train_samples": NUM_TRAIN_SAMPLES,
                "dataset_size": len(dataset),
            },
            allow_val_change=True,
        )

    sample = dataset[0]
    print('sample keys  :', list(sample.keys()))
    print('trajectory   :', sample['sample'].shape)
    print('map          :', sample['map'].shape)

    # 4. Model construction
    if config.network_config.get('use_xcloud', False):
        global_cond_dim = config.network_config['xcloud_encoder']['embed_dim']
    else:
        global_cond_dim = config.network_config['cnn_config']['output_dim']

    net = ConditionalUnet1D(
        input_dim       = config.action_dim,
        global_cond_dim = global_cond_dim,
        network_config  = config.network_config,
        is_cnn          = config.is_CNN,
    )
    print(f'Total parameters: {sum(p.numel() for p in net.parameters()):,}')

    fm = FlowMatching(num_infer_steps=config.flow_matching['num_infer_steps'])

    # shape sanity check
    dummy_action = torch.randn(2, config.horizon, config.action_dim)
    dummy_map    = torch.randn(2, 3, 64, 64)
    dummy_t      = torch.rand(2)
    total_points = config.network_config.get('xcloud_encoder', {}).get('total_points', 128)
    dummy_xcloud = torch.randn(2, total_points, 2) if config.network_config.get('use_xcloud', False) else None
    out = net(dummy_action, dummy_t, dummy_map, xcloud=dummy_xcloud)
    print(f'Forward pass output shape: {out.shape}')

    # 4. Setup eval & output directories
    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(exist_ok=True)

    viz_dir = Path('intermediate_results_fm')
    (viz_dir / 'fixed').mkdir(parents=True, exist_ok=True)
    (viz_dir / 'random').mkdir(parents=True, exist_ok=True)

    print('Generating fixed test scenario...')
    fixed_scenario = make_random_scenario()
    while fixed_scenario is None:
        fixed_scenario = make_random_scenario()

    eval_fn = make_eval_fn(fm, config_dict, config, viz_dir, fixed_scenario)

    # 5. Training
    trainer = FlowMatchingTrainer(
        net=net,
        dataset=dataset,
        config=config,
        wandb_run=wandb_run,
    )
    trainer.train(
        num_epochs      = NUM_EPOCHS,
        save_ckpt_epoch = SAVE_CKPT_EPOCH,
        eval_fn         = eval_fn,
        eval_every      = EVAL_EVERY,
    )

    trainer.save_checkpoint('ckpt/fm_final.ckpt')
    print('Training complete.')

    # 6. Inference test
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = FlowMatchingPolicy(
        model=net, flow_matching=fm, config=config_dict, device=DEVICE,
    )

    raw_data = np.load(OUTFILE, allow_pickle=True).item()
    test_idx = 10
    start_t  = raw_data['start'][test_idx]
    goal_t   = raw_data['goal'][test_idx]
    occ_t    = raw_data['map'][test_idx]
    ch1_t    = _gaussian_blob(start_t, BOUNDS, shape=occ_t.shape)
    ch2_t    = _gaussian_blob(goal_t,  BOUNDS, shape=occ_t.shape)
    map3_t   = np.stack([occ_t, ch1_t, ch2_t], axis=0)   # (3, H, W)

    obs_dict = {
        'env': np.concatenate([start_t, goal_t]),
        'map': map3_t,
    }
    x0 = torch.randn(1, config.horizon, config.action_dim, device=DEVICE)
    pred_path = policy.predict_action(obs_dict, x0)
    print('Predicted path shape:', pred_path.shape)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
