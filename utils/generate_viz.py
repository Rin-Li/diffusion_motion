import io
from pathlib import Path
ROOT = Path(__file__).parent.parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config.plane_continuous import PlaneContinuousConfig
from config.plane_fm import PlaneFMConfig
from core.datasets.plane_dataset_3ch import _gaussian_blob
from core.diffusion.builder import build_noise_scheduler_from_config
from core.diffusion.policy import PlaneDiffusionPolicy
from core.flow_matching import FlowMatching
from core.flow_matching.policy import FlowMatchingPolicy
from core.networks.embedUnet import ConditionalUnet1D
from utils.dataset_utils import validate_path_circle_collision_free
from utils.normalizer import LinearNormalizer
from utils.scenario_utils import line_blocked
from utils.viz_utils import (
    MNP_INSPIRED_PALETTE,
    draw_circle_obstacles,
    draw_start_goal,
    style_continuous_axis,
)

OUT  = ROOT / 'res' / 'tmp'
OUT.mkdir(exist_ok=True)

BOUNDS     = [(0, 8), (0, 8)]
MAP_RES    = 64
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
bounds_arr = np.asarray(BOUNDS, dtype=float)
xmin, xmax = bounds_arr[0]
ymin, ymax = bounds_arr[1]


def build_scenario(rng, max_obs=8):
    xs, ys = np.linspace(xmin, xmax, MAP_RES), np.linspace(ymin, ymax, MAP_RES)
    gx, gy = np.meshgrid(xs, ys)
    for _ in range(300):
        start = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        goal  = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        if np.linalg.norm(start - goal) < 2.0:
            continue
        obstacles = []
        for _ in range(rng.integers(1, max_obs + 1)):
            c = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
            r = rng.uniform(0.3, 1.2)
            obstacles.append((c, r))
        if any(np.linalg.norm(start - c) <= r for c, r in obstacles):
            continue
        if any(np.linalg.norm(goal  - c) <= r for c, r in obstacles):
            continue
        if not line_blocked(start, goal, obstacles):
            continue
        occ = np.zeros((MAP_RES, MAP_RES), dtype=np.float32)
        for (cx, cy), rad in obstacles:
            occ[np.sqrt((gx - cx)**2 + (gy - cy)**2) <= rad] = 1.0
        ch1  = _gaussian_blob(start.astype(np.float32), BOUNDS, shape=(MAP_RES, MAP_RES))
        ch2  = _gaussian_blob(goal.astype(np.float32),  BOUNDS, shape=(MAP_RES, MAP_RES))
        map3 = np.stack([occ, ch1, ch2], axis=0)
        return start.astype(np.float32), goal.astype(np.float32), map3, obstacles
    return None


def fm_predict_with_steps(policy, obs_dict, x0):
    fm, net  = policy.fm, policy.net
    env_cond = policy._prep_env(obs_dict)
    map_cond = policy._prep_map(obs_dict)
    xcloud   = policy._prep_xcloud(obs_dict)
    xt        = x0.to(DEVICE)
    timesteps = fm.get_inference_timesteps()
    dt        = 1.0 / fm.num_infer_steps
    norm, stats = LinearNormalizer(), policy.norm_stats
    steps = [norm.unnormalize_data(xt.detach().cpu().numpy()[0], stats=stats)]
    for t in timesteps[:-1]:
        t_batch = t.expand(xt.shape[0]).to(DEVICE)
        with torch.no_grad():
            v_pred = net(sample=xt, timestep=t_batch,
                         map_cond=map_cond, env_cond=env_cond, xcloud=xcloud)
        xt = fm.euler_step(xt, v_pred, dt)
        steps.append(norm.unnormalize_data(xt.detach().cpu().numpy()[0].copy(), stats=stats))
    final = steps[-1].copy()
    env_np = obs_dict.get('env')
    if env_np is not None:
        obs_dim   = len(env_np) // 2
        final[0]  = env_np[:obs_dim]
        final[-1] = env_np[obs_dim:]
    return final, steps


def load_fm_policy():
    cfg = PlaneFMConfig()
    net = ConditionalUnet1D(
        input_dim       = cfg.action_dim,
        global_cond_dim = cfg.network_config['cnn_config']['output_dim'],
        network_config  = cfg.network_config,
        is_cnn          = cfg.is_CNN,
    )
    fm     = FlowMatching(num_infer_steps=cfg.flow_matching['num_infer_steps'])
    policy = FlowMatchingPolicy(model=net, flow_matching=fm, config=cfg.to_dict(), device=DEVICE)
    policy.load_weights(ROOT / 'ckpt' / 'fm_final.ckpt')
    return policy, cfg


def load_diff_policy():
    cfg      = PlaneContinuousConfig()
    cfg_dict = cfg.to_dict()
    net = ConditionalUnet1D(
        input_dim       = cfg.action_dim,
        global_cond_dim = cfg.network_config['cnn_config']['output_dim'],
        network_config  = cfg.network_config,
        is_cnn          = cfg.is_CNN,
    )
    sched  = build_noise_scheduler_from_config(cfg_dict)
    policy = PlaneDiffusionPolicy(model=net, noise_scheduler=sched, config=cfg_dict, device=DEVICE)
    policy.load_weights(ROOT / 'ckpt' / 'ckpt_continuous_final.ckpt')
    return policy, cfg


def render_frame(obstacles, start, goal, waypoints, title=''):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor(MNP_INSPIRED_PALETTE['background'])
    style_continuous_axis(ax, (xmin, xmax), (ymin, ymax))
    draw_circle_obstacles(ax, obstacles)
    pts = np.asarray(waypoints)  # (T, 2)
    ax.scatter(pts[:, 0], pts[:, 1], c=MNP_INSPIRED_PALETTE['success'],
               s=20, alpha=0.75, zorder=5,
               edgecolors=MNP_INSPIRED_PALETTE['path_marker'], linewidths=0.3)
    draw_start_goal(ax, start, goal, size=60)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=MNP_INSPIRED_PALETTE['background'])
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_process_gif(obstacles, start, goal, all_steps, label, out_path, max_frames=40, fps=12):
    step     = max(1, len(all_steps) // max_frames)
    selected = all_steps[::step]
    if all_steps[-1] is not selected[-1]:
        selected = selected + [all_steps[-1]]
    total  = len(all_steps) - 1
    frames = [render_frame(obstacles, start, goal, s,
                           title=f'{label}  step {min(i * step, total)}/{total}')
              for i, s in enumerate(selected)]
    frames += [frames[-1]] * 8
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   loop=0, duration=int(1000 / fps), optimize=False)
    print(f'[OK] {out_path}  ({len(frames)} frames @ {fps} fps)')


def save_final_path(obstacles, start, goal, path, label, out_path, color_key='success'):
    color = MNP_INSPIRED_PALETTE[color_key]
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(MNP_INSPIRED_PALETTE['background'])
    style_continuous_axis(ax, (xmin, xmax), (ymin, ymax))
    draw_circle_obstacles(ax, obstacles)
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5, alpha=0.9, zorder=5)
    ax.scatter(path[:, 0], path[:, 1], c=color, s=18, alpha=0.7, zorder=5,
               edgecolors=MNP_INSPIRED_PALETTE['path_marker'], linewidths=0.3)
    draw_start_goal(ax, start, goal, size=70)
    ax.set_title(f'{label} — Planned Path', fontsize=13, fontweight='bold', pad=8)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=MNP_INSPIRED_PALETTE['background'])
    plt.close(fig)
    print(f'[OK] {out_path}')


def save_env_no_path(obstacles, start, goal, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(MNP_INSPIRED_PALETTE['background'])
    style_continuous_axis(ax, (xmin, xmax), (ymin, ymax))
    draw_circle_obstacles(ax, obstacles)
    draw_start_goal(ax, start, goal, size=70)
    ax.set_title('Environment', fontsize=13, fontweight='bold', pad=8)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=MNP_INSPIRED_PALETTE['background'])
    plt.close(fig)
    print(f'[OK] {out_path}')


def find_cf(rng, method, fm_policy, fm_cfg, diff_policy, diff_cfg_dict, max_tries=50):
    for _ in range(max_tries):
        sc = build_scenario(rng)
        if sc is None:
            continue
        start, goal, map3, obstacles = sc
        obs_dict = {'env': np.concatenate([start, goal]), 'map': map3}
        if method == 'fm':
            x0 = torch.randn(1, fm_cfg.horizon, fm_cfg.action_dim, device=DEVICE)
            path, steps = fm_predict_with_steps(fm_policy, obs_dict, x0)
        else:
            x0 = torch.randn(1, diff_cfg_dict['horizon'], diff_cfg_dict['action_dim'], device=DEVICE)
            path, steps = diff_policy.predict_action(obs_dict, x0)
        if not validate_path_circle_collision_free(path, obstacles):
            continue
        if np.linalg.norm(path[-1] - goal) >= 0.5:
            continue
        print(f'[{method}] collision-free case found.')
        return start, goal, path, steps, obstacles
    raise RuntimeError(f'No collision-free case found for {method}')


if __name__ == '__main__':
    print('Loading policies...')
    fm_policy, fm_cfg     = load_fm_policy()
    diff_policy, diff_cfg = load_diff_policy()
    diff_cfg_dict         = diff_cfg.to_dict()
    rng                   = np.random.default_rng()

    print('\n-- FM --')
    fm_start, fm_goal, fm_path, fm_steps, fm_obs = find_cf(
        rng, 'fm', fm_policy, fm_cfg, diff_policy, diff_cfg_dict)
    make_process_gif(fm_obs, fm_start, fm_goal, fm_steps,
                     'FM', OUT / 'fm_process.gif', max_frames=40, fps=12)
    save_final_path(fm_obs, fm_start, fm_goal, fm_path,
                    'Flow Matching', OUT / 'final_path_fm.png')
    save_env_no_path(fm_obs, fm_start, fm_goal, OUT / 'env_no_path.png')

    print('\n-- Diffusion --')
    d_start, d_goal, d_path, d_steps, d_obs = find_cf(
        rng, 'diff', fm_policy, fm_cfg, diff_policy, diff_cfg_dict)
    make_process_gif(d_obs, d_start, d_goal, d_steps,
                     'Diffusion', OUT / 'diffusion_process.gif', max_frames=40, fps=12)
    save_final_path(d_obs, d_start, d_goal, d_path,
                    'Diffusion', OUT / 'final_path_diffusion.png')

    print('\nAll done →', OUT)
