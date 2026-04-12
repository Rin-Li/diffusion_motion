# Diffusion Path Planning in 2D

**2D trajectory generation using generative models**, producing smooth and collision-free paths in environments with circular or rectangular obstacles.

Two generative backbones are supported:

| Model | Algorithm | Train script |
|-------|-----------|--------------|
| DDPM | Denoising Diffusion Probabilistic Model | `train_continuous.py` |
| Flow Matching | Conditional Flow Matching (ODE-based) | `train_fm.py` |

Two environment modes are supported:

| Mode | Map | Obstacle type | Encoder |
|------|-----|---------------|---------|
| `original` | 8×8 binary grid | Rectangles | ViT + MLP |
| `continuous` | 64×64 3-channel image | Circles (continuous space) | CNN |

## Results (original mode)

| No Obstacle | With Obstacle |
|-------------|----------------|
| ![](res/trajectory_evolution_noobstacle.gif) | ![](res/trajectory_evolution_obstacle.gif) |
| ![](res/trajectory_noobstacle.png) | ![](res/trajectory_obstacle.png) |

> Out-of-distribution scenarios (single obstacle or no obstacle) that the model was never trained on.

## Model Architecture

![](res/all_step.png)

A **conditional 1D U-Net** generates a trajectory conditioned on the environment via FiLM modulation. The two generative backbones differ only in training objective and inference loop — the network and conditioning are shared.

### DDPM
- Iterative denoising over 100 timesteps (DDPM scheduler)
- Predicts noise `ε` at each step (`prediction_type = "epsilon"`)
- Inference: reverse diffusion chain

### Flow Matching
- Learns a velocity field between noise and data distributions
- Straight-line conditional flow: `x_t = (1−t)·x₀ + t·x₁`
- Inference: Euler ODE integration (10 steps)

### Environment conditioning

**Original mode**
- **MLP** encodes concatenated start + goal positions
- **ViT** encodes the 8×8 binary obstacle grid
- Both embeddings concatenated → global FiLM conditioning

**Continuous mode**
- Start and goal embedded as **Gaussian blobs** in map channels 1 and 2
- **CNN** encodes the full 3-channel 64×64 map (ch0=obstacles, ch1=start blob, ch2=goal blob)
- No separate MLP branch

## Data Generation

Expert trajectories are generated offline using a custom **RRT\*** planner and saved as `.npy` datasets.

### RRT\* planner

Two planner variants:

| Variant | Obstacle type | Class |
|---------|--------------|-------|
| `RRTStar` | Circular obstacles (continuous space) | `data_generator/RRT_star.py` |
| `RRTStarGrid` | Rectangular obstacles (occupancy grid) | `data_generator/RRT_star_grid.py` |

Both support optional post-processing on the raw RRT\* path:

- **Crop** (`prune=True`) — shortcut pruning: greedily removes intermediate waypoints that are line-of-sight reachable, yielding a shorter path
- **Smooth** (`smooth=True`) — spline smoothing on the pruned path; falls back to linear interpolation if the spline clips an obstacle

### Multiprocessing generation

Continuous-space data generation (`DataGenerator2D`) uses `multiprocessing.Pool` with a map-reduce pattern. Each worker independently samples a random (start, goal, obstacles) triple and runs RRT\*, so all CPU cores are used in parallel.

## Training

**DDPM (continuous mode)**
```bash
python train_continuous.py
```

**Flow Matching**
```bash
python train_fm.py
```

Both scripts:
1. Generate (or load) the dataset via RRT\* with multiprocessing
2. Train the conditional U-Net with periodic visual evaluation
3. Save checkpoints to `ckpt/` and intermediate plots to `intermediate_results[_fm]/`

Key hyperparameters live in `config/plane_continuous.py` (DDPM) and `config/plane_fm.py` (FM). Set `GENERATE_DATA = True` at the top of each script to force dataset regeneration.

## Evaluation

```bash
# Set MODE = "continuous" or "original" at the top of test_diffusion.py
python test_diffusion.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `diffusers`, `numpy`, `scipy`, `matplotlib`, `imageio`, `tqdm`
