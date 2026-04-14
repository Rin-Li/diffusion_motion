# Diffusion Path Planning in 2D

**2D trajectory generation using generative models**, producing smooth and collision-free paths in environments with circular or rectangular obstacles.

| Model | Algorithm |
|-------|-----------|
| DDPM | Denoising Diffusion Probabilistic Model |
| Flow Matching | Conditional Flow Matching (ODE-based) |

## Results

|  DDPM | Flow Matching |
|------|---------------|
| ![](res/final_path_diffusion.png) | ![](res/final_path_fm.png) |

### Inference process

| DDPM denoising | Flow Matching ODE |
|----------------|-------------------|
| ![](res/diffusion_process.gif) | ![](res/fm_process.gif) |

## Architecture

![](res/all_step.png)

The pipeline has two inputs: the **environment map** and a **noisy trajectory**.

- The 64×64 3-channel map (ch0 = obstacles, ch1 = start blob, ch2 = goal blob) is encoded by a **CNN** into a fixed-size embed feature vector.
- A **noisy trajectory** (Gaussian noise for DDPM; random x₀ for Flow Matching) enters as a sequence of 2D waypoints.
- A **conditional 1D U-Net** uses the CNN embed as a global FiLM condition to iteratively refine the noisy trajectory into a clean path.

The two backbones share the same network and conditioning — they differ only in training objective and inference loop.

## Setup

```bash
pip install -r requirements.txt
```

All scripts are run from the project root with `PYTHONPATH=.`:

```bash
export PYTHONPATH=.
```

## Train

```bash
python train/train_continuous.py   # DDPM
python train/train_fm.py           # Flow Matching
```

Both scripts generate (or load) the dataset via RRT\*, train the U-Net with periodic evaluation, and save checkpoints to `ckpt/` and intermediate plots to `intermediate_results[_fm]/`.

Set `GENERATE_DATA = True` at the top of each script to force dataset regeneration. Key hyperparameters: `config/plane_continuous.py` (DDPM) and `config/plane_fm.py` (FM).

## Evaluate

```bash
# Set MODE = "continuous" or "original" at the top of the file
python test/test_diffusion.py

python test/test_fm.py
```

## Visualize

```bash
python utils/generate_viz.py
```

Outputs process GIFs and final path images to `res/tmp/`.
