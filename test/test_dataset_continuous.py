"""
Visualize 200 random samples from the continuous dataset.
Saves a multi-page PDF to test/dataset_continuous_viz.pdf.

Usage (from repo root):
    python test/test_dataset_continuous.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

DATASET_PATH = 'dataset/train_continuous.npy'
OUTPUT_PDF   = 'test/dataset_continuous_viz.pdf'
NUM_SAMPLES  = 200
COLS, ROWS   = 5, 4          # 20 samples per page → 10 pages
BOUNDS       = (0, 8)

data    = np.load(DATASET_PATH, allow_pickle=True).item()
n_total = data['paths'].shape[0]
print(f'Dataset loaded: {n_total} samples')
print(f'  paths : {data["paths"].shape}')
print(f'  map   : {data["map"].shape}')
print(f'  start : {data["start"].shape}')
print(f'  goal  : {data["goal"].shape}')

indices = np.random.choice(n_total, size=min(NUM_SAMPLES, n_total), replace=False)

per_page = COLS * ROWS
pages    = int(np.ceil(len(indices) / per_page))

out_path = Path(OUTPUT_PDF)
out_path.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(out_path) as pdf:
    for page in range(pages):
        batch = indices[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3, ROWS * 3))
        axes = axes.flatten()

        for ax_idx, sample_idx in enumerate(batch):
            ax  = axes[ax_idx]
            occ = data['map'][sample_idx]       # (H, W)
            path  = data['paths'][sample_idx]   # (T, 2)
            start = data['start'][sample_idx]   # (2,)
            goal  = data['goal'][sample_idx]    # (2,)

            ax.imshow(occ, origin='lower', cmap='gray_r',
                      extent=[BOUNDS[0], BOUNDS[1], BOUNDS[0], BOUNDS[1]])
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=1.2)
            ax.scatter(*start, c='lime',   s=40, zorder=5)
            ax.scatter(*goal,  c='red',    s=40, zorder=5)
            ax.set_xlim(BOUNDS); ax.set_ylim(BOUNDS)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'#{sample_idx}', fontsize=7)

        # hide unused axes on last page
        for ax_idx in range(len(batch), len(axes)):
            axes[ax_idx].set_visible(False)

        fig.suptitle(f'Continuous dataset — page {page + 1}/{pages}', fontsize=10)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print(f'  page {page + 1}/{pages} done')

print(f'Saved → {out_path.resolve()}')
