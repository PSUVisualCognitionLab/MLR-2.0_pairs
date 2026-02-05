# show_pairs.py
import os, sys, random
import numpy as np

# use non-GUI backend (works on remote servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import your dataset class
try:
    from dataset_fast import QuickDrawPairDatasetFast
except ModuleNotFoundError:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from dataset_fast import QuickDrawPairDatasetFast


DATA_ROOT = "/home/bwyble/data/quickdraw_npy"   # where your .npy files are
PAIRS = [
    ("airplane", "bird"),
    ("car", "cat"),
    ("dog", "duck"),
    ("frog", "horse"),
    ("sailboat", "truck"),
]
OUT_DIR = "samples"
OUT_PATH = os.path.join(OUT_DIR, "pairs_grid.png")

N_ROWS, N_COLS = 2, 5  # grid size (2x5 = 10 samples)
ITEMS_PER_PAIR = 2000
SEED = 123

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    ds = QuickDrawPairDatasetFast(
        root=DATA_ROOT,
        class_pairs=PAIRS,
        classes_dirname="",   # since .npy files are directly in DATA_ROOT
        items_per_pair=ITEMS_PER_PAIR,
        seed=SEED,
    )

    total = N_ROWS * N_COLS
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3, N_ROWS * 2.2))
    axes = axes.flatten()

    for ax in axes[:total]:
        idx = random.randrange(len(ds))
        l, r, y, pair_idx = ds[idx]
        left_name, right_name = PAIRS[pair_idx]
        l_img = l.squeeze(0).numpy()
        r_img = r.squeeze(0).numpy()

        # combine the two images horizontally (28x56)
        stitched = np.concatenate([l_img, r_img], axis=1)
        ax.imshow(stitched, cmap="gray")
        ax.set_title(f"{left_name} | {right_name}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved image grid: {OUT_PATH}")

if __name__ == "__main__":
    main()
