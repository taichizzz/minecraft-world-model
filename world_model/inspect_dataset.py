"""
Diagnostic: are the episodes actually ending on diamond_block touches,
or are they timing out at a step cap?

Plots:
  1. histogram of episode lengths per dataset directory
  2. grid of LAST frames from random episodes (visually scan for diamond blocks)
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIRS = ["dataset/dataset2_turn", "dataset/dataset3_turn"]
N_LAST_FRAMES = 24
SEED = 42
OUT_DIR = "world_model_out/April26th"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    random.seed(SEED)

    fig_hist, ax_hist = plt.subplots(figsize=(9, 4))
    for d in DATASET_DIRS:
        if not os.path.isdir(d):
            print(f"SKIP {d} (not found)")
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("episode_") and f.endswith(".npz")])
        lengths = []
        for f in files:
            data = np.load(os.path.join(d, f))
            lengths.append(int(data["obs"].shape[0]))
        max_T = max(lengths)
        n_at_max = sum(1 for x in lengths if x == max_T)
        n_short = sum(1 for x in lengths if x < max_T)
        print(f"\n{d}: {len(files)} episodes")
        print(f"  T:  min={min(lengths)}  max={max_T}  mean={np.mean(lengths):.1f}")
        print(f"  at T={max_T}: {n_at_max} ({n_at_max/len(lengths)*100:.1f}%)  "
              f"<-- likely TIMEOUT (no touch)")
        print(f"  shorter:    {n_short} ({n_short/len(lengths)*100:.1f}%)  "
              f"<-- likely real diamond TOUCH")
        ax_hist.hist(lengths, bins=40, alpha=0.5, label=d)

    ax_hist.set_xlabel("episode length T")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("episode length distribution\n(spike at max = timeouts; spread = touches)")
    ax_hist.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "episode_lengths.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved length histogram: {out}")

    # last frames from random episodes
    for d in DATASET_DIRS:
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("episode_") and f.endswith(".npz")])
        chosen = random.sample(files, min(N_LAST_FRAMES, len(files)))
        cols, rows = 6, 4
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()
        for i, f in enumerate(chosen):
            data = np.load(os.path.join(d, f))
            obs = data["obs"]
            T = obs.shape[0]
            axes[i].imshow(obs[-1] / 255.0)
            axes[i].axis("off")
            axes[i].set_title(f"{f}  T={T}", fontsize=6)
        for j in range(len(chosen), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"last_frames_{os.path.basename(d)}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved last frames: {out}")


if __name__ == "__main__":
    main()
