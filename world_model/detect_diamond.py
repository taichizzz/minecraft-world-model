"""
Detect "diamond_block visible in frame" by counting cyan pixels.

Visualization step before we retrain the value head. Verifies that the
color rule fires on actual diamond blocks and not on random scenery.

"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIRS = ["dataset/dataset2_turn", "dataset/dataset3_turn"]

# Diamond block: light cyan/aqua. Distinctive feature is G≈B and both >> R.
# Use color RATIOS rather than absolute thresholds so detection survives
# lighting changes (Minecraft day/night cycle, shaded faces, etc.).
MIN_BRIGHTNESS = 80   # R+G+B floor — below this it's basically pitch dark
G_RATIO_MIN = 0.32    # green share of total
B_RATIO_MIN = 0.32    # blue share of total
R_RATIO_MAX = 0.30    # red share — diamond's red is suppressed
GR_DELTA_MIN = 15     # absolute G-R, prevents triggering on gray
BR_DELTA_MIN = 15     # absolute B-R
MIN_PIXELS = 30       # frame counts as "diamond visible" if >= this many cyan pixels

OUT_DIR = "world_model_out/April26th"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
N_DIAG = 6


def diamond_pixel_mask(frame):
    """Return a HxW boolean mask of pixels that look like diamond_block.

    Uses color RATIOS (channel / brightness) so the rule survives Minecraft
    day/night lighting drift; absolute G-R / B-R deltas keep us off pure
    grayscale; the brightness floor stops us firing on noise in dark frames.
    """
    rgb = frame.astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    brightness = R + G + B + 1e-6
    r_ratio = R / brightness
    g_ratio = G / brightness
    b_ratio = B / brightness

    return (
        (brightness > MIN_BRIGHTNESS)
        & (g_ratio > G_RATIO_MIN)
        & (b_ratio > B_RATIO_MIN)
        & (r_ratio < R_RATIO_MAX)
        & ((G - R) > GR_DELTA_MIN)
        & ((B - R) > BR_DELTA_MIN)
    )


def cyan_count(frame):
    return int(diamond_pixel_mask(frame).sum())


def relabel_episode(obs):
    counts = np.array([cyan_count(obs[t]) for t in range(obs.shape[0])])
    labels = (counts >= MIN_PIXELS).astype(np.float32)
    return labels, counts


def main():
    random.seed(SEED)
    diag = DATASET_DIRS[0]
    files = sorted([f for f in os.listdir(diag)
                    if f.startswith("episode_") and f.endswith(".npz")])
    chosen = random.sample(files, min(N_DIAG, len(files)))

    # 1) cyan-count curve over time, per episode
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.flatten()
    for i, f in enumerate(chosen):
        data = np.load(os.path.join(diag, f))
        obs = data["obs"]
        labels, counts = relabel_episode(obs)
        T = obs.shape[0]
        axes[i].plot(range(T), counts, lw=0.8)
        axes[i].axhline(MIN_PIXELS, color="red", ls="--", lw=0.6,
                        label=f"threshold ({MIN_PIXELS})")
        axes[i].set_title(f"{f}  T={T}  positives={int(labels.sum())}", fontsize=8)
        axes[i].set_xlabel("frame")
        axes[i].set_ylabel("# cyan pixels")
        axes[i].legend(fontsize=6)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "diamond_visibility_curves.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    # 2) sample frames classified positive vs negative for one episode
    f = chosen[0]
    data = np.load(os.path.join(diag, f))
    obs = data["obs"]
    labels, counts = relabel_episode(obs)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(SEED)
    n_show = 6
    pos_sampled = (rng.choice(pos_idx, min(n_show, len(pos_idx)), replace=False)
                   if len(pos_idx) else np.array([], dtype=int))
    neg_sampled = (rng.choice(neg_idx, min(n_show, len(neg_idx)), replace=False)
                   if len(neg_idx) else np.array([], dtype=int))

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for j in range(n_show):
        if j < len(pos_sampled):
            t = pos_sampled[j]
            axes[0, j].imshow(obs[t] / 255.0)
            axes[0, j].set_title(f"POS t={t}  cyan={counts[t]}", fontsize=7)
        axes[0, j].axis("off")
        if j < len(neg_sampled):
            t = neg_sampled[j]
            axes[1, j].imshow(obs[t] / 255.0)
            axes[1, j].set_title(f"neg t={t}  cyan={counts[t]}", fontsize=7)
        axes[1, j].axis("off")
    plt.suptitle(f"{f}: row 1 = diamond detected; row 2 = not detected", fontsize=10)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "diamond_pos_neg_samples.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")

    # 3) dataset-level stats: how many positives total, how many episodes have any
    for d in DATASET_DIRS:
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("episode_") and f.endswith(".npz")])
        total_frames = 0
        total_pos = 0
        n_eps_with_pos = 0
        for f in files:
            data = np.load(os.path.join(d, f))
            obs = data["obs"]
            labels, _ = relabel_episode(obs)
            total_frames += len(labels)
            total_pos += int(labels.sum())
            if labels.sum() > 0:
                n_eps_with_pos += 1
        pos_pct = 100 * total_pos / max(total_frames, 1)
        ep_pct = 100 * n_eps_with_pos / max(len(files), 1)
        print(f"\n{d}:")
        print(f"  {len(files)} episodes, {total_frames} frames")
        print(f"  positives:                {total_pos}  ({pos_pct:.1f}%)")
        print(f"  episodes with >=1 positive: {n_eps_with_pos}  ({ep_pct:.1f}%)")


if __name__ == "__main__":
    main()
