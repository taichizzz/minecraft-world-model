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

# Diamond block detection in HSV. Hue (color tone) is mathematically
# invariant to brightness, so this works at any time of day:
#   - daylight diamond block: RGB ~(137, 232, 226), HSV H≈176°, S≈0.41, V≈232
#   - 50% dim:                RGB ~(68, 116, 113),  HSV H≈176°, S≈0.41, V≈116
#   - 10% dim:                RGB ~(14, 23, 22),    HSV H≈176°, S≈0.39, V≈23
# Hue and saturation barely move; only V drops. So we threshold H+S only.
# Sky is also blue but its hue is ~210° (further from 176), and emerald is
# green hue ~120°, gold yellow hue ~50° — none overlap our window.
HUE_MIN = 165         # diamond hue ≈ 176°, range ±10° gives some slack
HUE_MAX = 195
SAT_MIN = 0.20        # require some saturation to reject grayscale walls
VAL_MIN = 15          # tiny floor to avoid pure-black noise
MIN_PIXELS = 30       # frame counts as "diamond visible" if >= this many cyan pixels

OUT_DIR = "world_model_out/April26th"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
N_DIAG = 6


def diamond_pixel_mask(frame):
    """Return a HxW boolean mask of pixels that look like diamond_block,
    using HSV thresholds so detection survives lighting changes."""
    rgb = frame.astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    M = np.maximum(np.maximum(R, G), B)
    m = np.minimum(np.minimum(R, G), B)
    delta = M - m

    # Saturation: 0 for grayscale, 1 for fully saturated
    S = np.where(M > 0, delta / (M + 1e-6), 0.0)
    V = M

    # Hue in degrees [0, 360). Compute per dominant channel.
    H = np.zeros_like(R)
    safe = delta > 1e-4
    mask_r = safe & (M == R)
    mask_g = safe & (M == G) & ~mask_r
    mask_b = safe & (M == B) & ~mask_r & ~mask_g
    # standard HSV formulas
    H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / (delta[mask_r] + 1e-6))) % 360.0
    H[mask_g] = 60.0 * ((B[mask_g] - R[mask_g]) / (delta[mask_g] + 1e-6)) + 120.0
    H[mask_b] = 60.0 * ((R[mask_b] - G[mask_b]) / (delta[mask_b] + 1e-6)) + 240.0

    return (
        (H >= HUE_MIN) & (H <= HUE_MAX)
        & (S >= SAT_MIN)
        & (V >= VAL_MIN)
    )


def cyan_count(frame):
    return int(diamond_pixel_mask(frame).sum())


def relabel_episode(obs):
    counts = np.array([cyan_count(obs[t]) for t in range(obs.shape[0])])
    labels = (counts >= MIN_PIXELS).astype(np.float32)
    return labels, counts

def largest_diamond_blob(frame):
    """
    Returns info about the largest connected diamond-like blob.
    This filters out scattered cyan noise and wall-like cyan strips.
    """
    mask = diamond_pixel_mask(frame)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    best = None

    for y0 in range(H):
        for x0 in range(W):
            if not mask[y0, x0] or visited[y0, x0]:
                continue

            stack = [(x0, y0)]
            visited[y0, x0] = True
            xs, ys = [], []

            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)

                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < W and 0 <= ny < H:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

            area = len(xs)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            aspect = bw / max(bh, 1)
            fill = area / max(bw * bh, 1)

            info = {
                "area": area,
                "bbox": (x1, y1, x2, y2),
                "w": bw,
                "h": bh,
                "cx": cx,
                "cy": cy,
                "aspect": aspect,
                "fill": fill,
            }

            if best is None or area > best["area"]:
                best = info

    return best


def diamond_blob_visible(frame):
    """
    More reliable diamond visibility test than raw cyan_count.
    Rejects wall strips and tiny noise.
    """
    blob = largest_diamond_blob(frame)
    if blob is None:
        return False, None

    area = blob["area"]
    bw = blob["w"]
    bh = blob["h"]
    aspect = blob["aspect"]
    fill = blob["fill"]

    # Tune these after 2–3 runs.
    if area < 40:
        return False, blob

    # Reject huge vertical/horizontal wall-like regions.
    if bh > 45 or bw > 45:
        return False, blob

    # Reject very long strips.
    if aspect < 0.35 or aspect > 2.8:
        return False, blob

    # Reject very sparse scattered pixels.
    if fill < 0.20:
        return False, blob

    return True, blob


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
