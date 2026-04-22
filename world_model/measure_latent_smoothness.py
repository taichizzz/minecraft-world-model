import os
import random
import numpy as np
import torch

from model import AutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset/dataset2_turn"

# AE checkpoints to compare. Missing files are skipped with a warning.
# aeturn1 is the current temporal-smoothness AE; ae3/ae4 are pre-smoothness baselines.
AE_CHECKPOINTS = ["aeturn1.pth", "ae4.pth", "ae3.pth"]

LATENT_DIM = 128
N_EPISODES = 10            # how many episodes to average over
MIN_GAP = 10               # min |i - j| for "random non-adjacent" pair
N_RANDOM_PAIRS = 200       # random pairs per episode
SEED = 42


def to_torch_img(batch_hwc_uint8):
    x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def encode_episode(ae, obs):
    with torch.no_grad():
        imgs = to_torch_img(obs).to(DEVICE)
        z = ae.encoder(imgs)
    return z


def sample_random_pairs(T, min_gap, n_pairs, rng):
    """Sample n_pairs (i, j) with |i - j| >= min_gap via oversample + filter."""
    oversample = max(n_pairs * 4, 200)
    i = rng.integers(0, T, size=oversample)
    j = rng.integers(0, T, size=oversample)
    mask = np.abs(i - j) >= min_gap
    i = i[mask][:n_pairs]
    j = j[mask][:n_pairs]
    if len(i) == 0:
        return None, None
    return i, j


def episode_stats(z, rng):
    T = z.shape[0]

    z_mag = z.norm(dim=1).mean().item()
    consec = (z[1:] - z[:-1]).norm(dim=1).mean().item()

    if T <= MIN_GAP + 1:
        return {"z_mag": z_mag, "consec": consec, "random": None}

    i, j = sample_random_pairs(T, MIN_GAP, N_RANDOM_PAIRS, rng)
    if i is None:
        return {"z_mag": z_mag, "consec": consec, "random": None}

    i_t = torch.from_numpy(i).long().to(z.device)
    j_t = torch.from_numpy(j).long().to(z.device)
    random_dist = (z[i_t] - z[j_t]).norm(dim=1).mean().item()

    return {"z_mag": z_mag, "consec": consec, "random": random_dist}


def evaluate_ae(ae_path, chosen_episodes, rng):
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    ae.eval()

    per_ep = []
    for ep_file in chosen_episodes:
        data = np.load(os.path.join(DATASET_DIR, ep_file))
        obs = data["obs"]
        z = encode_episode(ae, obs)
        per_ep.append(episode_stats(z, rng))

    z_mags = np.array([s["z_mag"] for s in per_ep])
    consecs = np.array([s["consec"] for s in per_ep])
    randoms = np.array([s["random"] for s in per_ep if s["random"] is not None])

    return {
        "ae": ae_path,
        "n_episodes": len(per_ep),
        "n_with_random": len(randoms),
        "z_mag_mean": z_mags.mean(),
        "z_mag_std": z_mags.std(),
        "consec_mean": consecs.mean(),
        "consec_std": consecs.std(),
        "random_mean": randoms.mean() if len(randoms) else float("nan"),
        "random_std": randoms.std() if len(randoms) else float("nan"),
    }


def print_summary(r):
    consec_ratio = r["consec_mean"] / r["z_mag_mean"] if r["z_mag_mean"] > 0 else float("inf")
    spread_ratio = r["random_mean"] / r["consec_mean"] if r["consec_mean"] > 0 else float("inf")
    print(f"\n=== {r['ae']}  (N={r['n_episodes']} episodes, {r['n_with_random']} with pairs) ===")
    print(f"||z||            mean={r['z_mag_mean']:.4f}  std={r['z_mag_std']:.4f}")
    print(f"consec step      mean={r['consec_mean']:.4f}  std={r['consec_std']:.4f}")
    print(f"random pair      mean={r['random_mean']:.4f}  std={r['random_std']:.4f}")
    print(f"consec / ||z||   = {consec_ratio:.4f}   (small = smooth)")
    print(f"random / consec  = {spread_ratio:.4f}   (>>1 = informative, ~1 = collapsed)")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    pick_rng = random.Random(SEED)

    episode_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not episode_files:
        raise RuntimeError(f"No episodes in {DATASET_DIR}")

    chosen = pick_rng.sample(
        episode_files, k=min(N_EPISODES, len(episode_files))
    )

    print(f"Dataset: {DATASET_DIR}  ({len(episode_files)} episodes available)")
    print(f"Using {len(chosen)} episodes for all AEs (same set for fair comparison)")
    print(f"AE checkpoints: {AE_CHECKPOINTS}")

    results = []
    for ae_path in AE_CHECKPOINTS:
        if not os.path.exists(ae_path):
            print(f"\nSKIP {ae_path} (not found)")
            continue
        # fresh numpy rng per AE so pair sampling is identical across AEs
        pair_rng = np.random.default_rng(SEED)
        r = evaluate_ae(ae_path, chosen, pair_rng)
        print_summary(r)
        results.append(r)

    if len(results) > 1:
        print("\n\n=== side-by-side ===")
        print(f"{'AE':<18} {'||z||':>10} {'consec':>10} {'random':>10} {'rand/cons':>10}")
        for r in results:
            ratio = r["random_mean"] / r["consec_mean"] if r["consec_mean"] > 0 else float("inf")
            print(f"{r['ae']:<18} {r['z_mag_mean']:>10.4f} {r['consec_mean']:>10.4f} "
                  f"{r['random_mean']:>10.4f} {ratio:>10.3f}")

        print("\nInterpretation:")
        print("  rand/cons >> 1  -> latent is informative (different scenes map far apart)")
        print("  rand/cons ~= 1  -> latent has collapsed (all scenes map close together)")
        print("  Compare aeturn1 against ae3/ae4 to isolate the effect of the smoothness penalty.")


if __name__ == "__main__":
    main()
