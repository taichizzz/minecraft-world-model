"""
Quantify how dyn rollout error compounds over horizon.

For each episode in the dataset:
  - encode every real frame -> z_real[t]
  - starting from z_real[0], apply the REAL action sequence to dyn:
      z_pred[t] = dyn(z_pred[t-1], a[t-1])
  - record MSE(z_pred[t], z_real[t]) at each horizon t

Then average across episodes and plot MSE vs horizon. This tells us
exactly where dyn drift becomes a problem.
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn3.pth"
DYN_WEIGHTS = "dynamics_turn_3.pth"
DATASET_DIR = "dataset/dataset2_turn"

LATENT_DIM = 128
NUM_ACTIONS = 3
N_EPISODES = 50
MAX_HORIZON = 32
SEED = 42

OUT_DIR = "world_model_out/April26th"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    random.seed(SEED)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=512
    ).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    files = sorted([f for f in os.listdir(DATASET_DIR)
                    if f.startswith("episode_") and f.endswith(".npz")])
    chosen = random.sample(files, min(N_EPISODES, len(files)))
    print(f"Using {len(chosen)} episodes from {DATASET_DIR}")

    # collect MSE per horizon across episodes
    horizon_mses = [[] for _ in range(MAX_HORIZON + 1)]

    # also track ||z|| as a sanity scale
    z_norms = []

    with torch.no_grad():
        for f in chosen:
            data = np.load(os.path.join(DATASET_DIR, f))
            obs = data["obs"]
            actions = data["actions"]
            T = obs.shape[0]
            if T < MAX_HORIZON + 1:
                continue

            imgs = (torch.from_numpy(obs).float().to(DEVICE)
                    .div_(255.0).permute(0, 3, 1, 2))
            z_real = ae.encoder(imgs)  # (T, latent_dim)
            z_norms.append(z_real.norm(dim=1).mean().item())

            z_pred = z_real[0:1].clone()
            for h in range(1, MAX_HORIZON + 1):
                a = torch.tensor([int(actions[h - 1])], dtype=torch.long, device=DEVICE)
                z_pred = dyn(z_pred, a)
                mse = ((z_pred - z_real[h:h + 1]) ** 2).mean().item()
                horizon_mses[h].append(mse)

    means = np.array([np.mean(m) if m else np.nan for m in horizon_mses])
    stds = np.array([np.std(m) if m else 0.0 for m in horizon_mses])
    avg_z_norm = float(np.mean(z_norms)) if z_norms else float("nan")

    horizons = np.arange(MAX_HORIZON + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(horizons, means, marker="o", label="mean MSE")
    plt.fill_between(horizons, means - stds, means + stds, alpha=0.25, label="±1 std")
    plt.xlabel("rollout horizon (steps)")
    plt.ylabel("MSE(z_pred, z_real)")
    plt.title(
        f"Dyn rollout error vs horizon  ({len(chosen)} episodes)\n"
        f"avg ||z||={avg_z_norm:.2f} so MSE>{(0.1*avg_z_norm)**2:.3f} ≈ visibly drifted"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dyn_rollout_error.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")

    print(f"\nAverage ||z||: {avg_z_norm:.4f}")
    print("\nMSE by horizon:")
    for h in [1, 2, 4, 6, 8, 12, 16, 24, 32]:
        if h <= MAX_HORIZON and horizon_mses[h]:
            n = len(horizon_mses[h])
            print(f"  h={h:2d}: mean={means[h]:.4f}  std={stds[h]:.4f}  n={n}")


if __name__ == "__main__":
    main()
