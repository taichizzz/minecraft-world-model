"""
Per-action real consec-step breakdown.

For each real frame pair (t, t+1) in the dataset, record
  ||AE.encode(obs[t+1]) - AE.encode(obs[t])||
grouped by the action taken at t.

This tells us whether the dynamics' per-action step size matches reality:
  dyn(z, a=fwd) step  vs  real fwd step
  dyn(z, a=turnL) step vs  real turnL step
  dyn(z, a=turnR) step vs  real turnR step
"""
import os
import numpy as np
import torch

from model import AutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn2.pth"
DATASET_DIR = "dataset/dataset2_turn"
LATENT_DIM = 128
NUM_ACTIONS = 3
ACTION_NAMES = ["fwd", "turnL", "turnR"]
N_EPISODES = 10
SEED = 42


def to_torch_img(obs):
    x = torch.from_numpy(obs).float() / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def main():
    rng = np.random.default_rng(SEED)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    chosen = rng.choice(files, size=min(N_EPISODES, len(files)), replace=False)
    print(f"AE: {AE_WEIGHTS}  dataset: {DATASET_DIR}  episodes: {len(chosen)}")

    steps_by_action = {a: [] for a in range(NUM_ACTIONS)}

    for ep_file in chosen:
        data = np.load(os.path.join(DATASET_DIR, ep_file))
        obs = data["obs"]
        actions = data["actions"]
        T = obs.shape[0]
        if T < 2:
            continue

        with torch.no_grad():
            imgs = to_torch_img(obs).to(DEVICE)
            z = ae.encoder(imgs)  # (T, 128)
            dz = (z[1:] - z[:-1]).norm(dim=1).cpu().numpy()  # (T-1,)

        a = actions[:T - 1]
        for act in range(NUM_ACTIONS):
            mask = (a == act)
            if mask.any():
                steps_by_action[act].extend(dz[mask].tolist())

    print("\n=== real per-action consec step on AE latent ===")
    all_steps = []
    for act in range(NUM_ACTIONS):
        arr = np.array(steps_by_action[act])
        all_steps.extend(arr.tolist())
        if len(arr) == 0:
            print(f"{ACTION_NAMES[act]:5s}: no samples")
            continue
        print(f"{ACTION_NAMES[act]:5s}: n={len(arr):5d}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"median={np.median(arr):.4f}")
    all_arr = np.array(all_steps)
    print(f"\nall  : n={len(all_arr):5d}  mean={all_arr.mean():.4f}  "
          f"std={all_arr.std():.4f}")

    print("\nCompare to dyn predictions from test_action_discrimination.py.")
    print("If real fwd >> dyn fwd, dynamics is undermodeling forward motion.")


if __name__ == "__main__":
    main()
