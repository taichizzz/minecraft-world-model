import os
import random
import numpy as np
import torch

DATASET_DIR = "dataset/dataset2"   

def main():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not files:
        raise RuntimeError("No episodes found")

    ep_file = random.choice(files)
    ep_path = os.path.join(DATASET_DIR, ep_file)
    print("Using episode:", ep_path)

    data = np.load(ep_path)
    obs = data["obs"]          # (T,64,64,3)
    actions = data["actions"]  # (T,)

    T = obs.shape[0]

    print("\nAction histogram:")
    bincount = np.bincount(actions)
    for i, count in enumerate(bincount):
        print(f"Action {i}: {count}")

    obs_tensor = torch.from_numpy(obs).float() / 255.0  # (T,H,W,3)

    diffs = obs_tensor[1:] - obs_tensor[:-1]  # (T-1,H,W,3)

    mse_per_step = (diffs ** 2).mean(dim=(1,2,3))
    l2_per_step = diffs.view(diffs.shape[0], -1).norm(dim=1)

    print("\nPixel transition statistics:")
    print("Mean pixel MSE per step:", mse_per_step.mean().item())
    print("Std pixel MSE per step:", mse_per_step.std().item())

    print("Mean pixel L2 per step:", l2_per_step.mean().item())
    print("Std pixel L2 per step:", l2_per_step.std().item())

if __name__ == "__main__":
    main()