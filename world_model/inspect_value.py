"""
Diagnostic: plot V(z) over real episodes (NOT imagined trajectories).

If V works as intended, we should see V near 0 for most frames and rising
toward 1.0 in the last NEAR_GOAL_K=5 frames before the agent touches the
diamond. If the curve is a sharp spike only at the very end, the value
function is too sparse to give the planner a useful gradient.
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AutoEncoder
from train_value_head import ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn3.pth"
VALUE_WEIGHTS = "value_head_2.pth"
DATASET_DIR = "dataset/dataset2_turn"
N_EPISODES = 6
LATENT_DIM = 128
HIDDEN = 256
SEED = 7

OUT_PATH = "world_model_out/April26th/v_on_real_episodes.png"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def to_torch_img(x):
    x = torch.from_numpy(x).float() / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def main():
    random.seed(SEED)
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    value = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE))
    value.eval()

    files = sorted([f for f in os.listdir(DATASET_DIR)
                    if f.startswith("episode_") and f.endswith(".npz")])
    chosen = random.sample(files, min(N_EPISODES, len(files)))

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.flatten()

    for i, f in enumerate(chosen):
        data = np.load(os.path.join(DATASET_DIR, f))
        obs = data["obs"]
        T = obs.shape[0]
        with torch.no_grad():
            imgs = to_torch_img(obs).to(DEVICE)
            z = ae.encoder(imgs)
            v = value(z).cpu().numpy()
        axes[i].plot(range(T), v, marker=".", markersize=3)
        axes[i].axhline(0.5, color="gray", ls="--", lw=0.6)
        axes[i].axvline(T - 5, color="red", ls="--", lw=0.6, label="T-5 (positive cutoff)")
        axes[i].set_title(f"{f}  T={T}", fontsize=8)
        axes[i].set_xlabel("frame")
        axes[i].set_ylabel("V(z)")
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
