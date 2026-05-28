"""
Multi-step dynamics training on human data from all three environments.

Uses the new ae_multienv.pth encoder (trained on env1+env2+env3) so the
dynamics model learns transitions across all rooms.

Same curriculum as train_dynamics_human_multistep.py:
  - K=1 warmup for first 25% of epochs
  - K=4 for the rest (matches MPC_HORIZON)

Saves dynamics_multienv.pth.
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn3.pth"
OUT_WEIGHTS = "dynamics_multienv.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 512

K_FULL = 4

BATCH = 64
EPOCHS = 100
WARMUP_FRAC = 0.25
LR = 1e-3
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


class MultiStepDataset(Dataset):
    """K+1 consecutive frames + K actions from a single episode.
    Skips windows containing idle (action==3)."""

    def __init__(self, data_dir, K=K_FULL):
        self.K = K
        windows_obs, windows_acts = [], []
        skipped_idle = 0

        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        for fp in files:
            data = np.load(fp)
            obs = data["obs"]
            actions = data["actions"]
            T_obs = obs.shape[0]
            last_start = min(T_obs - K - 1, len(actions) - K)
            for t in range(0, last_start + 1):
                a_win = actions[t:t + K]
                if (a_win == 3).any():
                    skipped_idle += 1
                    continue
                windows_obs.append(obs[t:t + K + 1])
                windows_acts.append(a_win)

        if not windows_obs:
            raise RuntimeError(f"No valid windows from {data_dir}")

        self.obs = np.stack(windows_obs)
        self.actions = np.stack(windows_acts).astype(np.int64)
        print(f"  {data_dir}: {len(files)} eps, {len(self.actions)} windows "
              f"(skipped {skipped_idle} idle)")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        f = torch.from_numpy(self.obs[idx]).float().div_(255.0).permute(0, 3, 1, 2)
        a = torch.from_numpy(self.actions[idx])
        return f, a


def rollout_loss(dyn, encoder, frames, actions, K):
    B, Kp1, C, H, W = frames.shape
    flat = frames.reshape(B * Kp1, C, H, W)
    with torch.no_grad():
        z_all = encoder(flat).view(B, Kp1, -1)

    z_pred = z_all[:, 0]
    losses = []
    for k in range(K):
        a_k = actions[:, k]
        z_pred = dyn(z_pred, a_k)
        z_true = z_all[:, k + 1]
        losses.append(F.mse_loss(z_pred, z_true))
    mean_loss = torch.stack(losses).mean()
    return mean_loss, [l.item() for l in losses]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE: {DEVICE}")
    print(f"K_FULL={K_FULL}  WARMUP_FRAC={WARMUP_FRAC}  EPOCHS={EPOCHS}")

    print("\nBuilding multi-env multi-step datasets...")
    datasets = [MultiStepDataset(d, K=K_FULL) for d in DATASET_DIRS]
    combined = ConcatDataset(datasets)
    print(f"  combined: {len(combined)} windows")

    n = len(combined)
    n_val = max(1, int(VAL_FRAC * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(combined, [n_train, n_val], generator=g)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    encoder = ae.encoder

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=HIDDEN
    ).to(DEVICE)
    opt = optim.Adam(dyn.parameters(), lr=LR)

    warmup_epochs = max(1, int(EPOCHS * WARMUP_FRAC))
    train_losses, val_losses_mean = [], []
    best_val = float("inf")

    print(f"\nWarmup (K=1) for first {warmup_epochs} epochs, then K={K_FULL}.\n")
    for epoch in range(EPOCHS):
        K_cur = 1 if epoch < warmup_epochs else K_FULL

        dyn.train()
        tot, nb = 0.0, 0
        for frames, acts in train_loader:
            frames = frames.to(DEVICE); acts = acts.to(DEVICE)
            loss, _ = rollout_loss(
                dyn, encoder,
                frames[:, :K_cur + 1], acts[:, :K_cur], K_cur)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        train_loss = tot / max(nb, 1)
        train_losses.append(train_loss)

        dyn.eval()
        vtot, vn = 0.0, 0
        last_steps = None
        with torch.no_grad():
            for frames, acts in val_loader:
                frames = frames.to(DEVICE); acts = acts.to(DEVICE)
                loss, step_losses = rollout_loss(
                    dyn, encoder,
                    frames[:, :K_cur + 1], acts[:, :K_cur], K_cur)
                vtot += loss.item(); vn += 1
                last_steps = step_losses
        val_loss = vtot / max(vn, 1)
        val_losses_mean.append(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dyn.state_dict(), OUT_WEIGHTS)
            marker = "  <- best"
        step_str = "  ".join(
            f"k={i+1}:{v:.4f}" for i, v in enumerate(last_steps or []))
        print(f"epoch {epoch:02d} K={K_cur} | train {train_loss:.6f} | "
              f"val {val_loss:.6f}  [{step_str}]{marker}")

    print(f"\nBest val mean MSE: {best_val:.6f}")
    print(f"Saved: {OUT_WEIGHTS}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses_mean, label="val")
    plt.axvline(warmup_epochs - 0.5, color="grey", linestyle="--",
                label=f"K: 1 -> {K_FULL}")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title(f"Dynamics multi-env (K=1 -> K={K_FULL})")
    plt.legend(); plt.tight_layout()
    out = os.path.join(OUT_DIR, "dynamics_multienv_loss.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
