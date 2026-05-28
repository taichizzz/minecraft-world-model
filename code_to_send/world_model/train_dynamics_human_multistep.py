"""
Multi-step dynamics training on human data.

Why: the current `dynamics_turn_human.pth` was trained on single-step
prediction — given (z_t, a_t), predict z_{t+1}. MPC rolls the model 4
steps at inference, feeding its own predictions back in. The model has
never seen its own (slightly wrong) predictions during training, so
errors compound and by step 4 the predicted latent is often garbage.

That's what caused V-driven MPC to hallucinate high-V states from
facing-the-wall poses in the last run.

Fix: train on K-step rollouts so gradients flow through the whole chain:

    z_1_pred = dyn(z_0,       a_0)
    z_2_pred = dyn(z_1_pred,  a_1)
    z_3_pred = dyn(z_2_pred,  a_2)
    z_4_pred = dyn(z_3_pred,  a_3)

Loss = mean MSE between z_k_pred and the real encoded z_k, summed over
all K steps.

Curriculum: warm up with K=1 for the first 25% of epochs (= same as the
old single-step loss; gets the basics right), then K=K_FULL. Training
multi-step from scratch is unstable because the model has no good z_1_pred
to start the chain from.

Saves dynamics_turn_human_ms.pth. Drop-in replacement for
dynamics_turn_human.pth in agent_map_step3.py.
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn3.pth"
OUT_WEIGHTS = "dynamics_turn_human_ms.pth"
DATA_DIRS = ["dataset/dataset_3_human2"]   # env3, no idle frames

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 512

# Rollout length — matches MPC_HORIZON in agent_map_step3.py.
# Train the model to be accurate over the same number of steps MPC
# actually rolls it.
K_FULL = 4

BATCH = 64
EPOCHS = 100
WARMUP_FRAC = 0.25            # first 25% of epochs: K=1 warm-up
LR = 1e-3
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


class MultiStepDataset(Dataset):
    """Each item is K+1 consecutive frames + K consecutive actions from a
    single episode. Skips any window containing an idle action so the
    3-action dynamics model only sees real transitions."""

    def __init__(self, data_dirs, K=K_FULL):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.K = K

        windows_obs, windows_acts = [], []
        total_eps = 0
        skipped_idle = 0

        for d in data_dirs:
            files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
            if not files:
                print(f"  WARNING: no episodes in {d}")
                continue
            total_eps += len(files)
            for fp in files:
                data = np.load(fp)
                obs = data["obs"]           # (T_obs, H, W, 3)
                actions = data["actions"]   # (T_act,)
                T_obs = obs.shape[0]
                # transition convention: obs[t] --action[t]--> obs[t+1]
                # need obs[t..t+K] (K+1 frames) and actions[t..t+K-1] (K).
                last_start = min(T_obs - K - 1, len(actions) - K)
                for t in range(0, last_start + 1):
                    a_win = actions[t:t + K]
                    if (a_win == 3).any():
                        skipped_idle += 1
                        continue
                    windows_obs.append(obs[t:t + K + 1])
                    windows_acts.append(a_win)
            print(f"  loaded {d}: {len(files)} episodes")

        if not windows_obs:
            raise RuntimeError("no valid windows from any data dir")

        self.obs = np.stack(windows_obs)              # (N, K+1, H, W, 3)
        self.actions = np.stack(windows_acts).astype(np.int64)  # (N, K)

        print(f"  total episodes: {total_eps}")
        print(f"  windows kept:   {len(self.actions)}"
              f"   (skipped {skipped_idle} containing idle)")
        print(f"  obs shape:      {self.obs.shape}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        f = torch.from_numpy(self.obs[idx]).float().div_(255.0).permute(0, 3, 1, 2)
        a = torch.from_numpy(self.actions[idx])
        return f, a


def rollout_loss(dyn, encoder, frames, actions, K):
    """Roll dynamics K steps from encoded frame[0]; return mean MSE
    across steps and the per-step MSE list (for diagnostics)."""
    B, Kp1, C, H, W = frames.shape
    flat = frames.reshape(B * Kp1, C, H, W)
    with torch.no_grad():
        z_all = encoder(flat).view(B, Kp1, -1)         # (B, K+1, D)

    z_pred = z_all[:, 0]                               # start state
    losses = []
    for k in range(K):
        a_k = actions[:, k]
        z_pred = dyn(z_pred, a_k)                      # roll forward
        z_true = z_all[:, k + 1]
        losses.append(F.mse_loss(z_pred, z_true))
    mean_loss = torch.stack(losses).mean()
    return mean_loss, [l.item() for l in losses]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE: {DEVICE}")
    print(f"K_FULL={K_FULL}  WARMUP_FRAC={WARMUP_FRAC}  EPOCHS={EPOCHS}")
    print("\nBuilding multi-step dataset...")
    ds = MultiStepDataset(DATA_DIRS, K=K_FULL)

    n = len(ds)
    n_val = max(1, int(VAL_FRAC * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
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
    train_losses = []
    val_losses_mean = []
    val_losses_per_step = []
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
        val_losses_per_step.append(last_steps)

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
    plt.plot(train_losses, label="train (mean over steps)")
    plt.plot(val_losses_mean, label="val (mean over steps)")
    plt.axvline(warmup_epochs - 0.5, color="grey", linestyle="--",
                label=f"K: 1 -> {K_FULL}")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(f"Multi-step dynamics (warmup K=1 -> K={K_FULL})")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dynamics_turn_human_ms_loss.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
