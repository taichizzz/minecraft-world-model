"""
P2a — Stronger dynamics on the FROZEN ae_multienv latent.

Why: the action-conditioned diagnostic (dynamics_action_diag.py / fig6) showed
the current dynamics_multienv.pth has latent error ~= latent displacement
(err/disp ~ 1.0-1.2) for EVERY action, i.e. it barely beats "predict no change".
The open question is whether that is a *training* limitation or a *latent*
limitation. This trainer pushes the dynamics as hard as is reasonable on the
SAME frozen latent (same encoder, same architecture, drop-in weights):

  * cache per-frame latents ONCE (no re-encoding every epoch -> ~10x faster,
    so we can afford a longer curriculum)
  * action-frequency-WEIGHTED rollout loss: turns are ~6x rarer than moves in
    the human data, so an unweighted MSE lets the model ignore them. Balanced
    class weights force it to actually model turns.
  * longer K curriculum (1 -> 2 -> 4 -> 6) with LR decay
  * SAME DynamicsTurningMLP(hidden=512) so it drops into agent_map_step3.py

Output: dynamics_multienv_v2.pth

If, AFTER this, `dynamics_action_diag.py --dyn dynamics_multienv_v2.pth --tag v2`
still shows err/disp ~ 1, the reconstruction latent itself is unpredictable
(the wall), which justifies the P2b predictable-latent rebuild rather than more
dynamics tuning.
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "ae_multienv.pth"
OUT_WEIGHTS = "dynamics_multienv_v2.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 512

K_MAX = 6
# (start_fraction, K) — K active once epoch >= frac*EPOCHS
K_SCHEDULE = [(0.00, 1), (0.20, 2), (0.45, 4), (0.70, 6)]

BATCH = 128
EPOCHS = 120
LR = 1e-3
LR_MIN = 1e-5
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
ENC_CHUNK = 512
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def encode_frames(encoder, frames_u8):
    """frames_u8: (T,H,W,C) uint8 -> (T,latent) float32 on CPU."""
    N = len(frames_u8)
    out = torch.empty((N, LATENT_DIM), dtype=torch.float32)
    for i in range(0, N, ENC_CHUNK):
        chunk = frames_u8[i:i + ENC_CHUNK]
        t = (torch.from_numpy(chunk).float().div_(255.0)
             .permute(0, 3, 1, 2).to(DEVICE))
        out[i:i + ENC_CHUNK] = encoder(t).cpu()
    return out


def build_windows(encoder):
    """Encode every episode once, then slice K_MAX+1 latent windows + K_MAX
    actions. Skips windows containing idle (action==3)."""
    z_windows, a_windows = [], []
    n_eps = skipped_idle = 0
    act_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    for d in DATASET_DIRS:
        files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
        n_dir_windows = 0
        for fp in files:
            data = np.load(fp)
            obs = data["obs"]
            actions = data["actions"].astype(np.int64)
            T_obs = obs.shape[0]
            last_start = min(T_obs - K_MAX - 1, len(actions) - K_MAX)
            if last_start < 0:
                continue
            n_eps += 1
            z = encode_frames(encoder, obs)            # (T,latent) cpu
            for t in range(0, last_start + 1):
                a_win = actions[t:t + K_MAX]
                if (a_win == 3).any():
                    skipped_idle += 1
                    continue
                z_windows.append(z[t:t + K_MAX + 1].numpy())
                a_windows.append(a_win)
                act_counts += np.bincount(a_win, minlength=NUM_ACTIONS)
                n_dir_windows += 1
        print(f"  {d}: {n_dir_windows} windows")

    if not z_windows:
        raise RuntimeError("No valid windows.")

    Z = torch.from_numpy(np.stack(z_windows)).float()   # (N,K+1,latent)
    A = torch.from_numpy(np.stack(a_windows)).long()    # (N,K)
    print(f"  episodes={n_eps}  windows={len(A)}  (skipped {skipped_idle} idle)")
    print(f"  action counts (in windows): "
          + "  ".join(f"a{i}={act_counts[i]}" for i in range(NUM_ACTIONS)))

    # balanced class weights: total / (num_actions * count)
    total = act_counts.sum()
    w = total / (NUM_ACTIONS * np.maximum(act_counts, 1))
    aw = torch.from_numpy(w).float()
    print(f"  action loss weights: "
          + "  ".join(f"a{i}={w[i]:.3f}" for i in range(NUM_ACTIONS)))
    return Z, A, aw


def current_K(epoch):
    K = 1
    for frac, k in K_SCHEDULE:
        if epoch >= frac * EPOCHS:
            K = k
    return K


def rollout_loss(dyn, z_win, a_win, K, act_w):
    """Weighted multi-step rollout MSE.
    z_win:(B,K_MAX+1,L)  a_win:(B,K_MAX)  act_w:(NUM_ACTIONS,)
    Returns scalar loss + per-step plain MSE for logging."""
    z_pred = z_win[:, 0]
    step_mse = []
    losses = []
    for k in range(K):
        a_k = a_win[:, k]
        z_pred = dyn(z_pred, a_k)
        z_true = z_win[:, k + 1]
        per = ((z_pred - z_true) ** 2).mean(dim=1)         # (B,)
        wk = act_w[a_k]                                     # (B,)
        losses.append((per * wk).sum() / wk.sum())
        step_mse.append(per.mean().item())
    loss = torch.stack(losses).mean()
    return loss, step_mse


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  K_MAX={K_MAX}  EPOCHS={EPOCHS}  out={OUT_WEIGHTS}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("\nEncoding + windowing (cached latents)...")
    Z, A, act_w = build_windows(ae.encoder)
    act_w = act_w.to(DEVICE)

    ds = TensorDataset(Z, A)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(ds, [n_tr, n_val], generator=g)
    print(f"  train={len(tr)}  val={len(va)}")

    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False)

    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS,
                             hidden=HIDDEN).to(DEVICE)
    opt = optim.Adam(dyn.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    train_losses, val_losses = [], []
    best_val = float("inf")
    print(f"\nK schedule: {K_SCHEDULE}\n")
    for epoch in range(EPOCHS):
        K = current_K(epoch)

        dyn.train()
        tot = nb = 0
        for z_win, a_win in tl:
            z_win = z_win.to(DEVICE); a_win = a_win.to(DEVICE)
            loss, _ = rollout_loss(dyn, z_win, a_win, K, act_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)
        train_losses.append(tr_loss)

        dyn.eval()
        vtot = vn = 0
        last_steps = None
        with torch.no_grad():
            for z_win, a_win in vl:
                z_win = z_win.to(DEVICE); a_win = a_win.to(DEVICE)
                loss, step_mse = rollout_loss(dyn, z_win, a_win, K, act_w)
                vtot += loss.item(); vn += 1
                last_steps = step_mse
        v_loss = vtot / max(vn, 1)
        val_losses.append(v_loss)
        sched.step()

        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            torch.save(dyn.state_dict(), OUT_WEIGHTS)
            marker = "  <- best"
        step_str = "  ".join(f"k={i+1}:{v:.4f}"
                             for i, v in enumerate(last_steps or []))
        print(f"epoch {epoch:03d} K={K} lr={sched.get_last_lr()[0]:.2e} | "
              f"train {tr_loss:.5f} | val {v_loss:.5f}  [{step_str}]{marker}")

    print(f"\nBest val weighted MSE: {best_val:.6f}")
    print(f"Saved: {OUT_WEIGHTS}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    for frac, k in K_SCHEDULE[1:]:
        plt.axvline(frac * EPOCHS - 0.5, color="grey", linestyle="--", alpha=0.5)
    plt.xlabel("epoch"); plt.ylabel("weighted MSE")
    plt.title("Dynamics v2 (cached latent, action-weighted, K 1->6)")
    plt.legend(); plt.tight_layout()
    out = os.path.join(OUT_DIR, "dynamics_multienv_v2_loss.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved plot: {out}")
    print("\nNext: py -3.9 world_model/dynamics_action_diag.py "
          "--dyn dynamics_multienv_v2.pth --tag v2")


if __name__ == "__main__":
    main()
