"""
P2b — Predictable-latent rebuild (the real world-model fix).

P2a proved the wall is the LATENT, not dynamics training: a harder-trained
dynamics on the frozen ae_multienv latent still has err/disp ~= 1 on every
action. That latent was trained for RECONSTRUCTION only, so it packs
high-frequency texture that isn't a function of (z, a) and can't be predicted.

This trainer stops freezing the encoder and trains encoder + decoder + dynamics
JOINTLY so the latent is shaped to be predictable (Dreamer-style). Per K-step
window:

  z_t      = enc(o_t)                              (all frames)
  z^_{k+1} = dyn(z^_k, a_k)                         (imagined rollout, z^_0=z_0)

  loss = W_RECON * MSE(dec(z_t), o_t)              # anti-collapse anchor
       + W_FWD   * MSE(z^_{k+1}, sg(z_{k+1}))      # latent-consistency (the fix)
       + W_DEC   * MSE(dec(z^_{k+1}), o_{k+1})      # imagined latent must decode

Anti-collapse: without the recon term the encoder could map every frame to a
constant (perfectly predictable, useless). The stop-grad sg(.) on the target
latent keeps the encoder from cheating by moving the target toward the
prediction (BYOL/Dreamer trick).

The live gate to watch: 1-step err/disp on val — the exact quantity P2a showed
stuck at ~1. If it falls well below 1 while recon stays low, the rebuild worked.

Outputs (NEW filenames — does NOT touch ae_multienv/dynamics_multienv, so a
running eval is unaffected):
  - ae_predictive.pth        (full AutoEncoder: drop-in for AE_WEIGHTS)
  - dynamics_predictive.pth  (drop-in for DYN_WEIGHTS)

IMPORTANT follow-on: the new latent invalidates value_head_dist.pth /
value_head_stepstogo.pth. Retrain the value head on the new latent before
using the agent (the value trainers need an --ae override pointing here).

Gate after training (also needs an --ae override on the diag):
  py -3.9 world_model/dynamics_action_diag.py --ae ae_predictive.pth \
        --dyn dynamics_predictive.pth --tag pred
"""
import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMSTART_AE = "ae_multienv.pth"        # set None to train encoder from scratch
OUT_AE = "ae_predictive.pth"
OUT_DYN = "dynamics_predictive.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 512

K_MAX = 4
K_SCHEDULE = [(0.00, 1), (0.20, 2), (0.45, 4)]

# loss weights (recon : forward-latent : decoded-rollout)
W_RECON = 1.0
W_FWD = 1.0
W_DEC = 1.0

BATCH = 64
EPOCHS = 200
LR = 1e-3
LR_MIN = 1e-5
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


class WindowDataset(Dataset):
    """Per-episode frames held once; windows indexed by (ep, t) to avoid the
    ~Kx memory blow-up of stacking overlapping copies. Skips idle (a==3)."""

    def __init__(self, dirs, K=K_MAX):
        self.K = K
        self.episodes = []          # list of (obs uint8, actions int64)
        self.index = []             # (ep_idx, t)
        skipped_idle = 0
        for d in dirs:
            for fp in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
                data = np.load(fp)
                obs = data["obs"]
                actions = data["actions"].astype(np.int64)
                T_obs = obs.shape[0]
                last_start = min(T_obs - K - 1, len(actions) - K)
                if last_start < 0:
                    continue
                ei = len(self.episodes)
                self.episodes.append((obs, actions))
                for t in range(0, last_start + 1):
                    if (actions[t:t + K] == 3).any():
                        skipped_idle += 1
                        continue
                    self.index.append((ei, t))
        if not self.index:
            raise RuntimeError("No valid windows.")
        print(f"  episodes={len(self.episodes)}  windows={len(self.index)} "
              f"(skipped {skipped_idle} idle)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        obs, actions = self.episodes[ei]
        f = (torch.from_numpy(obs[t:t + self.K + 1]).float().div_(255.0)
             .permute(0, 3, 1, 2))                 # (K+1,3,64,64)
        a = torch.from_numpy(actions[t:t + self.K])
        return f, a


def current_K(epoch):
    K = 1
    for frac, k in K_SCHEDULE:
        if epoch >= frac * EPOCHS:
            K = k
    return K


def compute_losses(ae, dyn, frames, acts, K):
    """frames:(B,K_MAX+1,3,64,64) acts:(B,K_MAX). Uses first K steps.
    Returns total loss + dict of components + 1-step err/disp proxy."""
    B = frames.shape[0]
    o = frames[:, :K + 1]                                  # (B,K+1,3,64,64)
    flat = o.reshape(B * (K + 1), 3, 64, 64)
    z = ae.encoder(flat).view(B, K + 1, LATENT_DIM)        # (B,K+1,L)

    # reconstruction of every real frame (anti-collapse anchor)
    recon_flat = ae.decoder(z.reshape(B * (K + 1), LATENT_DIM))
    recon_loss = F.mse_loss(recon_flat, flat)

    # imagined rollout
    fwd_terms, dec_terms = [], []
    z_hat = z[:, 0]
    err1 = disp1 = None
    for k in range(K):
        z_hat = dyn(z_hat, acts[:, k])
        z_tgt = z[:, k + 1]
        fwd_terms.append(F.mse_loss(z_hat, z_tgt.detach()))    # stop-grad target
        dec_terms.append(F.mse_loss(ae.decoder(z_hat), o[:, k + 1]))
        if k == 0:
            with torch.no_grad():
                err1 = torch.norm(z_hat - z_tgt, dim=1).mean().item()
                disp1 = torch.norm(z_tgt - z[:, 0], dim=1).mean().item()
    fwd_loss = torch.stack(fwd_terms).mean()
    dec_loss = torch.stack(dec_terms).mean()

    total = W_RECON * recon_loss + W_FWD * fwd_loss + W_DEC * dec_loss
    comp = {"recon": recon_loss.item(), "fwd": fwd_loss.item(),
            "dec": dec_loss.item(), "err1": err1, "disp1": disp1}
    return total, comp


def main():
    global W_RECON, W_FWD, W_DEC, EPOCHS, BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="lower this (e.g. 32) if you OOM on 6GB VRAM")
    ap.add_argument("--scratch", action="store_true",
                    help="train encoder from scratch instead of warm-starting")
    ap.add_argument("--w-recon", type=float, default=W_RECON)
    ap.add_argument("--w-fwd", type=float, default=W_FWD)
    ap.add_argument("--w-dec", type=float, default=W_DEC)
    args = ap.parse_args()
    W_RECON, W_FWD, W_DEC = args.w_recon, args.w_fwd, args.w_dec
    EPOCHS = args.epochs
    BATCH = args.batch

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  K_MAX={K_MAX}  EPOCHS={EPOCHS}")
    print(f"loss weights  recon={W_RECON}  fwd={W_FWD}  dec={W_DEC}")

    print("\nBuilding windows...")
    ds = WindowDataset(DATASET_DIRS, K=K_MAX)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(ds, [n_tr, n_val], generator=g)
    print(f"  train={len(tr)}  val={len(va)}")
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=0)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    if not args.scratch and WARMSTART_AE and os.path.exists(WARMSTART_AE):
        ae.load_state_dict(torch.load(WARMSTART_AE, map_location=DEVICE))
        print(f"warm-started AE from {WARMSTART_AE}")
    else:
        print("AE trained from scratch")
    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS,
                             hidden=HIDDEN).to(DEVICE)

    params = list(ae.parameters()) + list(dyn.parameters())
    opt = optim.Adam(params, lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    tr_hist, va_hist, ratio_hist = [], [], []
    best_val = float("inf")
    print(f"\nK schedule: {K_SCHEDULE}\n")
    for epoch in range(EPOCHS):
        K = current_K(epoch)

        ae.train(); dyn.train()
        tot = nb = 0
        for frames, acts in tl:
            frames = frames.to(DEVICE); acts = acts.to(DEVICE)
            loss, _ = compute_losses(ae, dyn, frames, acts, K)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)
        tr_hist.append(tr_loss)

        ae.eval(); dyn.eval()
        vtot = vn = 0
        agg = {"recon": 0.0, "fwd": 0.0, "dec": 0.0, "err1": 0.0, "disp1": 0.0}
        with torch.no_grad():
            for frames, acts in vl:
                frames = frames.to(DEVICE); acts = acts.to(DEVICE)
                loss, comp = compute_losses(ae, dyn, frames, acts, K)
                vtot += loss.item(); vn += 1
                for kk in agg:
                    agg[kk] += comp[kk]
        v_loss = vtot / max(vn, 1)
        for kk in agg:
            agg[kk] /= max(vn, 1)
        ratio = agg["err1"] / max(agg["disp1"], 1e-6)
        va_hist.append(v_loss); ratio_hist.append(ratio)
        sched.step()

        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            torch.save(ae.state_dict(), OUT_AE)
            torch.save(dyn.state_dict(), OUT_DYN)
            marker = "  <- best (saved)"
        print(f"epoch {epoch:03d} K={K} lr={sched.get_last_lr()[0]:.2e} | "
              f"train {tr_loss:.5f} val {v_loss:.5f} | "
              f"recon {agg['recon']:.5f} fwd {agg['fwd']:.5f} dec {agg['dec']:.5f} | "
              f"err/disp(1) {ratio:.3f}{marker}")

    print(f"\nBest val total: {best_val:.6f}")
    print(f"Saved: {OUT_AE}  {OUT_DYN}")
    print(f"Final 1-step err/disp: {ratio_hist[-1]:.3f}  "
          f"(P2a baseline ~1.1-1.4; want well below 1)")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(tr_hist, label="train total", color="tab:blue")
    ax1.plot(va_hist, label="val total", color="tab:orange")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("total loss")
    for frac, k in K_SCHEDULE[1:]:
        ax1.axvline(frac * EPOCHS - 0.5, color="grey", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ratio_hist, label="err/disp (1-step)", color="tab:red", alpha=0.7)
    ax2.axhline(1.0, color="tab:red", linestyle=":", alpha=0.5)
    ax2.set_ylabel("err/disp (1-step)", color="tab:red")
    ax1.legend(loc="upper right"); ax1.set_title("P2b predictable-latent rebuild")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dynamics_predictive_loss.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved plot: {out}")
    print("\nNext: retrain value head on the new latent, then gate with "
          "dynamics_action_diag.py --ae ae_predictive.pth --dyn dynamics_predictive.pth --tag pred")


if __name__ == "__main__":
    main()
