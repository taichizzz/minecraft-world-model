"""
Appearance-augmented predictive AE  (for FACTOR generalization).

Your encoder/value/dynamics only ever saw env1/2/3, which share wall/floor
materials -> the world model is likely overfit to those textures/lighting and
would struggle in visually-different rooms (and on a real robot). This trains an
APPEARANCE-INVARIANT predictive latent: the encoder sees a brightness/contrast/
color/noise-jittered frame but must reconstruct the CLEAN scene and predict the
CLEAN-latent future. So it learns to see *through* appearance to the content.

Built on the P2b predictive recipe (joint encoder+decoder+dynamics), warm-started
from ae_predictive.pth + dynamics_predictive.pth. Per K-step window:

   o            = clean frames
   o_aug        = appearance-jitter(o)              # encoder input
   z            = enc(o_aug)                          # what the agent must encode
   z_clean      = sg(enc(o))                          # stable target (clean)

   recon = MSE(dec(z), o)                  # reconstruct CLEAN from jittered input
   fwd   = MSE(dyn-rollout(z), z_clean)    # predict the clean-latent future
   dec   = MSE(dec(rollout), o_next)       # imagined latent decodes to clean scene

Outputs (NEW names -- does NOT touch your current p2b / p2b_vc stack):
  ae_predictive_aug.pth, dynamics_predictive_aug.pth

Morning follow-up (fast): retrain value head + value-consistent dynamics on the
new latent, then test on visually-different rooms.
  py -3.9 world_model/train_value_distance.py --ae ae_predictive_aug.pth --out value_head_dist_aug.pth
  py -3.9 world_model/train_dynamics_value_consistent.py --ae ae_predictive_aug.pth --value value_head_dist_aug.pth --dyn-init dynamics_predictive_aug.pth --out dynamics_predictive_aug_vc.pth

Watch: err/disp(1) should fall like P2b did; recon should stay LOW (if recon
blows up, augmentation is too strong -> lower --aug).
"""
import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMSTART_AE = "ae_predictive.pth"
WARMSTART_DYN = "dynamics_predictive.pth"
OUT_AE = "ae_predictive_aug.pth"
OUT_DYN = "dynamics_predictive_aug.pth"

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

W_RECON, W_FWD, W_DEC, W_INV = 1.0, 1.0, 1.0, 1.0
# appearance-jitter strengths (scaled by --aug)
AUG_BRIGHT, AUG_CONTRAST, AUG_COLOR, AUG_NOISE = 0.4, 0.3, 0.2, 0.03

BATCH = 64
EPOCHS = 120
LR = 5e-4
LR_MIN = 1e-5
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


class WindowDataset(Dataset):
    """Per-episode frames; windows indexed by (ep,t). Skips idle (a==3)."""
    def __init__(self, dirs, K=K_MAX):
        self.K = K
        self.episodes = []
        self.index = []
        skipped = 0
        for d in dirs:
            for fp in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
                data = np.load(fp)
                obs = data["obs"]
                actions = data["actions"].astype(np.int64)
                last_start = min(obs.shape[0] - K - 1, len(actions) - K)
                if last_start < 0:
                    continue
                ei = len(self.episodes)
                self.episodes.append((obs, actions))
                for t in range(0, last_start + 1):
                    if (actions[t:t + K] == 3).any():
                        skipped += 1
                        continue
                    self.index.append((ei, t))
        if not self.index:
            raise RuntimeError("No valid windows.")
        print(f"  episodes={len(self.episodes)}  windows={len(self.index)} "
              f"(skipped {skipped} idle)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        obs, actions = self.episodes[ei]
        f = (torch.from_numpy(obs[t:t + self.K + 1]).float().div_(255.0)
             .permute(0, 3, 1, 2))                 # (K+1,3,64,64)
        a = torch.from_numpy(actions[t:t + self.K])
        return f, a


def augment(o, scale):
    """Appearance jitter on (B,K+1,3,64,64) in [0,1]. One brightness/contrast/
    color draw per window (lighting is consistent within a short window);
    per-pixel noise. scale in [0,1] multiplies the strengths."""
    B = o.shape[0]
    dev = o.device

    def u(lo, hi, shape):
        return torch.empty(shape, device=dev).uniform_(lo, hi)

    b = u(1 - AUG_BRIGHT * scale, 1 + AUG_BRIGHT * scale, (B, 1, 1, 1, 1))
    x = o * b
    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    c = u(1 - AUG_CONTRAST * scale, 1 + AUG_CONTRAST * scale, (B, 1, 1, 1, 1))
    x = (x - mean) * c + mean
    col = u(1 - AUG_COLOR * scale, 1 + AUG_COLOR * scale, (B, 1, 3, 1, 1))
    x = x * col
    if AUG_NOISE * scale > 0:
        x = x + torch.randn_like(x) * (AUG_NOISE * scale)
    return x.clamp_(0.0, 1.0)


def current_K(epoch):
    K = 1
    for frac, k in K_SCHEDULE:
        if epoch >= frac * EPOCHS:
            K = k
    return K


def compute_losses(ae, dyn, frames, acts, K, aug_scale):
    """Dynamics trains on the CLEAN latent path (exactly P2b -> stable). A
    SEPARATE invariance term pulls augmented-frame encodings onto the clean
    ones, so appearance shifts encode the same. (The earlier 'roll from aug,
    target clean' design let the two latent paths diverge -> fwd exploded.)"""
    B = frames.shape[0]
    o = frames[:, :K + 1]                                   # clean (B,K+1,3,64,64)
    flat = o.reshape(B * (K + 1), 3, 64, 64)
    z = ae.encoder(flat).view(B, K + 1, LATENT_DIM)         # clean latents (main path)
    recon_loss = F.mse_loss(ae.decoder(z.reshape(B * (K + 1), LATENT_DIM)), flat)

    fwd_terms, dec_terms = [], []
    z_hat = z[:, 0]
    err1 = disp1 = None
    for k in range(K):
        z_hat = dyn(z_hat, acts[:, k])
        z_tgt = z[:, k + 1]
        fwd_terms.append(F.mse_loss(z_hat, z_tgt.detach()))    # stop-grad clean target
        dec_terms.append(F.mse_loss(ae.decoder(z_hat), o[:, k + 1]))
        if k == 0:
            with torch.no_grad():
                err1 = torch.norm(z_hat - z_tgt, dim=1).mean().item()
                disp1 = torch.norm(z_tgt - z[:, 0], dim=1).mean().item()
    fwd_loss = torch.stack(fwd_terms).mean()
    dec_loss = torch.stack(dec_terms).mean()

    # appearance invariance: aug(frame) must encode like the clean frame
    if aug_scale > 0:
        o_aug = augment(o, aug_scale)
        z_aug = ae.encoder(o_aug.reshape(B * (K + 1), 3, 64, 64)).view(B, K + 1, LATENT_DIM)
        inv_loss = F.mse_loss(z_aug, z.detach())
    else:
        inv_loss = torch.zeros((), device=flat.device)

    total = (W_RECON * recon_loss + W_FWD * fwd_loss + W_DEC * dec_loss
             + W_INV * inv_loss)
    comp = {"recon": recon_loss.item(), "fwd": fwd_loss.item(), "dec": dec_loss.item(),
            "inv": inv_loss.item(), "err1": err1, "disp1": disp1}
    return total, comp


def main():
    global EPOCHS, BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--aug", type=float, default=1.0,
                    help="augmentation strength multiplier (0=off, 1=full)")
    ap.add_argument("--scratch", action="store_true",
                    help="train encoder from scratch instead of warm-starting")
    args = ap.parse_args()
    EPOCHS, BATCH = args.epochs, args.batch

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  EPOCHS={EPOCHS}  aug={args.aug}  K_MAX={K_MAX}")
    print(f"aug strengths x{args.aug}: bright={AUG_BRIGHT} contrast={AUG_CONTRAST} "
          f"color={AUG_COLOR} noise={AUG_NOISE}")

    print("\nBuilding windows...")
    ds = WindowDataset(DATASET_DIRS, K=K_MAX)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  train={len(tr)}  val={len(va)}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    if not args.scratch and os.path.exists(WARMSTART_AE):
        ae.load_state_dict(torch.load(WARMSTART_AE, map_location=DEVICE))
        print(f"warm-started AE from {WARMSTART_AE}")
    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS,
                             hidden=HIDDEN).to(DEVICE)
    if not args.scratch and os.path.exists(WARMSTART_DYN):
        dyn.load_state_dict(torch.load(WARMSTART_DYN, map_location=DEVICE))
        print(f"warm-started dynamics from {WARMSTART_DYN}")

    params = list(ae.parameters()) + list(dyn.parameters())
    opt = optim.Adam(params, lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)

    tr_hist, va_hist, ratio_hist = [], [], []
    best = float("inf")
    print(f"\nK schedule {K_SCHEDULE}  (checkpoint only at K={K_MAX})\n")
    for epoch in range(EPOCHS):
        K = current_K(epoch)
        ae.train(); dyn.train()
        tot = nb = 0
        for frames, acts in tl:
            frames = frames.to(DEVICE); acts = acts.to(DEVICE)
            loss, _ = compute_losses(ae, dyn, frames, acts, K, args.aug)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        tr_hist.append(tot / max(nb, 1))

        ae.eval(); dyn.eval()
        vtot = vn = 0
        agg = {"recon": 0.0, "fwd": 0.0, "dec": 0.0, "inv": 0.0, "err1": 0.0, "disp1": 0.0}
        with torch.no_grad():
            for frames, acts in vl:
                frames = frames.to(DEVICE); acts = acts.to(DEVICE)
                loss, comp = compute_losses(ae, dyn, frames, acts, K, args.aug)
                vtot += loss.item(); vn += 1
                for kk in agg:
                    agg[kk] += comp[kk]
        v_loss = vtot / max(vn, 1)
        for kk in agg:
            agg[kk] /= max(vn, 1)
        ratio = agg["err1"] / max(agg["disp1"], 1e-6)
        va_hist.append(v_loss); ratio_hist.append(ratio)
        sched.step()

        mark = ""
        if K == K_MAX and v_loss < best:      # only checkpoint the full-curriculum model
            best = v_loss
            torch.save(ae.state_dict(), OUT_AE)
            torch.save(dyn.state_dict(), OUT_DYN)
            mark = "  <- best (saved)"
        print(f"ep {epoch:03d} K={K} lr={sched.get_last_lr()[0]:.2e} | "
              f"val {v_loss:.5f} | recon {agg['recon']:.5f} fwd {agg['fwd']:.5f} "
              f"dec {agg['dec']:.5f} inv {agg['inv']:.5f} | err/disp(1) {ratio:.3f}{mark}")

    print(f"\nBest val (K={K_MAX}): {best:.6f}\nSaved: {OUT_AE}  {OUT_DYN}")
    print("If recon stayed low and err/disp fell, the appearance-invariant latent "
          "trained. Next: retrain value head + value-consistent dynamics on it "
          "(see header), then test on visually-different rooms.")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(tr_hist, label="train total", color="tab:blue")
    ax1.plot(va_hist, label="val total", color="tab:orange")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("total loss")
    for frac, k in K_SCHEDULE[1:]:
        ax1.axvline(frac * EPOCHS - 0.5, color="grey", ls="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ratio_hist, label="err/disp (1-step)", color="tab:red", alpha=0.7)
    ax2.axhline(1.0, color="tab:red", ls=":", alpha=0.5)
    ax2.set_ylabel("err/disp (1-step)", color="tab:red")
    ax1.legend(loc="upper right"); ax1.set_title("Appearance-augmented predictive AE")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ae_predictive_aug_loss.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
