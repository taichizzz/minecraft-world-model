"""
train_value_dist_aug.py — appearance-robust value head (SAFE path).

Last night's joint encoder+augmentation retrain diverged (unbounded latent
runaway). This is the safe alternative: keep the encoder FROZEN (ae_predictive)
and train ONLY the value head, feeding it appearance-jittered frames
(brightness/contrast/color/noise). The head learns to read goal-distance
robustly across photometric shifts. A frozen encoder + small supervised MLP
CANNOT diverge — it's exactly train_value_distance with jittered inputs.

Reports clean-val AND augmented-val correlation each epoch. The AUG corr is the
robustness number: if it stays high, the head reads the goal through lighting/
color changes. Output: value_head_dist_pred_aug.pth (pairs with the p2b_vc stack).

Fast (minutes). Run a short check first, then the full run:
    py -3.9 world_model/train_value_dist_aug.py --epochs 10
    py -3.9 world_model/train_value_dist_aug.py
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import AutoEncoder
from train_value_head import ValueHead
from train_value_distance import load_frames_labels   # proven label builder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
VAL_HIDDEN = 256
AE_WEIGHTS = "ae_predictive.pth"
OUT = "value_head_dist_pred_aug.pth"
AUG_BRIGHT, AUG_CONTRAST, AUG_COLOR, AUG_NOISE = 0.4, 0.3, 0.2, 0.03
BATCH = 256
EPOCHS = 60
LR = 1e-3
VAL_FRAC = 0.1
SEED = 42


def augment(x, scale):
    """Appearance jitter on (B,3,64,64) in [0,1], one draw per image."""
    B = x.shape[0]; dev = x.device

    def u(lo, hi, shape):
        return torch.empty(shape, device=dev).uniform_(lo, hi)

    x = x * u(1 - AUG_BRIGHT * scale, 1 + AUG_BRIGHT * scale, (B, 1, 1, 1))
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * u(1 - AUG_CONTRAST * scale, 1 + AUG_CONTRAST * scale, (B, 1, 1, 1)) + m
    x = x * u(1 - AUG_COLOR * scale, 1 + AUG_COLOR * scale, (B, 3, 1, 1))
    if AUG_NOISE * scale > 0:
        x = x + torch.randn_like(x) * (AUG_NOISE * scale)
    return x.clamp_(0.0, 1.0)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float((a * b).sum().item() / d) if d > 1e-8 else 0.0


def prep(xb):                              # uint8 (B,64,64,3) -> float (B,3,64,64)
    return xb.float().div(255.0).permute(0, 3, 1, 2).to(DEVICE)


def main():
    global EPOCHS, BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ae", default=AE_WEIGHTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--aug", type=float, default=1.0,
                    help="augmentation strength (0 = no jitter, clean baseline)")
    args = ap.parse_args()
    EPOCHS, BATCH = args.epochs, args.batch

    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  AE(frozen)={args.ae}  aug={args.aug}  out={args.out}")

    frames, labels, d_max = load_frames_labels("distance", 0.95)
    X = torch.from_numpy(frames)           # uint8 (N,64,64,3)
    Y = torch.from_numpy(labels)           # (N,)
    ds = TensorDataset(X, Y)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False)
    print(f"  frames={len(ds)}  train={len(tr)}  val={len(va)}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(args.ae, map_location=DEVICE)); ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    head = ValueHead(latent_dim=LATENT_DIM, hidden=VAL_HIDDEN).to(DEVICE)
    opt = optim.Adam(head.parameters(), lr=LR)

    best = float("inf")
    print(f"\nTraining {EPOCHS} epochs (frozen encoder, jittered inputs)...\n")
    for ep in range(EPOCHS):
        head.train()
        for xb, yb in tl:
            x = augment(prep(xb), args.aug)
            with torch.no_grad():
                z = ae.encoder(x)
            loss = F.mse_loss(head(z).reshape(-1), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        pc, pa, yy = [], [], []
        with torch.no_grad():
            for xb, yb in vl:
                x = prep(xb)
                pc.append(head(ae.encoder(x)).reshape(-1).cpu())
                pa.append(head(ae.encoder(augment(x, args.aug))).reshape(-1).cpu())
                yy.append(yb)
        pc = torch.cat(pc); pa = torch.cat(pa); yy = torch.cat(yy)
        corr_clean = pearson(pc, yy); corr_aug = pearson(pa, yy)
        vmse = F.mse_loss(pc, yy).item()
        mark = ""
        if vmse < best:
            best = vmse
            torch.save(head.state_dict(), args.out)
            mark = "  <- best"
        print(f"ep {ep:02d} | clean corr {corr_clean:+.3f} | AUG corr {corr_aug:+.3f} "
              f"| val MSE {vmse:.5f}{mark}")

    print(f"\nSaved {args.out}.")
    print("AUG corr is the robustness number: high = reads goal-distance through "
          "lighting/color shifts. Compare to a clean-trained head (--aug 0) whose "
          "AUG corr should be lower.")


if __name__ == "__main__":
    main()
