"""
Track B — Goal-directed value head.

The current value head (value_head_multienv.pth) predicts "is the diamond
visible" (cyan-pixel count). The Track A diagnosis showed that as a navigator
it is useless: world-model-only control reaches the goal 0% of the time,
because V is ~0 whenever the diamond is not in view — i.e. during all of
exploration.

This trainer builds a *navigation* value head instead, using the GROUND-TRUTH
positions already stored in the human datasets. Two label modes:

  --mode distance  (default)
      V(z_t) = 1 - clip( dist(pos_t, GOAL) / D_MAX , 0, 1 )
      A dense "how close am I to the goal" signal. Trained WITH position
      (offline, allowed) but consumed FROM PIXELS at test time (no position
      needed online) — the bridge to the no-position experiment.

  --mode rtg
      V(z_t) = gamma^(T-1-t)   for successful episodes only.
      Return-to-go toward a terminal +1 at the diamond. Encodes "how many
      actions until success under the demo policy".

Key point: the goal (diamond block at x=4, z=4 -> center 4.5,4.5) is the same
in env1/env2/env3, so all three human datasets share one label definition.

Saves value_head_dist.pth (kept separate so we can A/B vs value_head_multienv.pth).
Architecture is the SAME ValueHead the live agent loads, so the weights drop in
by just pointing VALUE_WEIGHTS at the new file.
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import AutoEncoder
from train_value_head import ValueHead   # same class the agent imports

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
HIDDEN = 256
AE_WEIGHTS = "ae_multienv.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

GOAL = np.array([4.5, 4.5], dtype=np.float32)   # diamond block center
SUCCESS_RADIUS = 2.0                            # Manhattan, matches agent

BATCH = 256
EPOCHS = 60
LR = 1e-3
VAL_FRAC = 0.1
SEED = 42
ENC_CHUNK = 512


def manhattan(p):
    return np.abs(p[..., 0] - GOAL[0]) + np.abs(p[..., 1] - GOAL[1])


def euclid(p):
    return np.sqrt((p[..., 0] - GOAL[0]) ** 2 + (p[..., 1] - GOAL[1]) ** 2)


def load_frames_labels(mode, gamma):
    frames, labels = [], []
    n_eps = n_success = 0
    all_dist = []

    for d in DATASET_DIRS:
        files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
        for fp in files:
            data = np.load(fp)
            obs = data["obs"]
            pos = data["positions"].astype(np.float32)
            T = obs.shape[0]
            if T < 2:
                continue
            n_eps += 1

            final_man = manhattan(pos[-1])
            success = final_man < SUCCESS_RADIUS
            if success:
                n_success += 1

            if mode == "distance":
                d_e = euclid(pos)                       # (T,)
                all_dist.append(d_e)
                frames.append(obs)
                labels.append(d_e)                      # raw dist; normalize later
            elif mode == "rtg":
                if not success:
                    continue                            # only learn from wins
                t = np.arange(T, dtype=np.float32)
                rtg = gamma ** (T - 1 - t)              # 1.0 at goal, decays back
                frames.append(obs)
                labels.append(rtg)

    if not frames:
        raise RuntimeError("No frames collected — check dataset paths.")

    frames = np.concatenate(frames, axis=0)             # (N,64,64,3)
    labels = np.concatenate(labels, axis=0).astype(np.float32)

    if mode == "distance":
        all_dist = np.concatenate(all_dist)
        d_max = float(np.percentile(all_dist, 99))      # robust max
        labels = 1.0 - np.clip(labels / d_max, 0.0, 1.0)
        print(f"  distance mode: D_MAX(99pct)={d_max:.2f}")
    else:
        d_max = None

    print(f"  episodes: {n_eps}  successful: {n_success} "
          f"({n_success/max(n_eps,1):.0%})")
    print(f"  frames: {len(frames)}  "
          f"label mean={labels.mean():.3f} min={labels.min():.3f} max={labels.max():.3f}")
    return frames, labels, d_max


@torch.no_grad()
def encode_all(encoder, frames):
    """Encode (N,64,64,3) uint8 -> (N,128) latents with the frozen encoder."""
    N = len(frames)
    out = torch.empty((N, LATENT_DIM), dtype=torch.float32)
    for i in range(0, N, ENC_CHUNK):
        chunk = frames[i:i + ENC_CHUNK]
        t = (torch.from_numpy(chunk).float().div_(255.0)
             .permute(0, 3, 1, 2).to(DEVICE))
        z = encoder(t)
        out[i:i + ENC_CHUNK] = z.cpu()
    return out


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a * b).sum().item() / denom) if denom > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distance", "rtg"], default="distance")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--out", default="value_head_dist.pth")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  mode={args.mode}  gamma={args.gamma}  out={args.out}")

    print("\nLoading frames + labels...")
    frames, labels, d_max = load_frames_labels(args.mode, args.gamma)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("Encoding frames (frozen ae_multienv encoder)...")
    Z = encode_all(ae.encoder, frames)
    Y = torch.from_numpy(labels)

    ds = TensorDataset(Z, Y)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(ds, [n_tr, n_val], generator=g)
    print(f"  train={len(tr)}  val={len(va)}")

    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False)

    head = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    opt = optim.Adam(head.parameters(), lr=LR)

    best = float("inf")
    print(f"\nTraining {args.epochs} epochs, MSE loss...\n")
    for ep in range(args.epochs):
        head.train()
        tot = nb = 0
        for z, y in tl:
            z, y = z.to(DEVICE), y.to(DEVICE)
            pred = head(z)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)

        head.eval()
        vtot = vn = 0
        preds, ys = [], []
        with torch.no_grad():
            for z, y in vl:
                z, y = z.to(DEVICE), y.to(DEVICE)
                pred = head(z)
                vtot += F.mse_loss(pred, y).item(); vn += 1
                preds.append(pred.cpu()); ys.append(y.cpu())
        v_loss = vtot / max(vn, 1)
        corr = pearson(torch.cat(preds), torch.cat(ys))

        mark = ""
        if v_loss < best:
            best = v_loss
            torch.save(head.state_dict(), args.out)
            mark = "  <- best"
        print(f"epoch {ep:02d} | train {tr_loss:.5f} | val {v_loss:.5f} "
              f"| corr(pred,true)={corr:+.3f}{mark}")

    print(f"\nBest val MSE: {best:.5f}")
    print(f"Saved: {args.out}")
    print("\nTo use it live, set in agent_map_step3.py:")
    print(f'  VALUE_WEIGHTS = "{args.out}"')


if __name__ == "__main__":
    main()
