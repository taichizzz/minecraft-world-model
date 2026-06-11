"""
train_value_stg_aug.py — obstacle- & orientation-aware value head on the P2b
latent, trained with appearance augmentation (the SAFE frozen-encoder recipe).

MEASURED GAP (2026-06-10): the current head (value_head_dist_pred) is a pure
straight-line-distance function: corr(V,-euclid)=0.97-0.99 but corr(V,-true
steps-to-go) drops to 0.924 in env3 (obstacles), matching the chained
euclid->steps correlation — i.e. V carries no obstacle-routing and no turn
cost. Locally that means flat/wrong gradients exactly where control needs
them; it is why value-only driving is ~20%.

FIX: train on  V = gamma^steps_to_go  with steps from exact BFS over
(cell, heading) per env (train_value_stepstogo machinery): the value becomes a
COST-TO-GO FIELD that routes around obstacles and rewards facing the right
way. Encoder stays FROZEN (ae_predictive) -> cannot diverge; inputs are
appearance-jittered so the field is lighting-robust (same recipe that produced
value_head_dist_pred_aug: clean 0.969 / AUG 0.924).

History note: this label type was tried once (P1) and failed IN CONTROL —
later root-caused to the old dynamics being unable to imagine turns. That is
fixed (dynamics_predictive_vc: imagined-value corr@h4=0.963), so every link in
the chain now holds.

Gate to beat (current dist head): corr(V,-steps) = 0.966 / 0.968 / 0.924
(env1/2/3). Output: value_head_stg_aug.pth (pairs with ae_predictive +
dynamics_predictive_vc). Follow-on after gating: VC-finetune dynamics against
THIS head, then re-test value-only scoring closed-loop.

    py -3.9 world_model/train_value_stg_aug.py --epochs 10   # sanity first
    py -3.9 world_model/train_value_stg_aug.py               # full
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import AutoEncoder
from train_value_head import ValueHead
from train_value_stepstogo import load_frames_labels      # BFS (cell,heading) labels
from train_value_dist_aug import augment, pearson, prep   # proven-safe aug recipe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
VAL_HIDDEN = 256
BATCH = 256
EPOCHS = 60
LR = 1e-3
VAL_FRAC = 0.1
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="V = gamma^steps (smaller = steeper near-goal gradient)")
    ap.add_argument("--label", choices=["gamma", "linear"], default="linear",
                    help="gamma: V=gamma^steps (steep near goal, FLAT far away "
                         "-> far-field gradient ~0.02/action drowns in imagined-"
                         "value noise ~0.03, the spin-in-place failure). "
                         "linear: V=1-steps/20 -> constant 0.05/action margin "
                         "everywhere (default).")
    ap.add_argument("--aug", type=float, default=1.0,
                    help="augmentation strength (0 = clean baseline)")
    ap.add_argument("--ae", default="ae_predictive.pth")
    ap.add_argument("--out", default="value_head_stg_aug.pth")
    args = ap.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  AE(frozen)={args.ae}  gamma={args.gamma}  "
          f"aug={args.aug}  out={args.out}")

    print("\nBuilding BFS (cell,heading) steps-to-go labels...")
    frames, labels, steps = load_frames_labels(args.gamma)
    if args.label == "linear":
        labels = np.clip(1.0 - steps / 20.0, 0.0, 1.0).astype(np.float32)
        print(f"  linear labels: V = 1 - steps/20  "
              f"(range {labels.min():.2f}-{labels.max():.2f}, "
              f"gradient 0.05/action everywhere)")
    X = torch.from_numpy(frames)                      # uint8 (N,64,64,3)
    Y = torch.from_numpy(labels)                      # value targets
    S = torch.from_numpy(steps.astype(np.float32))    # raw steps (for corr gate)

    ds = TensorDataset(X, Y, S)
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
    print(f"\nTraining {args.epochs} epochs (frozen encoder, jittered inputs)...\n")
    for ep in range(args.epochs):
        head.train()
        for xb, yb, _ in tl:
            x = prep(xb)
            if args.aug > 0:
                x = augment(x, args.aug)
            with torch.no_grad():
                z = ae.encoder(x)
            loss = F.mse_loss(head(z).reshape(-1), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        pc, pa, yy, ss = [], [], [], []
        with torch.no_grad():
            for xb, yb, sb in vl:
                x = prep(xb)
                pc.append(head(ae.encoder(x)).reshape(-1).cpu())
                pa.append(head(ae.encoder(augment(x, max(args.aug, 1.0))))
                          .reshape(-1).cpu())
                yy.append(yb); ss.append(sb)
        pc = torch.cat(pc); pa = torch.cat(pa)
        yy = torch.cat(yy); ss = torch.cat(ss)
        vmse = F.mse_loss(pc, yy).item()
        c_lab = pearson(pc, yy)
        c_steps = pearson(pc, -ss)          # THE gate: track true cost-to-go
        c_steps_aug = pearson(pa, -ss)      # ...and under appearance shift
        mark = ""
        if vmse < best:
            best = vmse
            torch.save(head.state_dict(), args.out)
            mark = "  <- best"
        print(f"ep {ep:02d} | val MSE {vmse:.5f} | corr(label) {c_lab:+.3f} | "
              f"corr(-steps) {c_steps:+.3f} | AUG corr(-steps) {c_steps_aug:+.3f}{mark}")

    print(f"\nSaved {args.out}")
    print("Gate: corr(-steps) should beat the dist head's 0.966/0.968/0.924 "
          "(per-env check via the offline audit), with AUG corr close behind.")
    print("Next: VC-finetune dynamics against this head:\n"
          f"  py -3.9 world_model/train_dynamics_value_consistent.py "
          f"--ae {args.ae} --value {args.out} "
          f"--dyn-init dynamics_predictive_vc.pth --out dynamics_predictive_vc_stg.pth")


if __name__ == "__main__":
    main()
