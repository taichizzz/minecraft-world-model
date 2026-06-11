"""
train_dynamics_value_consistent.py  (GPT's proposed experiment, refined)

Targets the bottleneck value_consistency_ab.py measured directly:
   V(real latent)            is strong   (corr ~0.98, flat over horizon)
   V(dynamics-imagined z)    DRIFTS      (corr 0.98@h1 -> 0.90@h4 -> 0.83@h8)
MPC scores IMAGINED latents with the value head, so what matters is that
V(z_pred) tracks V(z_true) over the rollout -- not just that z_pred ~ z_true.

Fine-tunes ONLY the dynamics (AE + value head FROZEN):
   latent_loss = MSE(z_pred_k, z_true_k)            # anchor: stay on-manifold
   value_loss  = MSE(V(z_pred_k), V(z_true_k))      # the fix: preserve value
   total       = mean_k [ latent_loss + LAMBDA * value_loss ]

Why this is safer than the P2b/P2c latent rebuild: frozen AE+value can't collapse
the latent or move the value target, and the latent anchor stops the dynamics
from "cheating" to the right value via a geometrically wrong latent. It also
directly optimises the quantity MPC consumes.

MULTI-STEP on purpose: 1-step value consistency is already fine (corr@h1=0.96);
the drift is cumulative, so we roll K steps and sum the value-consistency.

COLLAPSE GUARD: a dynamics that outputs a near-constant latent would score low
value_loss trivially (V barely changes). We print std(V_hat) vs std(V_true) and
the latent err/disp every epoch -- if std(V_hat) collapses or err/disp explodes,
LAMBDA is too high.

Init: dynamics_multienv.pth.  Output: dynamics_value_consistent.pth
Drop-in replacement for DYN_WEIGHTS in the multienv stack
(pairs with ae_multienv.pth + value_head_dist.pth).

Gate after (before/after on the exact MPC-relevant metric):
  py -3.9 world_model/value_consistency_ab.py \
    --old-ae ae_multienv.pth --old-dyn dynamics_multienv.pth --old-val value_head_dist.pth \
    --new-ae ae_multienv.pth --new-dyn dynamics_value_consistent.pth --new-val value_head_dist.pth
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP
from train_value_head import ValueHead
from value_diagnostics import load_episodes, encode, pearson, LATENT_DIM, SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ACTIONS = 3
DYN_HIDDEN = 512
VAL_HIDDEN = 256

AE_WEIGHTS = "ae_multienv.pth"
VALUE_WEIGHTS = "value_head_dist.pth"
DYN_INIT = "dynamics_multienv.pth"
OUT_DYN = "dynamics_value_consistent.pth"

K_MAX = 4
K_SCHEDULE = [(0.0, 1), (0.25, 2), (0.5, 4)]
LAMBDA = 5.0           # weight on value-consistency (sweep with --lam)
EPOCHS = 50
LR = 3e-4
LR_MIN = 1e-5
BATCH = 256
VAL_FRAC = 0.1
GRAD_CLIP = 1.0


class LatentWindow(Dataset):
    """Windows of cached latents: returns (z[t:t+K+1], a[t:t+K]). Skips idle."""
    def __init__(self, Zs, As, K):
        self.Zs, self.As, self.K = Zs, As, K
        self.index = []
        for ei, (Z, A) in enumerate(zip(Zs, As)):
            for t in range(len(Z) - K):
                if (A[t:t + K] == 3).any():
                    continue
                self.index.append((ei, t))
        if not self.index:
            raise RuntimeError("No valid windows.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, t = self.index[i]
        return self.Zs[ei][t:t + self.K + 1], self.As[ei][t:t + self.K]


def current_K(epoch):
    K = 1
    for frac, k in K_SCHEDULE:
        if epoch >= frac * EPOCHS:
            K = k
    return K


@torch.no_grad()
def cache_latents(ae, eps):
    Zs, As = [], []
    for e in eps:
        z = encode(ae.encoder, e["obs"]).cpu()          # (T,128)
        Zs.append(z)
        As.append(torch.from_numpy(e["actions"]).long())
    return Zs, As


def rollout_losses(dyn, value, z_seq, a_seq, K):
    """z_seq:(B,K+1,128) a_seq:(B,K). Returns (lat, val, vhat_K, vtrue_K)."""
    z_hat = z_seq[:, 0]
    lat = val = 0.0
    vhat_K = vtrue_K = None
    for k in range(K):
        z_hat = dyn(z_hat, a_seq[:, k])
        z_tru = z_seq[:, k + 1]
        lat = lat + F.mse_loss(z_hat, z_tru)
        v_hat = value(z_hat).reshape(-1)
        with torch.no_grad():
            v_tru = value(z_tru).reshape(-1)
        val = val + F.mse_loss(v_hat, v_tru)
        if k == K - 1:
            vhat_K, vtrue_K = v_hat.detach(), v_tru
    return lat / K, val / K, vhat_K, vtrue_K


def main():
    global LAMBDA, EPOCHS, BATCH, AE_WEIGHTS, VALUE_WEIGHTS, DYN_INIT, OUT_DYN
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=LAMBDA,
                    help="weight on value-consistency loss")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ae", default=AE_WEIGHTS, help="frozen AE (latent provider)")
    ap.add_argument("--value", default=VALUE_WEIGHTS, help="frozen value head")
    ap.add_argument("--dyn-init", default=DYN_INIT, help="dynamics to fine-tune from")
    ap.add_argument("--out", default=OUT_DYN, help="output dynamics filename")
    args = ap.parse_args()
    LAMBDA, EPOCHS, BATCH = args.lam, args.epochs, args.batch
    AE_WEIGHTS, VALUE_WEIGHTS, DYN_INIT, OUT_DYN = args.ae, args.value, args.dyn_init, args.out

    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  LAMBDA={LAMBDA}  EPOCHS={EPOCHS}  K_MAX={K_MAX}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE)); ae.eval()
    value = ValueHead(latent_dim=LATENT_DIM, hidden=VAL_HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE)); value.eval()
    for m in (ae, value):
        for p in m.parameters():
            p.requires_grad = False

    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS,
                             hidden=DYN_HIDDEN).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_INIT, map_location=DEVICE))
    print(f"frozen AE={AE_WEIGHTS}  frozen V={VALUE_WEIGHTS}  dyn init={DYN_INIT}")

    print("Caching latents (frozen encoder)...")
    eps = load_episodes()
    Zs, As = cache_latents(ae, eps)
    ds = LatentWindow(Zs, As, K_MAX)
    n_val = max(1, int(VAL_FRAC * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(SEED))
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False)
    print(f"  windows: train={len(tr)} val={len(va)}")

    opt = optim.Adam(dyn.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    best = float("inf")
    print(f"\nK schedule {K_SCHEDULE}\n")
    for epoch in range(EPOCHS):
        K = current_K(epoch)
        dyn.train()
        for z_seq, a_seq in tl:
            z_seq = z_seq.to(DEVICE); a_seq = a_seq.to(DEVICE)
            lat, val, _, _ = rollout_losses(dyn, value, z_seq, a_seq, K)
            loss = lat + LAMBDA * val
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()

        dyn.eval()
        agg = {"lat": 0.0, "val": 0.0, "mae": 0.0, "n": 0}
        vh, vt = [], []
        with torch.no_grad():
            for z_seq, a_seq in vl:
                z_seq = z_seq.to(DEVICE); a_seq = a_seq.to(DEVICE)
                lat, val, vhat_K, vtrue_K = rollout_losses(dyn, value, z_seq, a_seq, K)
                agg["lat"] += lat.item(); agg["val"] += val.item()
                agg["mae"] += (vhat_K - vtrue_K).abs().mean().item(); agg["n"] += 1
                vh.append(vhat_K.cpu().numpy()); vt.append(vtrue_K.cpu().numpy())
        for k in ("lat", "val", "mae"):
            agg[k] /= max(agg["n"], 1)
        vh = np.concatenate(vh); vt = np.concatenate(vt)
        total = agg["lat"] + LAMBDA * agg["val"]
        corrK = pearson(vh, vt)
        mark = ""
        # Only checkpoint within the final curriculum stage. Val totals are NOT
        # comparable across K (a 1-step rollout is structurally lower-loss than
        # a 4-step one), so comparing across K saves an early short-horizon
        # epoch instead of the multi-step model MPC actually needs.
        if K == K_MAX and total < best:
            best = total
            torch.save(dyn.state_dict(), OUT_DYN)
            mark = "  <- best (saved)"
        print(f"ep {epoch:02d} K={K} | val total {total:.5f} | "
              f"latMSE {agg['lat']:.5f} valMSE {agg['val']:.5f} | "
              f"h{K}: MAE(V) {agg['mae']:.4f} corr {corrK:+.3f} "
              f"std(Vhat) {vh.std():.3f} std(Vtrue) {vt.std():.3f}{mark}")

    print(f"\nBest val total: {best:.6f}\nSaved: {OUT_DYN}")
    print("Collapse check: std(Vhat) should stay close to std(Vtrue); if it "
          "fell toward 0, LAMBDA is too high (dynamics learned a constant).")
    print(f"\nGate it:\n  py -3.9 world_model/value_consistency_ab.py "
          f"--old-ae {AE_WEIGHTS} --old-dyn {DYN_INIT} --old-val {VALUE_WEIGHTS} "
          f"--new-ae {AE_WEIGHTS} --new-dyn {OUT_DYN} --new-val {VALUE_WEIGHTS}")


if __name__ == "__main__":
    main()
