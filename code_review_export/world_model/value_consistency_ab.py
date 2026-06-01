"""
P2b gate — imagined-value reliability: old latent vs P2b latent.

The metric MPC actually consumes. The planner scores candidates with value(z_h)
where z_h is the IMAGINED latent rolled forward by the dynamics. err/disp (all
latent dims) and decoded-frame MSE (decoder's dims) both miss this; what matters
is whether value(imagined z_h) still tracks the truth at the planning horizon.

For each stack we roll dynamics forward from z_t with the RECORDED actions and
measure, per horizon h:
  corr_imag(h) = corr( V(imagined z_h), -true_distance )   # imagined-value reliability
  corr_real(h) = corr( V(real    z_h), -true_distance )    # per-stack ceiling
  mae_v(h)     = mean |V(imagined z_h) - V(real z_h)|       # imagined value error
  drift(h)     = mean ||imagined z_h - real z_h||           # latent drift (within-stack only)

Two stacks, SAME start points for fairness:
  OLD : ae_multienv   + dynamics_multienv   + value_head_dist
  P2b : ae_predictive + dynamics_predictive + value_head_dist_pred

Cross-stack comparable: corr_imag/corr_real (scale-free) and mae_v (value in
[0,1] for both). drift is NOT comparable across stacks (different latent scales);
read it within a stack only.

Headline = corr_imag at h=4 (the MPC horizon). If P2b > OLD there, the
predictable latent makes the planner's imagined value more trustworthy — the
agent-relevant payoff that err/disp could not show.

Output -> world_model_out/diagnostics/value_consistency_ab.png  (+ console table)

Usage:
    py -3.9 world_model/value_consistency_ab.py
    py -3.9 world_model/value_consistency_ab.py --hmax 8 --n-starts 6000
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from value_diagnostics import (load_episodes, encode, pearson, euclid,
                               OUT_DIR, SEED, LATENT_DIM, NUM_ACTIONS, DEVICE)
from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP
from train_value_head import ValueHead

VAL_HIDDEN = 256
DYN_HIDDEN = 512


def load_stack(ae_path, dyn_path, val_path):
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(ae_path, map_location=DEVICE)); ae.eval()
    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS,
                             hidden=DYN_HIDDEN).to(DEVICE)
    dyn.load_state_dict(torch.load(dyn_path, map_location=DEVICE)); dyn.eval()
    val = ValueHead(latent_dim=LATENT_DIM, hidden=VAL_HIDDEN).to(DEVICE)
    val.load_state_dict(torch.load(val_path, map_location=DEVICE)); val.eval()
    for m in (ae, dyn, val):
        for p in m.parameters():
            p.requires_grad = False
    print(f"  stack: AE={ae_path}  DYN={dyn_path}  V={val_path}")
    return ae, dyn, val


@torch.no_grad()
def rollout_fidelity(eps, ae, dyn, val, by_ep, hmax):
    """Per-horizon imagined-value reliability. by_ep: {ei: array of start t's}
    shared across stacks so both see identical transitions."""
    V_real = [[] for _ in range(hmax + 1)]
    V_imag = [[] for _ in range(hmax + 1)]
    true_d = [[] for _ in range(hmax + 1)]
    lat_drift = [[] for _ in range(hmax + 1)]

    for ei, starts in by_ep.items():
        e = eps[ei]
        zfull = encode(ae.encoder, e["obs"])          # (T,128)
        acts = e["actions"]; pos = e["pos"]
        starts = np.asarray(starts)
        z_im = zfull[starts]                           # imagined, h=0
        for h in range(hmax + 1):
            tph = starts + h
            zr = zfull[tph]
            vr = val(zr).reshape(-1).cpu().numpy()
            vi = val(z_im).reshape(-1).cpu().numpy()
            td = euclid(pos[tph])
            drift = torch.norm(z_im - zr, dim=1).cpu().numpy()
            V_real[h].append(vr); V_imag[h].append(vi)
            true_d[h].append(td); lat_drift[h].append(drift)
            if h < hmax:
                a_h = torch.from_numpy(acts[starts + h]).long().to(DEVICE)
                z_im = dyn(z_im, a_h)

    hs = np.arange(hmax + 1)
    out = {"hs": hs, "corr_real": [], "corr_imag": [], "mae_v": [], "drift": []}
    for h in hs:
        vr = np.concatenate(V_real[h]); vi = np.concatenate(V_imag[h])
        td = np.concatenate(true_d[h]); dr = np.concatenate(lat_drift[h])
        out["corr_real"].append(pearson(vr, -td))
        out["corr_imag"].append(pearson(vi, -td))
        out["mae_v"].append(float(np.mean(np.abs(vi - vr))))
        out["drift"].append(float(dr.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmax", type=int, default=8)
    ap.add_argument("--n-starts", type=int, default=4000)
    ap.add_argument("--old-ae", default="ae_multienv.pth")
    ap.add_argument("--old-dyn", default="dynamics_multienv.pth")
    ap.add_argument("--old-val", default="value_head_dist.pth")
    ap.add_argument("--new-ae", default="ae_predictive.pth")
    ap.add_argument("--new-dyn", default="dynamics_predictive.pth")
    ap.add_argument("--new-val", default="value_head_dist_pred.pth")
    args = ap.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading episodes...")
    eps = load_episodes()

    # shared start points (same transitions for both stacks)
    candidates = []
    for ei, e in enumerate(eps):
        T = len(e["obs"])
        for t in range(0, T - args.hmax):
            candidates.append((ei, t))
    rng.shuffle(candidates)
    candidates = candidates[:args.n_starts]
    by_ep = {}
    for ei, t in candidates:
        by_ep.setdefault(ei, []).append(t)
    print(f"  {len(candidates)} shared start points, hmax={args.hmax}")

    print("\n[OLD stack]")
    ae, dyn, val = load_stack(args.old_ae, args.old_dyn, args.old_val)
    old = rollout_fidelity(eps, ae, dyn, val, by_ep, args.hmax)

    print("\n[P2b stack]")
    ae, dyn, val = load_stack(args.new_ae, args.new_dyn, args.new_val)
    new = rollout_fidelity(eps, ae, dyn, val, by_ep, args.hmax)

    hs = old["hs"]
    print("\n  imagined-value reliability   corr(V(imagined z_h), -dist)")
    print("  h | OLD c_imag  P2b c_imag | OLD c_real  P2b c_real | OLD |dV|  P2b |dV|")
    for h in hs:
        print(f"  {h} |   {old['corr_imag'][h]:+.3f}      {new['corr_imag'][h]:+.3f}  |"
              f"   {old['corr_real'][h]:+.3f}      {new['corr_real'][h]:+.3f}  |"
              f"   {old['mae_v'][h]:.3f}     {new['mae_v'][h]:.3f}")

    H = min(4, len(hs) - 1)
    print(f"\n  HEADLINE @ h={H} (MPC horizon):")
    print(f"    corr_imag   OLD {old['corr_imag'][H]:+.3f}  ->  P2b {new['corr_imag'][H]:+.3f}")
    print(f"    |V_im-V_re| OLD {old['mae_v'][H]:.3f}   ->  P2b {new['mae_v'][H]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle("P2b gate — imagined-value reliability vs horizon "
                 "(OLD latent vs P2b latent)", fontsize=13, weight="bold")

    ax = axes[0]
    ax.plot(hs, old["corr_imag"], "s--", color="tab:red", lw=2, label="OLD imagined")
    ax.plot(hs, new["corr_imag"], "o-", color="tab:blue", lw=2, label="P2b imagined")
    ax.plot(hs, old["corr_real"], ":", color="tab:red", alpha=0.45, label="OLD real (ceiling)")
    ax.plot(hs, new["corr_real"], ":", color="tab:blue", alpha=0.45, label="P2b real (ceiling)")
    ax.axvline(4, color="gray", ls=":", label="MPC horizon H=4")
    ax.set_xlabel("horizon h (steps ahead)")
    ax.set_ylabel("corr(V(imagined), -distance)")
    ax.set_title("(A) Imagined-value reliability\nhigher = planner's value is trustworthy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.plot(hs, old["mae_v"], "s--", color="tab:red", lw=2, label="OLD")
    ax.plot(hs, new["mae_v"], "o-", color="tab:blue", lw=2, label="P2b")
    ax.axvline(4, color="gray", ls=":")
    ax.set_xlabel("horizon h"); ax.set_ylabel("mean |V(imagined) - V(real)|")
    ax.set_title("(B) Imagined value error (lower = better)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(hs, old["drift"], "s--", color="tab:red", lw=2, label="OLD")
    ax.plot(hs, new["drift"], "o-", color="tab:blue", lw=2, label="P2b")
    ax.axvline(4, color="gray", ls=":")
    ax.set_xlabel("horizon h"); ax.set_ylabel("mean ||z_imag - z_real|| (within-stack)")
    ax.set_title("(C) Latent drift (scales differ; do not compare across stacks)")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "value_consistency_ab.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  saved {p}")


if __name__ == "__main__":
    main()
