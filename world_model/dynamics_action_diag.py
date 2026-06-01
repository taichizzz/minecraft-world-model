"""
Action-conditioned dynamics fidelity.

The P1 A/B (yaw-aware value head) made WM-only navigation WORSE, not better.
New hypothesis: the dynamics model imagines MOVES acceptably but TURNS poorly,
so a yaw-aware value (which trusts the imagined orientation) is fed bad yaw and
misleads the planner. The yaw-BLIND distance head was accidentally robust to
this because it ignored orientation.

This script breaks single-step dynamics error down BY ACTION TYPE to test that:
  * latent error ||dyn(z_t,a) - z_{t+1}|| vs actual latent displacement
  * decoded next-frame MSE:  dynamics prediction vs two baselines
        - persistence (predict "no change", decode z_t)
        - encoder ceiling (decode the REAL next latent z_{t+1})
Grouped by action, with the move/turn mapping auto-detected from yaw deltas.

If for turns  dyn_pred_MSE ~= persistence_MSE  >>  ceiling_MSE, the dynamics is
essentially predicting "nothing happened" on turns -> it cannot imagine
rotation -> that is the thing to fix (P2), and explains the P1 result.

Output -> world_model_out/diagnostics/fig6_action_conditioned_dynamics.png
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from value_diagnostics import (load_episodes, encode, load_models,
                               OUT_DIR, SEED, LATENT_DIM)
from dynamics_model import DynamicsTurningMLP
from model import AutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRANS = 8000          # transitions to sample


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn", default=None,
                    help="override dynamics weights (e.g. dynamics_multienv_v2.pth)")
    ap.add_argument("--ae", default=None,
                    help="override AE weights (e.g. ae_predictive.pth) — "
                         "needed to gate the P2b rebuilt latent")
    ap.add_argument("--tag", default="",
                    help="suffix for the output figure filename")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    eps = load_episodes()
    ae, dyn, val = load_models()
    if args.ae:
        ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
        ae.load_state_dict(torch.load(args.ae, map_location=DEVICE))
        ae.eval()
        print(f"[override] AE -> {args.ae}")
    if args.dyn:
        dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM, num_actions=3,
                                 hidden=512).to(DEVICE)
        dyn.load_state_dict(torch.load(args.dyn, map_location=DEVICE))
        dyn.eval()
        print(f"[override] dynamics -> {args.dyn}")

    # sample transitions (episode, t) with t+1 valid
    cand = []
    for ei, e in enumerate(eps):
        for t in range(len(e["obs"]) - 1):
            cand.append((ei, t))
    rng.shuffle(cand)
    cand = cand[:N_TRANS]
    by_ep = {}
    for ei, t in cand:
        by_ep.setdefault(ei, []).append(t)

    # accumulators per action 0/1/2
    lat_err = {0: [], 1: [], 2: []}
    lat_disp = {0: [], 1: [], 2: []}
    mse_pred = {0: [], 1: [], 2: []}
    mse_persist = {0: [], 1: [], 2: []}
    mse_ceiling = {0: [], 1: [], 2: []}
    yaw_delta = {0: [], 1: [], 2: []}
    pos_delta = {0: [], 1: [], 2: []}

    for ei, ts in by_ep.items():
        e = eps[ei]
        ts = np.array(ts)
        z = encode(ae.encoder, e["obs"])              # (T,128)
        acts = e["actions"]; yaw = e["yaw"]; pos = e["pos"]

        zt = z[ts]; ztp = z[ts + 1]
        a = torch.from_numpy(acts[ts]).long().to(DEVICE)
        zpred = dyn(zt, a)

        le = torch.norm(zpred - ztp, dim=1).cpu().numpy()
        ld = torch.norm(ztp - zt, dim=1).cpu().numpy()

        real_next = (torch.from_numpy(e["obs"][ts + 1]).float().div_(255.0)
                     .permute(0, 3, 1, 2).to(DEVICE))
        dpred = ae.decoder(zpred)
        dpersist = ae.decoder(zt)
        dceil = ae.decoder(ztp)
        mp = ((dpred - real_next) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        mq = ((dpersist - real_next) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        mc = ((dceil - real_next) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

        ydl = np.abs(((yaw[ts + 1] - yaw[ts] + 180) % 360) - 180)
        pdl = np.abs(pos[ts + 1] - pos[ts]).sum(axis=1)

        for j, ai in enumerate(acts[ts]):
            lat_err[ai].append(le[j]); lat_disp[ai].append(ld[j])
            mse_pred[ai].append(mp[j]); mse_persist[ai].append(mq[j])
            mse_ceiling[ai].append(mc[j])
            yaw_delta[ai].append(ydl[j]); pos_delta[ai].append(pdl[j])

    # auto-detect move vs turn from yaw delta
    names = {}
    for ai in (0, 1, 2):
        ymean = np.mean(yaw_delta[ai]); pmean = np.mean(pos_delta[ai])
        kind = "TURN" if ymean > 30 else "move"
        names[ai] = f"a{ai} ({kind})"
        print(f"  action {ai}: n={len(lat_err[ai]):5d}  "
              f"mean|dyaw|={ymean:6.1f}  mean|dpos|={pmean:.3f}  -> {kind}")

    print("\n  action        lat_err  lat_disp  err/disp | dynMSE  persistMSE  ceilMSE")
    for ai in (0, 1, 2):
        le = np.mean(lat_err[ai]); ld = np.mean(lat_disp[ai])
        mp = np.mean(mse_pred[ai]); mq = np.mean(mse_persist[ai])
        mc = np.mean(mse_ceiling[ai])
        print(f"  {names[ai]:<12} {le:7.2f}  {ld:7.2f}   {le/max(ld,1e-6):6.2f}  |"
              f" {mp:.4f}   {mq:.4f}    {mc:.4f}")

    # ---- plot ----
    order = [0, 1, 2]
    labels = [names[ai] for ai in order]
    x = np.arange(len(order)); w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("Action-conditioned dynamics fidelity (single step)",
                 fontsize=13, weight="bold")

    ax = axes[0]
    ax.bar(x - w, [np.mean(mse_ceiling[ai]) for ai in order], w,
           label="encoder ceiling (decode real z_{t+1})", color="tab:green")
    ax.bar(x, [np.mean(mse_pred[ai]) for ai in order], w,
           label="dynamics prediction", color="tab:red")
    ax.bar(x + w, [np.mean(mse_persist[ai]) for ai in order], w,
           label="persistence (predict no change)", color="tab:gray")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("decoded next-frame MSE")
    ax.set_title("(A) Can the model imagine each action?\n"
                 "dyn ~ persistence  =>  it can't model that action")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w / 2, [np.mean(lat_disp[ai]) for ai in order], w,
           label="actual latent displacement", color="tab:blue")
    ax.bar(x + w / 2, [np.mean(lat_err[ai]) for ai in order], w,
           label="dynamics latent error", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("L2 in latent space")
    ax.set_title("(B) Latent error vs how far the latent actually moves")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    suffix = f"_{args.tag}" if args.tag else ""
    p = os.path.join(OUT_DIR, f"fig6_action_conditioned_dynamics{suffix}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  saved {p}")


if __name__ == "__main__":
    main()
