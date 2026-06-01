"""
Offline diagnostics for the goal-directed value head + world-model dynamics.

Produces paper-ready figures WITHOUT running Minecraft. Everything here is
deterministic and reproducible (SEED=42), computed from the frozen
ae_multienv encoder + dynamics_multienv + value_head_dist that the live agent
loads. The point is to *explain* the eval results:

  * The value head scores corr~0.98 offline yet WM-only navigation is weak.
    Why? Online, V is read off IMAGINED latents from the dynamics rollout,
    which drift from the real manifold. These figures quantify that gap.

Figures (-> world_model_out/diagnostics/):
  fig1_value_vs_distance.png   : V(z) read from REAL frames vs true distance
                                 to goal. The headline "the value head learned
                                 navigation from pixels" plot. (corr annotated)
  fig2_calibration.png         : V binned by true distance, mean +/- std.
                                 Shows V is a smooth monotone distance signal.
  fig3_rollout_fidelity.png    : 3 panels vs imagination horizon h:
                                   (A) corr(V, true) for REAL vs IMAGINED z_h
                                   (B) latent drift ||z_imag - z_real||
                                   (C) value error |V_imag - V_real|
                                 This is the figure that explains 0.98 -> weak
                                 WM-only: dynamics drift past the K=4 horizon.
  fig4_value_trajectories.png  : V(z_t) vs normalized true distance over time
                                 for a few sample episodes. Visual proof V
                                 tracks progress toward the goal.

Usage:
    py -3.9 world_model/value_diagnostics.py
    py -3.9 world_model/value_diagnostics.py --hmax 8 --n-starts 4000
"""
import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP
from train_value_head import ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 256

AE_WEIGHTS = "ae_multienv.pth"
DYN_WEIGHTS = "dynamics_multienv.pth"
VALUE_WEIGHTS = "value_head_dist.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

GOAL = np.array([4.5, 4.5], dtype=np.float32)
SUCCESS_RADIUS = 2.0
SEED = 42
ENC_CHUNK = 512

OUT_DIR = "world_model_out/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)


def euclid(p):
    return np.sqrt((p[..., 0] - GOAL[0]) ** 2 + (p[..., 1] - GOAL[1]) ** 2)


def manhattan(p):
    return np.abs(p[..., 0] - GOAL[0]) + np.abs(p[..., 1] - GOAL[1])


def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def load_episodes():
    """Return list of dicts {obs, actions, pos, success} per episode."""
    eps = []
    for d in DATASET_DIRS:
        for fp in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
            data = np.load(fp)
            obs = data["obs"]
            if obs.shape[0] < 2:
                continue
            pos = data["positions"].astype(np.float32)
            actions = data["actions"].astype(np.int64)
            yaw = data["yaw"].astype(np.float32)
            success = manhattan(pos[-1]) < SUCCESS_RADIUS
            eps.append({"obs": obs, "actions": actions, "pos": pos,
                        "yaw": yaw, "success": bool(success)})
    if not eps:
        raise RuntimeError("No episodes found - check DATASET_DIRS.")
    return eps


@torch.no_grad()
def encode(encoder, frames):
    """(N,64,64,3) uint8 -> (N,128) torch on DEVICE."""
    N = len(frames)
    out = torch.empty((N, LATENT_DIM), dtype=torch.float32, device=DEVICE)
    for i in range(0, N, ENC_CHUNK):
        chunk = frames[i:i + ENC_CHUNK]
        t = (torch.from_numpy(chunk).float().div_(255.0)
             .permute(0, 3, 1, 2).to(DEVICE))
        out[i:i + ENC_CHUNK] = encoder(t)
    return out


def load_models():
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE)); ae.eval()
    dyn = DynamicsTurningMLP(latent_dim=LATENT_DIM,
                             num_actions=NUM_ACTIONS, hidden=512).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE)); dyn.eval()
    val = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    val.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE)); val.eval()
    for m in (ae, dyn, val):
        for p in m.parameters():
            p.requires_grad = False
    print(f"Loaded AE={AE_WEIGHTS} DYN={DYN_WEIGHTS} V={VALUE_WEIGHTS}  DEVICE={DEVICE}")
    return ae, dyn, val


# ----------------------------------------------------------------------
@torch.no_grad()
def fig1_and_2_value_vs_distance(eps, ae, val):
    """V read from REAL frames vs true Euclidean distance to goal."""
    frames = np.concatenate([e["obs"] for e in eps], axis=0)
    pos = np.concatenate([e["pos"] for e in eps], axis=0)
    dist = euclid(pos)
    d_max = float(np.percentile(dist, 99))

    z = encode(ae.encoder, frames)
    v = val(z).cpu().numpy()
    label = 1.0 - np.clip(dist / d_max, 0.0, 1.0)

    corr_dist = pearson(v, -dist)       # V should rise as distance falls
    corr_lab = pearson(v, label)
    print(f"  fig1: frames={len(v)}  corr(V,-dist)={corr_dist:+.3f}  "
          f"corr(V,label)={corr_lab:+.3f}  d_max={d_max:.2f}")

    # ---- Figure 1: hexbin scatter ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hb = ax.hexbin(dist, v, gridsize=45, cmap="viridis", bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("True Euclidean distance to goal (blocks)")
    ax.set_ylabel("Predicted value  V(z)   [from pixels]")
    ax.set_title(f"Value head reads goal-distance from pixels\n"
                 f"Pearson r(V, -distance) = {corr_dist:+.3f}   (N={len(v)} frames)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "fig1_value_vs_distance.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    # ---- Figure 2: calibration (V binned by distance) ----
    nb = 20
    edges = np.linspace(0, dist.max(), nb + 1)
    idx = np.clip(np.digitize(dist, edges) - 1, 0, nb - 1)
    centers, means, stds = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 20:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(v[m].mean()); stds.append(v[m].std())
    centers = np.array(centers); means = np.array(means); stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.fill_between(centers, means - stds, means + stds, alpha=0.25,
                    color="tab:blue", label="+/- 1 std")
    ax.plot(centers, means, "o-", color="tab:blue", lw=2, label="mean V(z)")
    ideal = 1.0 - np.clip(centers / d_max, 0, 1)
    ax.plot(centers, ideal, "--", color="black", lw=1.5,
            label="training target  1 - d/d_max")
    ax.set_xlabel("True Euclidean distance to goal (blocks)")
    ax.set_ylabel("Predicted value  V(z)")
    ax.set_title("Value calibration: V is a smooth monotone distance signal")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "fig2_calibration.png")
    fig.savefig(p2, dpi=150); plt.close(fig)

    print(f"  saved {p1}\n  saved {p2}")
    return d_max


# ----------------------------------------------------------------------
@torch.no_grad()
def fig3_rollout_fidelity(eps, ae, dyn, val, d_max, hmax, n_starts, rng):
    """How fast does the world model drift? Compare V on REAL z_{t+h} vs
    IMAGINED z_h (rolled from z_t with the recorded actions)."""
    # collect (episode, t) start points with >= hmax future steps
    candidates = []
    for ei, e in enumerate(eps):
        T = len(e["obs"])
        for t in range(0, T - hmax):
            candidates.append((ei, t))
    rng.shuffle(candidates)
    candidates = candidates[:n_starts]
    print(f"  fig3: {len(candidates)} start points, hmax={hmax}")

    # encode each episode's needed frames once (cache per episode)
    # gather batched arrays: z0, real z at t+h, actions a_{t..t+h-1}, true dist
    # We process episode-by-episode to reuse encodings.
    by_ep = {}
    for ei, t in candidates:
        by_ep.setdefault(ei, []).append(t)

    # accumulators per horizon (h=0..hmax)
    V_real = [[] for _ in range(hmax + 1)]
    V_imag = [[] for _ in range(hmax + 1)]
    true_d = [[] for _ in range(hmax + 1)]
    lat_drift = [[] for _ in range(hmax + 1)]

    for ei, starts in by_ep.items():
        e = eps[ei]
        zfull = encode(ae.encoder, e["obs"])          # (T,128)
        acts = e["actions"]
        pos = e["pos"]
        starts = np.array(starts)

        z_im = zfull[starts]                          # (B,128) imagined, h=0
        # h = 0
        for h in range(0, hmax + 1):
            tph = starts + h
            zr = zfull[tph]
            vr = val(zr).cpu().numpy()
            vi = val(z_im).cpu().numpy()
            td = euclid(pos[tph])
            drift = torch.norm(z_im - zr, dim=1).cpu().numpy()
            V_real[h].append(vr); V_imag[h].append(vi)
            true_d[h].append(td); lat_drift[h].append(drift)
            if h < hmax:
                a_h = torch.from_numpy(acts[starts + h]).long().to(DEVICE)
                z_im = dyn(z_im, a_h)

    hs = np.arange(hmax + 1)
    corr_real, corr_imag, mae_v, mean_drift = [], [], [], []
    for h in hs:
        vr = np.concatenate(V_real[h]); vi = np.concatenate(V_imag[h])
        td = np.concatenate(true_d[h]); dr = np.concatenate(lat_drift[h])
        corr_real.append(pearson(vr, -td))
        corr_imag.append(pearson(vi, -td))
        mae_v.append(float(np.mean(np.abs(vi - vr))))
        mean_drift.append(float(dr.mean()))

    print("  h |  corr_real  corr_imag |  |V_im-V_re|  lat_drift")
    for h in hs:
        print(f"  {h} |   {corr_real[h]:+.3f}     {corr_imag[h]:+.3f}  |"
              f"    {mae_v[h]:.3f}      {mean_drift[h]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle("World-model rollout fidelity vs imagination horizon "
                 f"(dynamics trained at K=4)", fontsize=13, weight="bold")

    ax = axes[0]
    ax.plot(hs, corr_real, "o-", color="tab:green", lw=2,
            label="REAL z (encoder)")
    ax.plot(hs, corr_imag, "s--", color="tab:red", lw=2,
            label="IMAGINED z (dynamics)")
    ax.axvline(4, color="gray", ls=":", label="training horizon K=4")
    ax.set_xlabel("horizon h (steps ahead)")
    ax.set_ylabel("Pearson r(V, -distance)")
    ax.set_title("(A) Value reliability degrades as imagination drifts")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.02)

    ax = axes[1]
    ax.plot(hs, mean_drift, "o-", color="tab:purple", lw=2)
    ax.axvline(4, color="gray", ls=":")
    ax.set_xlabel("horizon h (steps ahead)")
    ax.set_ylabel("mean ||z_imag - z_real||")
    ax.set_title("(B) Latent drift accumulates with horizon")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(hs, mae_v, "o-", color="tab:orange", lw=2)
    ax.axvline(4, color="gray", ls=":")
    ax.set_xlabel("horizon h (steps ahead)")
    ax.set_ylabel("mean |V(imagined) - V(real)|")
    ax.set_title("(C) Imagined value error vs horizon")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "fig3_rollout_fidelity.png")
    fig.savefig(p3, dpi=150); plt.close(fig)
    print(f"  saved {p3}")


# ----------------------------------------------------------------------
@torch.no_grad()
def fig4_value_trajectories(eps, ae, val, d_max, rng, n_show=6):
    """V(z_t) and normalized true distance over time for sample episodes."""
    succ = [e for e in eps if e["success"] and 8 <= len(e["obs"]) <= 120]
    rng.shuffle(succ)
    succ = succ[:n_show]

    cols = 3
    rows = int(np.ceil(len(succ) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.6 * rows),
                             squeeze=False)
    fig.suptitle("Value tracks progress toward the goal (sample episodes)",
                 fontsize=13, weight="bold")

    for k, e in enumerate(succ):
        ax = axes[k // cols][k % cols]
        z = encode(ae.encoder, e["obs"])
        v = val(z).cpu().numpy()
        td = euclid(e["pos"])
        td_n = 1.0 - np.clip(td / d_max, 0, 1)
        t = np.arange(len(v))
        ax.plot(t, v, "-", color="tab:blue", lw=2, label="V(z_t) [pixels]")
        ax.plot(t, td_n, "--", color="black", lw=1.5,
                label="1 - d/d_max [truth]")
        ax.set_xlabel("timestep"); ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"r={pearson(v, td_n):+.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(len(succ), rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout()
    p4 = os.path.join(OUT_DIR, "fig4_value_trajectories.png")
    fig.savefig(p4, dpi=150); plt.close(fig)
    print(f"  saved {p4}")


@torch.no_grad()
def fig5_imagination_rollout(eps, ae, dyn, val, d_max, rng,
                             decode_h=8, n_show=3):
    """Decode imagined latents: top row REAL frames, bottom row what the
    world model IMAGINES from z_t rolled forward with the recorded actions.
    Per-column pixel MSE + imagined value annotated. Replaces the stale
    planner.py viz (correct weights, K=4 horizon)."""
    long_succ = [e for e in eps if e["success"] and len(e["obs"]) >= decode_h + 6]
    rng.shuffle(long_succ)
    long_succ = long_succ[:n_show]

    fig, axes = plt.subplots(2 * n_show, decode_h + 1,
                             figsize=(1.7 * (decode_h + 1), 1.9 * 2 * n_show),
                             squeeze=False)
    fig.suptitle("World-model imagination rollout  (top: real | bottom: decoded imagined)",
                 fontsize=13, weight="bold")
    name = ["move", "turnR", "turnL"]

    for k, e in enumerate(long_succ):
        T = len(e["obs"])
        t0 = int(rng.integers(0, max(1, T - decode_h - 1)))
        z = ae.encoder((torch.from_numpy(e["obs"][t0:t0 + 1]).float()
                        .div_(255.0).permute(0, 3, 1, 2).to(DEVICE)))  # (1,128)
        acts = e["actions"]
        for h in range(decode_h + 1):
            real = e["obs"][t0 + h]                              # (64,64,3) uint8
            recon = ae.decoder(z).squeeze(0).permute(1, 2, 0).cpu().numpy()
            mse = float(np.mean((recon - real / 255.0) ** 2))
            vimg = float(val(z).reshape(-1)[0].item())

            ax_r = axes[2 * k][h]; ax_i = axes[2 * k + 1][h]
            ax_r.imshow(real); ax_r.axis("off")
            ax_i.imshow(np.clip(recon, 0, 1)); ax_i.axis("off")
            if k == 0:
                act_lbl = "" if h == 0 else f"\n<{name[acts[t0 + h - 1]]}"
                ax_r.set_title(f"t+{h}{act_lbl}", fontsize=8)
            ax_i.set_title(f"mse {mse:.3f}\nV {vimg:.2f}", fontsize=7)
            if h < decode_h:
                a_h = torch.tensor([acts[t0 + h]], dtype=torch.long, device=DEVICE)
                z = dyn(z, a_h)
        axes[2 * k][0].set_ylabel("real", fontsize=9)
        axes[2 * k + 1][0].set_ylabel("imagined", fontsize=9)

    fig.tight_layout()
    p5 = os.path.join(OUT_DIR, "fig5_imagination_rollout.png")
    fig.savefig(p5, dpi=150); plt.close(fig)
    print(f"  saved {p5}")


def main():
    global AE_WEIGHTS, DYN_WEIGHTS, VALUE_WEIGHTS, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmax", type=int, default=8,
                    help="max imagination horizon for fig3")
    ap.add_argument("--n-starts", type=int, default=4000,
                    help="rollout start points for fig3")
    ap.add_argument("--decode-h", type=int, default=8,
                    help="imagination horizon to decode for fig5")
    ap.add_argument("--ae", default=AE_WEIGHTS, help="override AE weights")
    ap.add_argument("--dyn", default=DYN_WEIGHTS, help="override dynamics weights")
    ap.add_argument("--value", default=VALUE_WEIGHTS, help="override value head")
    ap.add_argument("--tag", default="", help="output subfolder, e.g. p2b")
    args = ap.parse_args()

    AE_WEIGHTS, DYN_WEIGHTS, VALUE_WEIGHTS = args.ae, args.dyn, args.value
    if args.tag:
        OUT_DIR = os.path.join(OUT_DIR, args.tag)
        os.makedirs(OUT_DIR, exist_ok=True)

    np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Loading episodes...")
    eps = load_episodes()
    ns = sum(e["success"] for e in eps)
    print(f"  episodes={len(eps)}  successful={ns} ({ns/len(eps):.0%})")

    ae, dyn, val = load_models()

    print("\n[fig1+2] value vs distance (real frames)...")
    d_max = fig1_and_2_value_vs_distance(eps, ae, val)

    print("\n[fig3] rollout fidelity...")
    fig3_rollout_fidelity(eps, ae, dyn, val, d_max,
                          args.hmax, args.n_starts, rng)

    print("\n[fig4] value trajectories...")
    fig4_value_trajectories(eps, ae, val, d_max, rng)

    print("\n[fig5] imagination rollout (decoded)...")
    fig5_imagination_rollout(eps, ae, dyn, val, d_max, rng,
                             decode_h=args.decode_h)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
