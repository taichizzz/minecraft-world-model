"""
P1 — Yaw-aware value head (the fix for WM-only navigation).

Diagnosis (see value_diagnostics.py fig1/fig3): the distance-to-goal value head
scores corr 0.98 offline yet WM-only navigation is only ~20%. Root cause: the
label  V = 1 - dist(position, goal)  depends ONLY on (x,z) and is BLIND to yaw.
Two of the three actions (turnR, turnL) change only yaw, so a yaw-blind value
gives the planner ZERO gradient for *which way to turn*. The geometric term
masks this in the hybrid; with geometry off (WM-only) the agent can't orient.

Fix: label each frame with the OPTIMAL number of actions to reach the goal,
counting move AND turn steps, computed by exact BFS on the (cell, heading)
state graph using the real action model. Now facing the goal scores higher than
facing away -> the value carries orientation -> the world model gets a turn
gradient.

  label  V = gamma ** steps_to_go        (1.0 at goal, decays with #actions)

Computed offline from the GT (positions + yaw) we already store; consumed from
PIXELS at test time (same bridge as the distance head). Saves
value_head_stepstogo.pth (separate file for clean A/B vs value_head_dist.pth).
Architecture is the SAME ValueHead the agent loads.
"""
import os
import glob
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import AutoEncoder
from train_value_head import ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
HIDDEN = 256
AE_WEIGHTS = "ae_multienv.pth"

# dataset dir -> env occupied cells (must match agent_map_step3._ENV_CONFIGS)
ENV1_OCC = {(4, 4), (-4, 4), (4, -4)}
ENV2_OCC = {(4, 4), (-4, 4), (4, -4)}
ENV3_OCC = {(-2, -1), (-1, 2), (2, 1), (-3, 0), (0, 3),
            (1, -2), (3, -1), (-2, 3), (3, 2), (-1, -3),
            (4, 4), (-4, 4), (4, -4)}
DATASET_DIRS = [
    ("dataset/dataset_1_human", ENV1_OCC),
    ("dataset/dataset_2_human", ENV2_OCC),
    ("dataset/dataset_3_human2", ENV3_OCC),
]

CELL_LO, CELL_HI = -4, 4                 # walkable room (matches pick_random_spawn)
HEADINGS = [0, 90, 180, 270]
# forward (dx,dz) per heading, matching the agent kinematic sim
FWD = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}
GOAL = np.array([4.5, 4.5], dtype=np.float32)
SUCCESS_RADIUS = 2.0                     # world Manhattan, matches agent

BATCH = 256
EPOCHS = 60
LR = 1e-3
VAL_FRAC = 0.1
SEED = 42
ENC_CHUNK = 512


def in_room(c):
    return CELL_LO <= c <= CELL_HI


def target_cells(occ):
    """Walkable cells counted as success: |cx-4|+|cz-4| < SUCCESS_RADIUS."""
    cells = set()
    for cx in range(CELL_LO, CELL_HI + 1):
        for cz in range(CELL_LO, CELL_HI + 1):
            if (cx, cz) in occ:
                continue
            if (abs((cx + 0.5) - GOAL[0]) + abs((cz + 0.5) - GOAL[1])) < SUCCESS_RADIUS:
                cells.add((cx, cz))
    return cells


def steps_to_go_table(occ):
    """Exact optimal #actions (move/turnR/turnL) from every (cx,cz,heading)
    to the goal region, via BFS over REVERSED edges from the goal states.
    Returns dict[(cx,cz,heading)] -> int steps (inf -> not present)."""
    goals = target_cells(occ)
    if not goals:
        raise RuntimeError("No reachable goal cells for this env.")

    # forward adjacency: state -> list of successor states (cost 1 each)
    def succ(s):
        cx, cz, h = s
        out = []
        # turns
        out.append((cx, cz, (h + 90) % 360))     # turnR
        out.append((cx, cz, (h - 90) % 360))      # turnL
        # move
        dx, dz = FWD[h]
        nx, nz = cx + dx, cz + dz
        if in_room(nx) and in_room(nz) and (nx, nz) not in occ:
            out.append((nx, nz, h))
        return out

    # build reverse adjacency
    rev = {}
    all_states = [(cx, cz, h)
                  for cx in range(CELL_LO, CELL_HI + 1)
                  for cz in range(CELL_LO, CELL_HI + 1)
                  for h in HEADINGS
                  if (cx, cz) not in occ]
    for s in all_states:
        for s2 in succ(s):
            rev.setdefault(s2, []).append(s)

    dist = {}
    dq = deque()
    for (cx, cz) in goals:
        for h in HEADINGS:
            s = (cx, cz, h)
            dist[s] = 0
            dq.append(s)
    while dq:
        s = dq.popleft()
        for p in rev.get(s, ()):
            if p not in dist:
                dist[p] = dist[s] + 1
                dq.append(p)
    return dist


def pos_yaw_to_state(x, z, yaw):
    cx = int(np.clip(np.floor(x), CELL_LO, CELL_HI))
    cz = int(np.clip(np.floor(z), CELL_LO, CELL_HI))
    h = int(round(yaw / 90.0)) % 4 * 90
    return (cx, cz, h)


def load_frames_labels(gamma):
    frames, labels = [], []
    n_eps = n_unreach = 0
    all_steps = []
    for d, occ in DATASET_DIRS:
        table = steps_to_go_table(occ)
        max_step = max(table.values())
        for fp in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
            data = np.load(fp)
            obs = data["obs"]
            pos = data["positions"].astype(np.float32)
            yaw = data["yaw"].astype(np.float32)
            T = obs.shape[0]
            if T < 2:
                continue
            n_eps += 1
            steps = np.empty(T, dtype=np.float32)
            for t in range(T):
                s = pos_yaw_to_state(pos[t, 0], pos[t, 1], yaw[t])
                if s in table:
                    steps[t] = table[s]
                else:
                    steps[t] = max_step          # unreachable -> worst
                    n_unreach += 1
            all_steps.append(steps)
            frames.append(obs)
            labels.append(gamma ** steps)         # 1.0 at goal

    frames = np.concatenate(frames, axis=0)
    labels = np.concatenate(labels, axis=0).astype(np.float32)
    all_steps = np.concatenate(all_steps)
    print(f"  episodes={n_eps}  frames={len(frames)}  unreachable_frames={n_unreach}")
    print(f"  steps_to_go: mean={all_steps.mean():.1f} "
          f"min={all_steps.min():.0f} max={all_steps.max():.0f} "
          f"p99={np.percentile(all_steps,99):.0f}")
    print(f"  label(gamma^steps): mean={labels.mean():.3f} "
          f"min={labels.min():.3f} max={labels.max():.3f}")
    return frames, labels, all_steps


@torch.no_grad()
def encode_all(encoder, frames):
    N = len(frames)
    out = torch.empty((N, LATENT_DIM), dtype=torch.float32)
    for i in range(0, N, ENC_CHUNK):
        chunk = frames[i:i + ENC_CHUNK]
        t = (torch.from_numpy(chunk).float().div_(255.0)
             .permute(0, 3, 1, 2).to(DEVICE))
        out[i:i + ENC_CHUNK] = encoder(t).cpu()
    return out


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a * b).sum().item() / denom) if denom > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="discount per action step (smaller=steeper gradient)")
    ap.add_argument("--out", default="value_head_stepstogo.pth")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"DEVICE={DEVICE}  gamma={args.gamma}  out={args.out}")

    print("\nLoading frames + yaw-aware labels (BFS steps-to-go)...")
    frames, labels, steps = load_frames_labels(args.gamma)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    print("Encoding frames (frozen ae_multienv encoder)...")
    Z = encode_all(ae.encoder, frames)
    Y = torch.from_numpy(labels)
    steps_t = torch.from_numpy(steps)

    ds = TensorDataset(Z, Y, steps_t)
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
        for z, y, _ in tl:
            z, y = z.to(DEVICE), y.to(DEVICE)
            pred = head(z)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)

        head.eval()
        vtot = vn = 0
        preds, ys, sts = [], [], []
        with torch.no_grad():
            for z, y, st in vl:
                z, y = z.to(DEVICE), y.to(DEVICE)
                pred = head(z)
                vtot += F.mse_loss(pred, y).item(); vn += 1
                preds.append(pred.cpu()); ys.append(y.cpu()); sts.append(st)
        v_loss = vtot / max(vn, 1)
        p = torch.cat(preds)
        corr_lab = pearson(p, torch.cat(ys))
        corr_steps = pearson(p, -torch.cat(sts))     # V should fall as steps rise

        mark = ""
        if v_loss < best:
            best = v_loss
            torch.save(head.state_dict(), args.out)
            mark = "  <- best"
        print(f"epoch {ep:02d} | train {tr_loss:.5f} | val {v_loss:.5f} "
              f"| corr(pred,label)={corr_lab:+.3f} "
              f"corr(pred,-steps)={corr_steps:+.3f}{mark}")

    print(f"\nBest val MSE: {best:.5f}")
    print(f"Saved: {args.out}")
    print("\nTo A/B live, set in agent_map_step3.py:")
    print(f'  VALUE_WEIGHTS = "{args.out}"')
    print("then re-run: py -3.9 world_model/eval_batch.py --n 20 --envs 1 2 3 --wm-only")


if __name__ == "__main__":
    main()
