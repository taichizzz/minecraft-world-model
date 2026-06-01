"""
Retrain the dynamics model on human-collected data (`dataset_3_human`).

Why: the existing dyn (`dynamics_turn_3.pth`) was trained on random
exploration, so it never saw goal-directed trajectories. When MPC rolls
it forward, predictions don't reflect "what happens during an approach
to the diamond" — which is exactly what the planner needs.

The human dataset records 4 actions: 0=move, 1=turnR, 2=turnL, 3=idle.
We FILTER OUT idle frames so the resulting model stays 3-action and
matches the existing planner interface. For each (frame_t, action_t,
frame_{t+1}) triple where action_t != 3, we have a clean transition
the human committed to: a real move/turn that changed the world state.

Saves `dynamics_turn_human.pth`. Drop-in replacement for
`dynamics_turn_3.pth` in `agent_map_step3.py`.
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn3.pth"
OUT_WEIGHTS = "dynamics_turn_human.pth"
DATA_DIRS = ["dataset/dataset_3_human2"]   # cleaner, no idle frames

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 512

BATCH = 64
EPOCHS = 100
LR = 1e-3
GRAD_CLIP = 1.0
VAL_FRAC = 0.1
SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


class HumanTransitionDataset(Dataset):
    """One single-step transition per item: (frame_before, action, frame_after).

    Drops every frame whose recorded action is 3 (idle / no-op). We keep
    `frame_t` and `frame_{t+1}` from the *raw* trajectory whenever
    `action_t in {0, 1, 2}`. `frame_{t+1}` is the state right after the
    human's real action — even if the next frame happens to be another
    idle moment, position has already updated, so the latent target is
    correct.
    """

    def __init__(self, data_dirs):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        before, acts, after = [], [], []
        action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        total_files = 0
        for data_dir in data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
            if not files:
                print(f"  WARNING: no episodes in {data_dir}")
                continue
            total_files += len(files)
            for fp in files:
                data = np.load(fp)
                obs = data["obs"]
                actions = data["actions"]
                T = len(actions)
                for t in range(T - 1):
                    a = int(actions[t])
                    action_hist[a] = action_hist.get(a, 0) + 1
                    if a == 3:
                        continue
                    before.append(obs[t])
                    acts.append(a)
                    after.append(obs[t + 1])
            print(f"  loaded {data_dir}: {len(files)} episodes")
        if not before:
            raise RuntimeError("No transitions collected from any data directory")
        self.before = np.stack(before)
        self.actions = np.array(acts, dtype=np.int64)
        self.after = np.stack(after)
        total = sum(action_hist.values())
        print(f"  total: {total_files} episodes, raw action distribution:")
        for a, c in sorted(action_hist.items()):
            name = {0: "move", 1: "turnR", 2: "turnL", 3: "idle"}.get(a, f"a{a}")
            print(f"    {name}: {c} ({100*c/total:.1f}%)")
        print(f"  kept {len(self.actions)} non-idle transitions")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        b = torch.from_numpy(self.before[idx]).float().div_(255.0).permute(2, 0, 1)
        a = torch.tensor(self.actions[idx], dtype=torch.int64)
        af = torch.from_numpy(self.after[idx]).float().div_(255.0).permute(2, 0, 1)
        return b, a, af


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"DEVICE: {DEVICE}")
    print("Building human transition dataset...")
    ds = HumanTransitionDataset(DATA_DIRS)

    n = len(ds)
    n_val = max(1, int(VAL_FRAC * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=HIDDEN
    ).to(DEVICE)
    opt = optim.Adam(dyn.parameters(), lr=LR)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(EPOCHS):
        # ---- train ----
        dyn.train()
        total, nb = 0.0, 0
        for b, a, af in train_loader:
            b = b.to(DEVICE); a = a.to(DEVICE); af = af.to(DEVICE)
            with torch.no_grad():
                z = ae.encoder(b)
                z_next_true = ae.encoder(af)
            z_pred = dyn(z, a)
            loss = F.mse_loss(z_pred, z_next_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item(); nb += 1
        train_loss = total / max(nb, 1)
        train_losses.append(train_loss)

        # ---- val ----
        dyn.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for b, a, af in val_loader:
                b = b.to(DEVICE); a = a.to(DEVICE); af = af.to(DEVICE)
                z = ae.encoder(b)
                z_next_true = ae.encoder(af)
                z_pred = dyn(z, a)
                vloss = F.mse_loss(z_pred, z_next_true)
                vtot += vloss.item(); vn += 1
        val_loss = vtot / max(vn, 1)
        val_losses.append(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dyn.state_dict(), OUT_WEIGHTS)
            marker = "  <- best"
        print(f"epoch {epoch:02d} | train {train_loss:.6f} | val {val_loss:.6f}{marker}")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved: {OUT_WEIGHTS}")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("single-step MSE")
    plt.title("Dyn training on human data (idle filtered)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dynamics_turn_human_loss.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
