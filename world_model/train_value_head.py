"""
Train a value head V(z) that predicts "is the diamond_block visible in the
current frame".

We label each frame by cyan-pixel detection (see detect_diamond.relabel_episode)
because most episodes timed out at T=113 without ever touching the diamond, so
the previous "last K frames are positive" rule was mostly noise. Color labels
are clean: ~21% positives across the dataset and ~43% of episodes contain
at least one positive frame.

We train a small MLP on top of the frozen aeturn3 encoder. Output passes through
a sigmoid so V(z) ∈ [0, 1] is interpretable as "P(diamond is visible)".
Loss is binary cross-entropy.

For the planner: higher V = diamond more clearly in view. Score plans by
maximizing V — the planner will turn until the diamond is in view, then walk
toward it.

Saves value_head_2.pth (kept separate from value_head_1.pth so we can compare).
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import AutoEncoder
from detect_diamond import relabel_episode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
AE_WEIGHTS = "aeturn3.pth"
SAVE_PATH = "value_head_2.pth"

DATASET_DIRS = ["dataset/dataset2_turn", "dataset/dataset3_turn"]

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
VAL_FRACTION = 0.1
HIDDEN = 256
SEED = 42

# how many frames before episode end count as "near goal" (positive class)
NEAR_GOAL_K = 5

class ValueDataset(Dataset):
    """All frames flattened across episodes, each with its normalized progress."""

    def __init__(self, episode_files):
        frames, targets = [], []
        for fpath in episode_files:
            data = np.load(fpath)
            obs = data["obs"]
            T = obs.shape[0]
            if T < 2:
                continue
            labels, _ = relabel_episode(obs)  # cyan-pixel diamond detection
            for t in range(T):
                frames.append(obs[t])
                targets.append(float(labels[t]))
        # keep frames as uint8 to save memory; convert to float on GPU later
        self.frames = torch.from_numpy(np.stack(frames, axis=0))   # (N, H, W, 3)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.frames[idx], self.targets[idx]

class ValueHead(nn.Module):
    def __init__(self, latent_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(-1)

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    all_files = []
    for d in DATASET_DIRS:
        files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
        print(f"{d}: {len(files)} episodes")
        all_files.extend(files)
    print(f"total: {len(all_files)} episodes")

    # episode-level train/val split (not frame-level) to prevent leakage
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * VAL_FRACTION))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    print(f"train episodes: {len(train_files)}, val episodes: {len(val_files)}")

    train_ds = ValueDataset(train_files)
    val_ds = ValueDataset(val_files)
    print(f"train frames: {len(train_ds)}, val frames: {len(val_ds)}")
    train_pos = train_ds.targets.mean().item()
    val_pos = val_ds.targets.mean().item()
    print(f"positive class frac — train: {train_pos:.4f}, val: {val_pos:.4f}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # frozen encoder
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    encoder = ae.encoder

    head = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=LR)

    print(f"\nDevice: {DEVICE}  |  epochs: {EPOCHS}  |  batch: {BATCH_SIZE}\n")

    best_val = float("inf")
    for epoch in range(EPOCHS):
        head.train()
        tot = 0.0
        n = 0
        for frames, progs in train_loader:
            frames = frames.to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2)
            progs = progs.to(DEVICE)

            with torch.no_grad():
                z = encoder(frames)

            pred = head(z)
            loss = F.binary_cross_entropy(pred, progs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot += loss.item()
            n += 1
        train_loss = tot / max(n, 1)

        head.eval()
        vtot = 0.0
        vn = 0
        with torch.no_grad():
            for frames, progs in val_loader:
                frames = frames.to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2)
                progs = progs.to(DEVICE)
                z = encoder(frames)
                pred = head(z)
                vtot += F.binary_cross_entropy(pred, progs).item()
                vn += 1
        val_loss = vtot / max(vn, 1)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), SAVE_PATH)
            marker = "  <- best"
        print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved value head to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
