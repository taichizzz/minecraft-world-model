"""
Retrain smooth V head on all three environments using ae_multienv.pth.

Same approach as train_value_head_smooth.py:
  - Smooth cyan-count labels: label = min(1.0, cyan_count / MAX_CYAN)
  - MSE loss
  - ValueHead architecture (128 -> 256 -> 256 -> 1 -> sigmoid)

But uses the new multi-env encoder so latents are meaningful across
all rooms.

Saves value_head_multienv.pth.
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import AutoEncoder
from detect_diamond import diamond_pixel_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
AE_WEIGHTS = "ae_multienv.pth"
SAVE_PATH = "value_head_multienv.pth"

DATASET_DIRS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human2",
]

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
VAL_FRACTION = 0.1
HIDDEN = 256
SEED = 42
MAX_CYAN = 800.0


def smooth_label(frame):
    cnt = float(diamond_pixel_mask(frame).sum())
    return min(1.0, cnt / MAX_CYAN)


class ValueDataset(Dataset):
    def __init__(self, episode_files):
        frames, targets = [], []
        for fpath in episode_files:
            data = np.load(fpath)
            obs = data["obs"]
            for t in range(obs.shape[0]):
                frames.append(obs[t])
                targets.append(smooth_label(obs[t]))
        self.frames = torch.from_numpy(np.stack(frames, axis=0))
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.frames[idx], self.targets[idx]


class ValueHead(torch.nn.Module):
    def __init__(self, latent_dim=128, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
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

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * VAL_FRACTION))
    val_files = [all_files[i] for i in perm[:n_val]]
    train_files = [all_files[i] for i in perm[n_val:]]
    print(f"train episodes: {len(train_files)}, val episodes: {len(val_files)}")

    train_ds = ValueDataset(train_files)
    val_ds = ValueDataset(val_files)
    print(f"train frames: {len(train_ds)}, val frames: {len(val_ds)}")

    train_mean = train_ds.targets.mean().item()
    n_high = (train_ds.targets > 0.5).sum().item()
    print(f"label stats - mean={train_mean:.3f}, frac>0.5={n_high/len(train_ds):.3f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=False)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    encoder = ae.encoder

    head = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=LR)

    print(f"\nDevice: {DEVICE}  |  epochs: {EPOCHS}  |  batch: {BATCH_SIZE}")
    print(f"MAX_CYAN={MAX_CYAN}  |  loss=MSE\n")

    best_val = float("inf")
    for epoch in range(EPOCHS):
        head.train()
        tot, n = 0.0, 0
        for frames, tgts in train_loader:
            frames = frames.to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2)
            tgts = tgts.to(DEVICE)
            with torch.no_grad():
                z = encoder(frames)
            pred = head(z)
            loss = F.mse_loss(pred, tgts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
            n += 1
        train_loss = tot / max(n, 1)

        head.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for frames, tgts in val_loader:
                frames = frames.to(DEVICE).float().div_(255.0).permute(0, 3, 1, 2)
                tgts = tgts.to(DEVICE)
                z = encoder(frames)
                pred = head(z)
                vtot += F.mse_loss(pred, tgts).item()
                vn += 1
        val_loss = vtot / max(vn, 1)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), SAVE_PATH)
            marker = "  <- best"
        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | "
              f"Val MSE: {val_loss:.6f}{marker}")

    print(f"\nBest val MSE: {best_val:.6f}")
    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
