import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP
from sequence_dataset import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
AE_WEIGHTS = "ae6.pth"
OUT_WEIGHTS = "dynamics_multistep_k7.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4

BATCH = 64
EPOCHS = 60
LR = 1e-3
GRAD_CLIP = 1.0

# short multi-step horizon first
K = 4

SEED = 42
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multistep_latent_loss(dyn, z0, acts, z_target):
    """
    dyn: dynamics model
    z0: (B, latent_dim)
    acts: (B, K) long
    z_target: (B, K, latent_dim)

    absolute dynamics version:
        z_{t+1} = dyn(z_t, a_t)
    """
    z = z0
    preds = []

    for t in range(K):
        a_t = acts[:, t]
        z = dyn(z, a_t)
        preds.append(z)

    z_pred = torch.stack(preds, dim=1)   # (B, K, latent_dim)
    loss = F.mse_loss(z_pred, z_target)
    return loss, z_pred


def split_dataset(ds, val_fraction=0.1):
    n = len(ds)
    n_val = max(1, int(val_fraction * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(SEED)
    return random_split(ds, [n_train, n_val], generator=g)


def main():
    set_seed(SEED)
    print("DEVICE:", DEVICE)
    print("Building sequence datasets...")

    ds1 = SequenceDataset("dataset/dataset1", K=K)
    ds2 = SequenceDataset("dataset/dataset2", K=K)
    ds3 = SequenceDataset("dataset/dataset3", K=K)
    ds5 = SequenceDataset("dataset/dataset5", K=K)

    print("Sequence samples ds1:", len(ds1))
    print("Sequence samples ds2:", len(ds2))
    print("Sequence samples ds3:", len(ds3))
    print("Sequence samples ds5:", len(ds5))

    train1, val1 = split_dataset(ds1)
    train2, val2 = split_dataset(ds2)
    train3, val3 = split_dataset(ds3)
    train5, val5 = split_dataset(ds5)

    train_ds = ConcatDataset([train1, train2, train3, train5])
    val_ds   = ConcatDataset([val1, val2, val3, val5])

    print("Train samples total:", len(train_ds))
    print("Val samples total  :", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # -------------------------
    # Load frozen AE
    # -------------------------
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # -------------------------
    # Dynamics model
    # -------------------------
    dyn = DynamicsMLP(
        latent_dim=LATENT_DIM,
        num_actions=NUM_ACTIONS,
        hidden=256
    ).to(DEVICE)

    opt = optim.Adam(dyn.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    # =========================
    # TRAIN
    # =========================
    for epoch in range(EPOCHS):
        dyn.train()
        total = 0.0

        for batch_i, (frames, acts) in enumerate(train_loader):
            # frames: (B, K+1, 3, 64, 64)
            # acts  : (B, K)
            frames = frames.to(DEVICE)
            acts = acts.to(DEVICE)
            B = frames.shape[0]

            with torch.no_grad():
                flat = frames.reshape(B * (K + 1), 3, 64, 64)
                z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

            z0 = z_all[:, 0]         # (B, latent_dim)
            z_target = z_all[:, 1:]  # (B, K, latent_dim)

            loss, z_pred = multistep_latent_loss(dyn, z0, acts, z_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()

            total += loss.item()

            if batch_i % 100 == 0:
                with torch.no_grad():
                    delta_true = (z_target[:, 0] - z0).norm(dim=1).mean().item()
                    delta_pred = (z_pred[:, 0] - z0).norm(dim=1).mean().item()
                print(
                    f"epoch {epoch:02d} batch {batch_i:04d}/{len(train_loader)} "
                    f"loss {loss.item():.6f}  Δtrue {delta_true:.4f}  Δpred {delta_pred:.4f}"
                )

        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        # =========================
        # VALIDATION
        # =========================
        dyn.eval()
        vtotal = 0.0

        with torch.no_grad():
            for frames, acts in val_loader:
                frames = frames.to(DEVICE)
                acts = acts.to(DEVICE)
                B = frames.shape[0]

                flat = frames.reshape(B * (K + 1), 3, 64, 64)
                z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

                z0 = z_all[:, 0]
                z_target = z_all[:, 1:]

                vloss, _ = multistep_latent_loss(dyn, z0, acts, z_target)
                vtotal += vloss.item()

        val_loss = vtotal / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(dyn.state_dict(), OUT_WEIGHTS)
    print("Saved", OUT_WEIGHTS)

    # =========================
    # SAVE LOSS CURVE
    # =========================
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Multi-step latent MSE")
    plt.title(f"Multi-step Dynamics Training (K={K})")
    plt.legend()
    plt.tight_layout()

    out_plot = os.path.join(OUT_DIR, f"dynamics_multistep_k{K}_loss_ds5.png")
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print("Saved loss curve:", out_plot)


if __name__ == "__main__":
    main()