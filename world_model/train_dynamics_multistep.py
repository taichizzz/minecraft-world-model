import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP
from sequence_dataset import SequenceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset"
AE_WEIGHTS = "ae.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4

BATCH = 64
EPOCHS = 30
LR = 1e-4
K = 8
LAMBDA_PIX = 0.1

GAMMA = 0.9   
GRAD_CLIP = 1.0 
SEED = 42
OUT_WEIGHTS = "dynamics_multistep4.pth"

OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def rollout_loss(dyn, z0, acts, z_target):
    """
    dyn: model
    z0: (B, latent)
    acts: (B, K) long
    z_target: (B, K, latent)
    returns: scalar loss
    """
    z = z0
    loss = 0.0

    for t in range(K):
        a_t = acts[:, t]
        delta = dyn(z, a_t)
        delta = torch.tanh(delta) * 0.2
        z = z + delta

        w = (GAMMA ** t)
        loss = loss + w * F.mse_loss(z, z_target[:, t])

    return loss


# def main():
#     print(f"DEVICE: {DEVICE}")
#     print("Building multi-step dataset...")
#     ds = SequenceDataset(DATASET_DIR, K=K)
#     print("Sequence samples:", len(ds))

#     n = len(ds)
#     n_val = max(1, int(0.1 * n))
#     n_train = n - n_val
#     train_ds, val_ds = random_split(ds, [n_train, n_val])

#     train_loader = DataLoader(
#         train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0
#     )

#     ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
#     ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
#     ae.eval()
#     for p in ae.parameters():
#         p.requires_grad = False

#     dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
#     opt = optim.AdamW(dyn.parameters(), lr=LR)

#     train_losses, val_losses = [], []

#     for epoch in range(EPOCHS):
#         dyn.train()
#         total = 0.0

#         for batch_i, (frames, acts) in enumerate(train_loader):
#             frames = frames.to(DEVICE)  # (B, K+1, 3, 64, 64)
#             acts = acts.to(DEVICE)      # (B, K)
#             B = frames.shape[0]

#             with torch.no_grad():
#                 flat = frames.reshape(B * (K + 1), 3, 64, 64)
#                 z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

#             z = z_all[:, 0]          # z_t0
#             z_target = z_all[:, 1:]  # z_t1..z_tK  (B, K, latent)

#             preds = []
#             for t in range(K):
#                 a_t = acts[:, t]

#                 delta = dyn(z, a_t)
#                 delta = torch.tanh(delta) * 0.3   
#                 z = z + delta

#                 preds.append(z)

#             z_pred = torch.stack(preds, dim=1)  
#             # loss = F.mse_loss(z_pred, z_target)
#             loss = F.smooth_l1_loss(z_pred, z_target)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()

#             #gradient clipping
#             torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)

#             opt.step()
#             total += loss.item()

#             if batch_i % 100 == 0:
#                 print(f"  epoch {epoch:02d} batch {batch_i:04d}/{len(train_loader)} loss {loss.item():.6f}")

#         train_loss = total / len(train_loader)
#         train_losses.append(train_loss)

#         dyn.eval()
#         vtotal = 0.0
#         with torch.no_grad():
#             for frames, acts in val_loader:
#                 frames = frames.to(DEVICE)
#                 acts = acts.to(DEVICE)
#                 B = frames.shape[0]

#                 flat = frames.reshape(B * (K + 1), 3, 64, 64)
#                 z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

#                 z = z_all[:, 0]
#                 z_target = z_all[:, 1:]

#                 preds = []
#                 for t in range(K):
#                     a_t = acts[:, t]
#                     z = dyn(z, a_t)
#                     preds.append(z)

#                 z_pred = torch.stack(preds, dim=1)
#                 vtotal += F.mse_loss(z_pred, z_target).item()

#         val_loss = vtotal / len(val_loader)
#         val_losses.append(val_loss)

#         print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

#     out_weights = "dynamics_multistep2.pth"
#     torch.save(dyn.state_dict(), out_weights)
#     print("Saved", out_weights)

#     plt.figure(figsize=(6, 4))
#     plt.plot(train_losses, label="Train")
#     plt.plot(val_losses, label="Val")
#     plt.xlabel("Epoch")
#     plt.ylabel("Latent MSE")
#     plt.title(f"Multi-step Dynamics (K={K})")
#     plt.legend()
#     plt.tight_layout()
#     out_plot = os.path.join(OUT_DIR, "dynamics_multistep_loss2.png")
#     plt.savefig(out_plot, dpi=200)
#     plt.close()
#     print("Saved loss curve:", out_plot)


# if __name__ == "__main__":
#     main()

def main():
    set_seed(SEED)
    print(f"DEVICE: {DEVICE}")
    print("Building multi-step dataset...")
    ds = SequenceDataset(DATASET_DIR, K=K)
    print("Sequence samples:", len(ds))

    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
    opt = optim.AdamW(dyn.parameters(), lr=LR)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        dyn.train()
        total = 0.0

        for batch_i, (frames, acts) in enumerate(train_loader):
            frames = frames.to(DEVICE)  # (B, K+1, 3, 64, 64)
            acts = acts.to(DEVICE)      # (B, K)
            B = frames.shape[0]

            with torch.no_grad():
                flat = frames.reshape(B * (K + 1), 3, 64, 64)
                z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

            z0 = z_all[:, 0]         # (B, latent)
            z_target = z_all[:, 1:]  # (B, K, latent)

            # loss = rollout_loss(dyn, z0, acts, z_target)

            # ----- latent rollout -----
            z = z0
            latent_loss = 0.0
            z_preds = []

            for t in range(K):
                a_t = acts[:, t]
                delta = dyn(z, a_t)
                delta = torch.tanh(delta) * 0.2
                z = z + delta

                w = (GAMMA ** t)
                latent_loss = latent_loss + w * F.mse_loss(z, z_target[:, t])
                z_preds.append(z)

            z_pred = torch.stack(z_preds, dim=1)  # (B, K, latent)

            true_next_frames = frames[:, 1:]  # (B, K, 3, 64, 64)

            flat_latent = z_pred.reshape(B * K, LATENT_DIM)
            decoded = ae.decoder(flat_latent).view(B, K, 3, 64, 64)

            pixel_loss = F.mse_loss(decoded, true_next_frames)

            loss = latent_loss + LAMBDA_PIX * pixel_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()

            total += loss.item()

            if batch_i % 100 == 0:
                print(f"  epoch {epoch:02d} batch {batch_i:04d}/{len(train_loader)} loss {loss.item():.6f}")

        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        # -------- val --------
        # -------- val --------
        dyn.eval()
        vtotal = 0.0
        with torch.no_grad():
            for frames, acts in val_loader:
                frames = frames.to(DEVICE)
                acts = acts.to(DEVICE)
                B = frames.shape[0]

                flat = frames.reshape(B * (K + 1), 3, 64, 64)
                z_all = ae.encoder(flat).view(B, K + 1, LATENT_DIM)

                z = z_all[:, 0]
                z_target = z_all[:, 1:]  # (B, K, latent)

                # rollout latent
                latent_loss = 0.0
                z_preds = []
                for t in range(K):
                    a_t = acts[:, t]
                    delta = dyn(z, a_t)
                    delta = torch.tanh(delta) * 0.2
                    z = z + delta

                    w = (GAMMA ** t)
                    latent_loss += w * F.mse_loss(z, z_target[:, t])
                    z_preds.append(z)

                z_pred = torch.stack(z_preds, dim=1)  # (B, K, latent)

                # rollout pixel
                true_next_frames = frames[:, 1:]  # (B, K, 3, 64, 64)
                decoded = ae.decoder(z_pred.reshape(B * K, LATENT_DIM)).view(B, K, 3, 64, 64)
                pixel_loss = F.mse_loss(decoded, true_next_frames)

                vloss = latent_loss + LAMBDA_PIX * pixel_loss
                vtotal += vloss.item()

        val_loss = vtotal / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    torch.save(dyn.state_dict(), OUT_WEIGHTS)
    print("Saved", OUT_WEIGHTS)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Discounted rollout MSE")
    plt.title(f"Multi-step Dynamics (K={K}, gamma={GAMMA}, delta_scale={0.2})")
    plt.legend()
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, "dynamics_multistep_loss3.png")
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print("Saved loss curve:", out_plot)


if __name__ == "__main__":
    main()