"""
Retrain the autoencoder on human data from all three environments.

The current aeturn3.pth was trained on random-agent data from env3 only.
This version trains on human demonstrations from env1, env2, and env3 so
the encoder produces meaningful latents regardless of which room the agent
is in.

Uses the same architecture and loss terms as train_ae_new.py:
  - Reconstruction (MSE)
  - Smoothness (consecutive-frame latents stay close)
  - Contrastive (InfoNCE: unrelated frames get pushed apart)

Warmstarts from aeturn3.pth so we don't lose the structure it already
learned from env3; the new data just teaches it to generalize.

Saves ae_multienv.pth and encoder_multienv.pth.
"""
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

from model import AutoEncoder
from transition_dataset import TransitionDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4

# Loss weights (same as train_ae_new.py)
LAMBDA_RECON = 1.0
LAMBDA_SMOOTH = 0.001
LAMBDA_CONTRAST = 1.0
TEMPERATURE = 0.1

# Warmstart from env3-only AE
WARMSTART = "aeturn3.pth"

SAVE_AE = "ae_multienv.pth"
SAVE_ENCODER = "encoder_multienv.pth"

VAL_FRAC = 0.1
SEED = 42

OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ============= DATASETS =============
DATASET_DIRS = [
    "dataset/dataset_1_human",    # env1: simple room, no obstacles
    "dataset/dataset_2_human",    # env2: env3 walls, no obstacles
    "dataset/dataset_3_human2",   # env3: walls + floor obstacles (no idle)
]

print("Loading datasets...")
datasets = []
for d in DATASET_DIRS:
    ds = TransitionDataset(d)
    print(f"  {d:30s}: {len(ds)} transitions")
    datasets.append(ds)

combined = ConcatDataset(datasets)
print(f"  {'combined':30s}: {len(combined)} transitions")

# Train/val split
torch.manual_seed(SEED)
n = len(combined)
n_val = max(1, int(VAL_FRAC * n))
n_train = n - n_val
train_ds, val_ds = random_split(combined, [n_train, n_val])
print(f"  train={n_train}  val={n_val}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0
)

# ============= MODEL =============
model = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
if WARMSTART and os.path.isfile(WARMSTART):
    model.load_state_dict(torch.load(WARMSTART, map_location=DEVICE))
    print(f"Warmstarted from {WARMSTART}")
else:
    print("Training from scratch (no warmstart)")

optimizer = optim.Adam(model.parameters(), lr=LR)


def info_nce_loss(z_t, z_tp1, temperature):
    """Symmetric InfoNCE — same as train_ae_new.py."""
    z_t_n = F.normalize(z_t, dim=1)
    z_tp1_n = F.normalize(z_tp1, dim=1)
    logits = z_t_n @ z_tp1_n.t() / temperature
    labels = torch.arange(z_t.shape[0], device=z_t.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    tot_recon = tot_smooth = tot_contrast = tot = 0.0
    n = 0

    for img_t, _, img_tp1 in loader:
        img_t = img_t.to(DEVICE)
        img_tp1 = img_tp1.to(DEVICE)

        if train:
            recon_t, z_t = model(img_t)
            recon_tp1, z_tp1 = model(img_tp1)

            recon_loss = F.mse_loss(recon_t, img_t) + F.mse_loss(recon_tp1, img_tp1)
            smooth_loss = F.mse_loss(z_tp1, z_t)
            contrast_loss = info_nce_loss(z_t, z_tp1, TEMPERATURE)

            loss = (
                LAMBDA_RECON * recon_loss
                + LAMBDA_SMOOTH * smooth_loss
                + LAMBDA_CONTRAST * contrast_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                recon_t, z_t = model(img_t)
                recon_tp1, z_tp1 = model(img_tp1)

                recon_loss = F.mse_loss(recon_t, img_t) + F.mse_loss(recon_tp1, img_tp1)
                smooth_loss = F.mse_loss(z_tp1, z_t)
                contrast_loss = info_nce_loss(z_t, z_tp1, TEMPERATURE)

                loss = (
                    LAMBDA_RECON * recon_loss
                    + LAMBDA_SMOOTH * smooth_loss
                    + LAMBDA_CONTRAST * contrast_loss
                )

        tot_recon += recon_loss.item()
        tot_smooth += smooth_loss.item()
        tot_contrast += contrast_loss.item()
        tot += loss.item()
        n += 1

    return tot / n, tot_recon / n, tot_smooth / n, tot_contrast / n


# ============= TRAIN =============
print(f"\nDevice: {DEVICE}  |  epochs: {EPOCHS}  |  batch: {BATCH_SIZE}")
print(f"lambdas: recon={LAMBDA_RECON} smooth={LAMBDA_SMOOTH} "
      f"contrast={LAMBDA_CONTRAST} | temperature={TEMPERATURE}\n")

best_val = float("inf")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    t_loss, t_recon, t_smooth, t_contrast = run_epoch(train_loader, train=True)
    v_loss, v_recon, v_smooth, v_contrast = run_epoch(val_loader, train=False)

    train_losses.append(t_loss)
    val_losses.append(v_loss)

    marker = ""
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), SAVE_AE)
        torch.save(model.encoder.state_dict(), SAVE_ENCODER)
        marker = "  <- best"

    print(
        f"Epoch {epoch:02d} | "
        f"train {t_loss:.4f} (r={t_recon:.4f} s={t_smooth:.4f} c={t_contrast:.4f}) | "
        f"val {v_loss:.4f} (r={v_recon:.4f} s={v_smooth:.4f} c={v_contrast:.4f})"
        f"{marker}"
    )

print(f"\nBest val loss: {best_val:.4f}")
print(f"Saved {SAVE_AE} and {SAVE_ENCODER}")

# ============= PLOT =============
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("total loss")
plt.title("AE multi-env training (env1 + env2 + env3)")
plt.legend()
plt.tight_layout()
out = os.path.join(OUT_DIR, "ae_multienv_loss.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved plot: {out}")

print("\nNext steps:")
print("  1. Retrain dynamics on all envs with new AE: train_dynamics_multienv.py")
print("  2. Retrain V head on all envs with new AE: train_value_head_multienv.py")
print("  3. Update agent_map_step3.py: AE_WEIGHTS = 'ae_multienv.pth'")
