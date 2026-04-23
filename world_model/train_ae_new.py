"""
AE training with a contrastive term to fix latent collapse (rand/cons ~= 1.3).

On top of the old objective (reconstruction + light smoothness) we add an
InfoNCE / NT-Xent contrastive term. For each minibatch of transitions
(img_t, img_tp1):
  - positives: (z_t[i], z_tp1[i])  -> same index, same moment in time
  - negatives: every OTHER sample z in the batch
This explicitly tells the AE: "unrelated scenes must have far-apart latents."

Saves aeturn3.pth and encoderturn3.pth.

After training, run:
    python measure_latent_smoothness.py     # expect rand/cons to jump well above 1.3
and if it looks good, retrain dynamics on aeturn3 before running the planner.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from model import AutoEncoder
from transition_dataset import TransitionDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4

# loss weights
LAMBDA_RECON = 1.0          # pixel reconstruction (primary)
LAMBDA_SMOOTH = 0.001       # light coupling of consecutive frames (safety vs. blowout)
LAMBDA_CONTRAST = 1.0       # push unrelated scenes apart (the fix)

# InfoNCE temperature: smaller -> harsher negatives
TEMPERATURE = 0.1

# warmstart from the latest AE. Set to None to train from scratch.
WARMSTART = "aeturn2.pth"
SAVE_AE = "aeturn3.pth"
SAVE_ENCODER = "encoderturn3.pth"

# ============= DATASETS =============
datasets = [
    TransitionDataset("dataset/dataset1"),
    TransitionDataset("dataset/dataset2"),
    TransitionDataset("dataset/dataset3"),
    TransitionDataset("dataset/dataset6"),
    TransitionDataset("dataset/dataset2_turn"),
    TransitionDataset("dataset/dataset3_turn"),
]
combined = ConcatDataset(datasets)
for name, ds in zip(
    ["dataset1", "dataset2", "dataset3", "dataset6",
     "dataset2_turn", "dataset3_turn"],
    datasets,
):
    print(f"{name:16s}: {len(ds)} transitions")
print(f"combined         : {len(combined)} transitions")

loader = DataLoader(
    combined, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
)

# ============= MODEL =============
model = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
if WARMSTART is not None:
    model.load_state_dict(torch.load(WARMSTART, map_location=DEVICE))
    print(f"Warmstarted from {WARMSTART}")

optimizer = optim.Adam(model.parameters(), lr=LR)


def info_nce_loss(z_t, z_tp1, temperature):
    """
    Symmetric InfoNCE.

    For anchor z_t[i], positive is z_tp1[i], negatives are all other z_tp1[j].
    Also computes the reverse direction (anchor z_tp1, positives z_t).

    Operates on L2-normalized features so only direction matters here;
    reconstruction loss still owns magnitude.
    """
    z_t_n = F.normalize(z_t, dim=1)
    z_tp1_n = F.normalize(z_tp1, dim=1)
    logits = z_t_n @ z_tp1_n.t() / temperature  # (B, B) cosine-similarity matrix
    labels = torch.arange(z_t.shape[0], device=z_t.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


# ============= TRAIN =============
print(f"\nDevice: {DEVICE}  |  epochs: {EPOCHS}  |  batch: {BATCH_SIZE}")
print(f"lambdas: recon={LAMBDA_RECON} smooth={LAMBDA_SMOOTH} "
      f"contrast={LAMBDA_CONTRAST} | temperature={TEMPERATURE}\n")

for epoch in range(EPOCHS):
    model.train()
    tot_recon = tot_smooth = tot_contrast = tot = 0.0
    n = 0

    for img_t, _, img_tp1 in loader:
        img_t = img_t.to(DEVICE)
        img_tp1 = img_tp1.to(DEVICE)

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

        tot_recon += recon_loss.item()
        tot_smooth += smooth_loss.item()
        tot_contrast += contrast_loss.item()
        tot += loss.item()
        n += 1

    print(
        f"Epoch {epoch:02d} | loss={tot/n:.4f} "
        f"| recon={tot_recon/n:.4f} "
        f"| smooth={tot_smooth/n:.4f} "
        f"| contrast={tot_contrast/n:.4f}"
    )

torch.save(model.state_dict(), SAVE_AE)
torch.save(model.encoder.state_dict(), SAVE_ENCODER)
print(f"\nSaved {SAVE_AE} and {SAVE_ENCODER}")
print("\nNext steps:")
print("  1. Run measure_latent_smoothness.py (add 'aeturn3.pth' to AE_CHECKPOINTS).")
print("     Target: rand/cons well above 1.3 (aim for 3-10x).")
print("  2. If healthy, retrain dynamics on aeturn3.pth, then re-run planner.py.")
