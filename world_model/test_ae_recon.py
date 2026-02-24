import os
import glob
import random
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from model import AutoEncoder

# ---- Config ----
DATASET_DIR = "dataset"          # folder with episode_000000.npz ...
AE_WEIGHTS  = "ae.pth"           # saved by train_ae.py
OUT_DIR     = "world_model_out"
N_SAMPLES   = 8                  # number of frames to visualize
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

def to_uint8_img(chw_tensor: torch.Tensor) -> np.ndarray:
    """
    chw_tensor: (3,H,W) in [0,1]
    returns uint8 HxWx3
    """
    chw_tensor = chw_tensor.detach().clamp(0, 1).cpu()
    img = (chw_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img

def make_grid_row(images_uint8, pad=4):
    """
    images_uint8: list of HxWx3 (same size)
    returns a single HxWgridx3 image
    """
    h, w, _ = images_uint8[0].shape
    grid_w = len(images_uint8) * w + (len(images_uint8) - 1) * pad
    canvas = np.zeros((h, grid_w, 3), dtype=np.uint8)

    x = 0
    for im in images_uint8:
        canvas[:, x:x+w, :] = im
        x += w + pad
    return canvas

def main():
    episode_files = sorted(glob.glob(os.path.join(DATASET_DIR, "episode_*.npz")))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz found in {DATASET_DIR}")

    ep_path = random.choice(episode_files)
    print("Using episode:", ep_path)

    data = np.load(ep_path)
    obs = data["obs"]
    print("obs shape:", obs.shape, "dtype:", obs.dtype)

    if obs.ndim != 4 or obs.shape[-1] != 3:
        raise ValueError(f"Expected obs shape (T,H,W,3), got {obs.shape}")

    T, H, W, C = obs.shape
    if H != 64 or W != 64:
        print(f"Warning: expected 64x64, got {H}x{W}")

    idxs = random.sample(range(T), k=min(N_SAMPLES, T))
    idxs.sort()
    frames = obs[idxs]  # (N,H,W,3)
    print("sample idxs:", idxs)

    batch = torch.from_numpy(frames).float() / 255.0
    batch = batch.permute(0, 3, 1, 2)  # (N,3,H,W)
    batch = batch.to(DEVICE)

    model = AutoEncoder(latent_dim=128).to(DEVICE)
    state = torch.load(AE_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        recon, z = model(batch)
        loss = F.mse_loss(recon, batch).item()

    print("batch:", tuple(batch.shape), "recon:", tuple(recon.shape), "z:", tuple(z.shape))
    print(f"MSE on these samples: {loss:.6f}")

    originals = [to_uint8_img(batch[i]) for i in range(batch.shape[0])]
    recons    = [to_uint8_img(recon[i]) for i in range(recon.shape[0])]

    row1 = make_grid_row(originals, pad=4)
    row2 = make_grid_row(recons, pad=4)

    # stack vertically with padding
    pad = 6
    grid = np.zeros((row1.shape[0]*2 + pad, row1.shape[1], 3), dtype=np.uint8)
    grid[0:row1.shape[0], :, :] = row1
    grid[row1.shape[0] + pad:row1.shape[0]*2 + pad, :, :] = row2

    out_path = os.path.join(OUT_DIR, "ae_recon_grid.png")
    Image.fromarray(grid).save(out_path)
    print("Saved:", out_path)
    print("Top row = original, bottom row = reconstruction")

if __name__ == "__main__":
    main()
