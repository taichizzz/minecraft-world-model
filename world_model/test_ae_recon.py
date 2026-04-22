import os
import glob
import random
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from model import AutoEncoder

# ---- Config ----
DATASET_DIR = "dataset/dataset3_turn"

# List of AE checkpoints to render side-by-side. Missing files are skipped.
# Row 0 in the output grid is originals; each subsequent row is one AE's recon.
AE_CHECKPOINTS = ["aeturn1.pth", "aeturn2.pth", "ae4.pth"]

OUT_DIR     = "world_model_out/April23nd"
N_SAMPLES   = 8                  # number of frames to visualize
SEED        = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


def to_uint8_img(chw_tensor: torch.Tensor) -> np.ndarray:
    chw_tensor = chw_tensor.detach().clamp(0, 1).cpu()
    img = (chw_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def make_grid_row(images_uint8, pad=4):
    h, w, _ = images_uint8[0].shape
    grid_w = len(images_uint8) * w + (len(images_uint8) - 1) * pad
    canvas = np.zeros((h, grid_w, 3), dtype=np.uint8)
    x = 0
    for im in images_uint8:
        canvas[:, x:x + w, :] = im
        x += w + pad
    return canvas


def stack_rows(rows, pad=6):
    """Stack rows (all same width) vertically with padding."""
    h = rows[0].shape[0]
    w = rows[0].shape[1]
    total_h = len(rows) * h + (len(rows) - 1) * pad
    canvas = np.zeros((total_h, w, 3), dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + h, :, :] = r
        y += h + pad
    return canvas


def load_ae(ae_weights):
    model = AutoEncoder(latent_dim=128).to(DEVICE)
    state = torch.load(ae_weights, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    T = obs.shape[0]
    idxs = sorted(random.sample(range(T), k=min(N_SAMPLES, T)))
    frames = obs[idxs]
    print("sample idxs:", idxs)

    batch = torch.from_numpy(frames).float() / 255.0
    batch = batch.permute(0, 3, 1, 2).to(DEVICE)

    originals = [to_uint8_img(batch[i]) for i in range(batch.shape[0])]
    row0 = make_grid_row(originals, pad=4)

    rows = [row0]
    labels = ["originals"]

    for ae_weights in AE_CHECKPOINTS:
        if not os.path.exists(ae_weights):
            print(f"SKIP {ae_weights} (not found)")
            continue
        model = load_ae(ae_weights)
        with torch.no_grad():
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch).item()
        print(f"{ae_weights}: pixel MSE = {loss:.6f}")

        recons = [to_uint8_img(recon[i]) for i in range(recon.shape[0])]
        rows.append(make_grid_row(recons, pad=4))
        labels.append(ae_weights)

    grid = stack_rows(rows, pad=6)

    ep_name = os.path.basename(ep_path).replace(".npz", "")
    ds_name = os.path.basename(DATASET_DIR.rstrip("/"))
    out_path = os.path.join(OUT_DIR, f"ae_recon_compare_{ds_name}_{ep_name}.png")
    Image.fromarray(grid).save(out_path)

    print("\nSaved:", out_path)
    print("Row order (top -> bottom):", labels)


if __name__ == "__main__":
    main()
