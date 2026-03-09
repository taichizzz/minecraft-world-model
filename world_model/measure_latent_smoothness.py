import os
import random
import numpy as np
import torch

from model import AutoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset/dataset2"  
AE_WEIGHTS = "ae3.pth"             

LATENT_DIM = 128

def to_torch_img(batch_hwc_uint8):
    x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

def main():
    files = sorted([f for f in os.listdir(DATASET_DIR)
                    if f.startswith("episode_") and f.endswith(".npz")])
    if not files:
        raise RuntimeError("No episodes found")

    ep_file = random.choice(files)
    ep_path = os.path.join(DATASET_DIR, ep_file)
    print("Using episode:", ep_path)

    data = np.load(ep_path)
    obs = data["obs"]  # (T,64,64,3)
    T = obs.shape[0]

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    with torch.no_grad():
        imgs = to_torch_img(obs).to(DEVICE)
        z = ae.encoder(imgs)  # (T,128)

    z_norm = z.norm(dim=1)
    delta = z[1:] - z[:-1]
    delta_norm = delta.norm(dim=1)

    print("Mean ||z_t||:", z_norm.mean().item())
    print("Mean ||z_{t+1} - z_t||:", delta_norm.mean().item())
    print("Ratio (delta / z):", (delta_norm.mean() / z_norm.mean()).item())

if __name__ == "__main__":
    main()