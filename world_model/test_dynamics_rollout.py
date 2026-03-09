# # world_model/test_dynamics_rollout.py
# import os
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from model import AutoEncoder
# from dynamics_model import DynamicsMLP

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SEED = 42

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# DATASET_DIR = "dataset"
# AE_WEIGHTS = "ae.pth"
# DYN_WEIGHTS = "dynamics2.pth"
# # DYN_WEIGHTS = "dynamics_multistep4.pth"

# LATENT_DIM = 128
# NUM_ACTIONS = 4
# USE_RESIDUAL = True

# # rollout length
# K = 16

# OUT_DIR = "world_model_out"
# os.makedirs(OUT_DIR, exist_ok=True)

# def load_episode(npz_path):
#     data = np.load(npz_path)
#     obs = data["obs"]          # (T,64,64,3) uint8
#     actions = data["actions"]  # (T,) int
#     return obs, actions

# def to_torch_img(batch_hwc_uint8):
#     # (B,H,W,C) uint8 -> (B,C,H,W) float in [0,1]
#     x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
#     x = x.permute(0, 3, 1, 2).contiguous()
#     return x

# def make_grid_row(imgs_bchw):
#     # imgs in [0,1], (B,3,64,64)
#     imgs = imgs_bchw.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,64,64,3)
#     return imgs

# def main():
#     files = sorted([f for f in os.listdir(DATASET_DIR) if f.startswith("episode_") and f.endswith(".npz")])
#     if not files:
#         raise RuntimeError(f"No episodes found in {DATASET_DIR}")
#     ep_file = random.choice(files)
#     ep_path = os.path.join(DATASET_DIR, ep_file)
#     print("Using episode:", ep_path)

#     obs, actions = load_episode(ep_path)
#     T = obs.shape[0]
#     if T < K + 1:
#         raise RuntimeError(f"Episode too short: T={T}, need at least {K+1}")

#     t0 = random.randint(0, T - (K + 1))
#     print("t0:", t0, "rollout K:", K)

#     gt_frames = obs[t0:t0+K+1]          # (K+1,64,64,3)
#     act_seq   = actions[t0:t0+K]        # (K,)

#     #loading ae
#     ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
#     ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
#     ae.eval()

#     #loading dynamics
#     dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
#     dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
#     dyn.eval()

#     #encode ground truth to latents
#     with torch.no_grad():
#         gt_imgs = to_torch_img(gt_frames).to(DEVICE)  
#         z_gt = ae.encoder(gt_imgs)                   

#     # rollout in latent space
#     z_pred_list = [z_gt[0]]  # start from true z0
#     with torch.no_grad():
#         z = z_gt[0].unsqueeze(0)  # (1, latent)
#         for i in range(K):
#             a = torch.tensor([int(act_seq[i])], dtype=torch.long, device=DEVICE)

#             if USE_RESIDUAL:
#                 delta = dyn(z, a)
#                 delta = torch.tanh(delta) * 0.2
#                 z = z + delta
#             else:
#                 z = dyn(z, a)

#             z_pred_list.append(z.squeeze(0))
#     z_pred = torch.stack(z_pred_list, dim=0)

#     # decode
#     with torch.no_grad():
#         recon_gt   = ae.decoder(z_gt)  
#         recon_pred = ae.decoder(z_pred)  

#     # compute rollout error in latent + pixel
#     with torch.no_grad():
#         latent_mse = F.mse_loss(z_pred, z_gt).item()
#         pixel_mse  = F.mse_loss(recon_pred, recon_gt).item()
#         gt_imgs_chw = to_torch_img(gt_frames).to(DEVICE)
#         pixel_mse_true = F.mse_loss(recon_pred, gt_imgs_chw).item()
    


#     print(f"Latent MSE over rollout: {latent_mse:.6f}")
#     print(f"Pixel MSE over rollout (decoded): {pixel_mse:.6f}")
#     print(f"Pixel MSE vs TRUE frames: {pixel_mse_true:.6f}")

#     # plot grid: top = GT recon, bottom = Pred recon
#     top = make_grid_row(recon_gt)      # (K+1,64,64,3)
#     bot = make_grid_row(recon_pred)    # (K+1,64,64,3)

#     fig, axes = plt.subplots(2, K+1, figsize=(1.6*(K+1), 3.2))
#     for i in range(K+1):
#         axes[0, i].imshow(top[i])
#         axes[0, i].axis("off")
#         axes[0, i].set_title(f"t{i}", fontsize=8)

#         axes[1, i].imshow(bot[i])
#         axes[1, i].axis("off")

#     axes[0, 0].set_ylabel("GT", fontsize=10)
#     axes[1, 0].set_ylabel("Pred", fontsize=10)
#     plt.tight_layout()

#     out_path = os.path.join(OUT_DIR, "dyn_rollout_combined_1.png")
#     # out_path = os.path.join(OUT_DIR, "dyn_multistep_rollout4.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close(fig)
#     print("Saved:", out_path)


# if __name__ == "__main__":
#     main()

# world_model/test_dynamics_rollout.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset/dataset2"
AE_WEIGHTS = "ae3.pth"
DYN_WEIGHTS = "dynamics_balanced2.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4

# rollout length
K = 16

OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)

def load_episode(npz_path):
    data = np.load(npz_path)
    obs = data["obs"]          # (T,64,64,3) uint8
    actions = data["actions"]  # (T,) int
    return obs, actions

def to_torch_img(batch_hwc_uint8):
    # (B,H,W,C) uint8 -> (B,C,H,W) float in [0,1]
    x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

def make_grid_row(imgs_bchw):
    # imgs in [0,1], (B,3,64,64)
    imgs = imgs_bchw.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,64,64,3)
    return imgs

def main():
    # pick a random episode file
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.startswith("episode_") and f.endswith(".npz")])
    # files = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "episode_*.npz"), recursive=True))
    if not files:
        raise RuntimeError(f"No episodes found in {DATASET_DIR}")
    ep_file = random.choice(files)
    ep_path = os.path.join(DATASET_DIR, ep_file)
    # ep_path = random.choice(files)
    print("Using episode:", ep_path)
    print("From folder:", os.path.dirname(ep_path))

    obs, actions = load_episode(ep_path)
    T = obs.shape[0]
    if T < K + 1:
        raise RuntimeError(f"Episode too short: T={T}, need at least {K+1}")

    t0 = random.randint(0, T - (K + 1))
    print("t0:", t0, "rollout K:", K)

    gt_frames = obs[t0:t0+K+1]          # (K+1,64,64,3)
    act_seq   = actions[t0:t0+K]        # (K,)

    #loading ae
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    #loading dynamics
    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    #encode ground truth to latents
    with torch.no_grad():
        gt_imgs = to_torch_img(gt_frames).to(DEVICE)  
        z_gt = ae.encoder(gt_imgs)                   

    # rollout in latent space
    z_pred_list = [z_gt[0]]  # start from true z0
    with torch.no_grad():
        z = z_gt[0].unsqueeze(0)  # (1,128)
        for i in range(K):
            a = torch.tensor([int(act_seq[i])], dtype=torch.long, device=DEVICE)  # (1,)
            z = dyn(z, a)  # (1,128)
            # delta = dyn(z, a)
            # z = z + delta
            z_pred_list.append(z.squeeze(0))
    z_pred = torch.stack(z_pred_list, dim=0)  # (K+1,128)

    # decode both sequences to images
    with torch.no_grad():
        recon_gt   = ae.decoder(z_gt)    # (K+1,3,64,64)
        recon_pred = ae.decoder(z_pred)  # (K+1,3,64,64)

    # compute rollout error in latent + pixel
    with torch.no_grad():
        latent_mse = F.mse_loss(z_pred, z_gt).item()
        pixel_mse  = F.mse_loss(recon_pred, recon_gt).item()

    print(f"Latent MSE over rollout: {latent_mse:.6f}")
    print(f"Pixel MSE over rollout (decoded): {pixel_mse:.6f}")

    print("z_pred norm:", z_pred.norm(dim=1))
    print("z_gt norm:", z_gt.norm(dim=1))

    # plot grid: top = GT recon, bottom = Pred recon
    top = make_grid_row(recon_gt)      # (K+1,64,64,3)
    bot = make_grid_row(recon_pred)    # (K+1,64,64,3)

    fig, axes = plt.subplots(2, K+1, figsize=(1.6*(K+1), 3.2))
    for i in range(K+1):
        axes[0, i].imshow(top[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t{i}", fontsize=8)

        axes[1, i].imshow(bot[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("GT", fontsize=10)
    axes[1, 0].set_ylabel("Pred", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "dyn_rollout_balanced_test4.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()