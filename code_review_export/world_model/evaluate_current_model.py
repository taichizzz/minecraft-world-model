import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset/dataset2"   
AE_WEIGHTS = "ae3.pth"
DYN_WEIGHTS = "dynamics_balanced2.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4
K = 16
OUT_DIR = "world_model_out"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_episode(npz_path):
    data = np.load(npz_path)
    obs = data["obs"]         
    actions = data["actions"]  
    return obs, actions

def to_torch_img(batch_hwc_uint8):
    x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

def make_grid_row(imgs_bchw):
    imgs = imgs_bchw.detach().cpu().permute(0, 2, 3, 1).numpy()
    return imgs

def main():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not files:
        raise RuntimeError(f"No episodes found in {DATASET_DIR}")

    ep_file = random.choice(files)
    ep_path = os.path.join(DATASET_DIR, ep_file)
    print("Using episode:", ep_path)

    obs, actions = load_episode(ep_path)
    T = obs.shape[0]

    if T < K + 1:
        raise RuntimeError(f"Episode too short: T={T}, need at least {K+1}")

    t0 = random.randint(0, T - (K + 1))
    print("t0:", t0, "rollout K:", K)

    gt_frames = obs[t0:t0+K+1]   
    act_seq   = actions[t0:t0+K] 

    # loading models
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    # encoding ground truth frames
    with torch.no_grad():
        gt_imgs = to_torch_img(gt_frames).to(DEVICE)   
        z_gt = ae.encoder(gt_imgs)                     

    z_norm = z_gt.norm(dim=1)
    delta_gt = z_gt[1:] - z_gt[:-1]
    delta_gt_norm = delta_gt.norm(dim=1)

    print("\n=== Latent Smoothness ===")
    print("Mean ||z_t||:", z_norm.mean().item())
    print("Mean ||z_{t+1} - z_t||:", delta_gt_norm.mean().item())
    print("Ratio (delta / z):", (delta_gt_norm.mean() / z_norm.mean()).item())

    one_step_preds = []
    with torch.no_grad():
        for i in range(K):
            z_t = z_gt[i].unsqueeze(0)   # (1,128)
            a_t = torch.tensor([int(act_seq[i])], dtype=torch.long, device=DEVICE)
            z_pred = dyn(z_t, a_t)
            one_step_preds.append(z_pred.squeeze(0))

    z_1step = torch.stack(one_step_preds, dim=0)   
    z_true_next = z_gt[1:]                         

    one_step_latent_mse = F.mse_loss(z_1step, z_true_next).item()

    with torch.no_grad():
        recon_1step = ae.decoder(z_1step)
        true_next_imgs = gt_imgs[1:]
        one_step_pixel_mse = F.mse_loss(recon_1step, true_next_imgs).item()

    print("\n=== 1-Step Dynamics ===")
    print(f"1-step latent MSE: {one_step_latent_mse:.6f}")
    print(f"1-step pixel MSE vs TRUE: {one_step_pixel_mse:.6f}")

    z_pred_list = [z_gt[0]]
    per_step_latent_mse = []
    per_step_pixel_mse = []

    with torch.no_grad():
        z = z_gt[0].unsqueeze(0)

        for i in range(K):
            a = torch.tensor([int(act_seq[i])], dtype=torch.long, device=DEVICE)
            z = dyn(z, a)
            z_pred_list.append(z.squeeze(0))

            # latent error 
            latent_err = F.mse_loss(z.squeeze(0), z_gt[i+1]).item()
            per_step_latent_mse.append(latent_err)

            # pixel error
            recon = ae.decoder(z)
            pixel_err = F.mse_loss(recon, gt_imgs[i+1:i+2]).item()
            per_step_pixel_mse.append(pixel_err)

    z_rollout = torch.stack(z_pred_list, dim=0)

    with torch.no_grad():
        recon_gt = ae.decoder(z_gt)
        recon_rollout = ae.decoder(z_rollout)

    rollout_latent_mse = F.mse_loss(z_rollout, z_gt).item()
    rollout_pixel_mse = F.mse_loss(recon_rollout, gt_imgs).item()

    print("\n=== Rollout ===")
    print(f"Rollout latent MSE: {rollout_latent_mse:.6f}")
    print(f"Rollout pixel MSE vs TRUE: {rollout_pixel_mse:.6f}")

    print("\nPer-step latent MSE:")
    print(per_step_latent_mse)

    print("\nPer-step pixel MSE:")
    print(per_step_pixel_mse)

   
    # rollout visualization
    top = make_grid_row(recon_gt)
    bot = make_grid_row(recon_rollout)

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

    out_img = os.path.join(OUT_DIR, "world_model_eval_rollout_ds2.png")
    plt.savefig(out_img, dpi=200)
    plt.close()
    print("\nSaved rollout image to:", out_img)

    # printing error curve
    plt.figure(figsize=(6,4))
    plt.plot(range(1, K+1), per_step_latent_mse, label="Latent MSE")
    plt.plot(range(1, K+1), per_step_pixel_mse, label="Pixel MSE")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title("Rollout Error Over Time")
    plt.legend()
    plt.tight_layout()

    out_curve = os.path.join(OUT_DIR, "world_model_eval_error_curve_ds2.png")
    plt.savefig(out_curve, dpi=200)
    plt.close()
    print("Saved error curve to:", out_curve)

if __name__ == "__main__":
    main()