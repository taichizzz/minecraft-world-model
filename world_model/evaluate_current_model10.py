import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset/dataset5"   # change to dataset2 when needed
AE_WEIGHTS = "ae6.pth"
# DYN_WEIGHTS = "dynamics_balanced2.pth"
DYN_WEIGHTS = "dynamics_multistep_k7.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4
K = 32
N_EPISODES = 10

OUT_DIR = "world_model_out/April6th"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# HELPERS
def load_episode(npz_path):
    data = np.load(npz_path)
    obs = data["obs"]          # (T,H,W,3)
    actions = data["actions"]  # (T,)
    return obs, actions

def to_torch_img(batch_hwc_uint8):
    x = torch.from_numpy(batch_hwc_uint8).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x

def evaluate_one_episode(ep_path, ae, dyn, K):
    obs, actions = load_episode(ep_path)
    T = obs.shape[0]
    if T < K + 1:
        return None

    t0 = random.randint(0, T - (K + 1))
    gt_frames = obs[t0:t0+K+1]   # (K+1,H,W,3)
    act_seq   = actions[t0:t0+K] # (K,)

    with torch.no_grad():
        gt_imgs = to_torch_img(gt_frames).to(DEVICE)
        z_gt = ae.encoder(gt_imgs)

    # Latent smoothness
    z_norm = z_gt.norm(dim=1)
    delta_gt = z_gt[1:] - z_gt[:-1]
    delta_gt_norm = delta_gt.norm(dim=1)
    smooth_ratio = (delta_gt_norm.mean() / z_norm.mean()).item()

    # 1-step dynamics
    one_step_preds = []
    with torch.no_grad():
        for i in range(K):
            z_t = z_gt[i].unsqueeze(0)
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

    # Rollout
    z_pred_list = [z_gt[0]]
    per_step_latent_mse = []
    per_step_pixel_mse = []

    with torch.no_grad():
        z = z_gt[0].unsqueeze(0)
        for i in range(K):
            a = torch.tensor([int(act_seq[i])], dtype=torch.long, device=DEVICE)
            z = dyn(z, a)
            z_pred_list.append(z.squeeze(0))

            latent_err = F.mse_loss(z.squeeze(0), z_gt[i+1]).item()
            per_step_latent_mse.append(latent_err)

            recon = ae.decoder(z)
            pixel_err = F.mse_loss(recon, gt_imgs[i+1:i+2]).item()
            per_step_pixel_mse.append(pixel_err)

    z_rollout = torch.stack(z_pred_list, dim=0)

    with torch.no_grad():
        recon_rollout = ae.decoder(z_rollout)

    rollout_latent_mse = F.mse_loss(z_rollout, z_gt).item()
    rollout_pixel_mse = F.mse_loss(recon_rollout, gt_imgs).item()

    return {
        "episode": ep_path,
        "t0": t0,
        "smooth_ratio": smooth_ratio,
        "one_step_latent_mse": one_step_latent_mse,
        "one_step_pixel_mse": one_step_pixel_mse,
        "rollout_latent_mse": rollout_latent_mse,
        "rollout_pixel_mse": rollout_pixel_mse,
        "per_step_latent_mse": per_step_latent_mse,
        "per_step_pixel_mse": per_step_pixel_mse,
    }

def main():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not files:
        raise RuntimeError(f"No episodes found in {DATASET_DIR}")

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    chosen_files = random.sample(files, min(N_EPISODES, len(files)))
    results = []

    for f in chosen_files:
        ep_path = os.path.join(DATASET_DIR, f)
        print("Evaluating:", ep_path)
        out = evaluate_one_episode(ep_path, ae, dyn, K)
        if out is not None:
            results.append(out)

    if not results:
        raise RuntimeError("No valid episodes were evaluated.")

    mean_smooth = np.mean([r["smooth_ratio"] for r in results])
    mean_1_lat = np.mean([r["one_step_latent_mse"] for r in results])
    mean_1_pix = np.mean([r["one_step_pixel_mse"] for r in results])
    mean_roll_lat = np.mean([r["rollout_latent_mse"] for r in results])
    mean_roll_pix = np.mean([r["rollout_pixel_mse"] for r in results])

    print("\n=== AVERAGED RESULTS OVER", len(results), "EPISODES ===")
    print(f"Mean smoothness ratio      : {mean_smooth:.6f}")
    print(f"Mean 1-step latent MSE     : {mean_1_lat:.6f}")
    print(f"Mean 1-step pixel MSE      : {mean_1_pix:.6f}")
    print(f"Mean rollout latent MSE    : {mean_roll_lat:.6f}")
    print(f"Mean rollout pixel MSE     : {mean_roll_pix:.6f}")


    mean_per_step_latent = np.mean(
        np.array([r["per_step_latent_mse"] for r in results]), axis=0
    )
    mean_per_step_pixel = np.mean(
        np.array([r["per_step_pixel_mse"] for r in results]), axis=0
    )

    print("\nMean per-step latent MSE:")
    print(mean_per_step_latent.tolist())

    print("\nMean per-step pixel MSE:")
    print(mean_per_step_pixel.tolist())

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, K+1), mean_per_step_latent, label="Latent MSE")
    plt.plot(range(1, K+1), mean_per_step_pixel, label="Pixel MSE")
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.title(f"Average Rollout Error Over Time ({len(results)} episodes)")
    plt.legend()
    plt.tight_layout()

    dataset_name = os.path.basename(DATASET_DIR.rstrip("/\\"))
    out_curve = os.path.join(OUT_DIR, f"avg_rollout_error_curve_{dataset_name}K32_multistep10_2.png")
    plt.savefig(out_curve, dpi=200)
    plt.close()
    print("\nSaved averaged error curve to:", out_curve)

    # -------------------------
    # Save text summary
    # -------------------------
    out_txt = os.path.join(OUT_DIR, f"avg_eval_summary_{dataset_name}K32_multistep10_2.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {DATASET_DIR}\n")
        f.write(f"AE: {AE_WEIGHTS}\n")
        f.write(f"Dynamics: {DYN_WEIGHTS}\n")
        f.write(f"K = {K}\n")
        f.write(f"Episodes evaluated: {len(results)}\n\n")
        f.write(f"Mean smoothness ratio      : {mean_smooth:.6f}\n")
        f.write(f"Mean 1-step latent MSE     : {mean_1_lat:.6f}\n")
        f.write(f"Mean 1-step pixel MSE      : {mean_1_pix:.6f}\n")
        f.write(f"Mean rollout latent MSE    : {mean_roll_lat:.6f}\n")
        f.write(f"Mean rollout pixel MSE     : {mean_roll_pix:.6f}\n\n")
        f.write("Mean per-step latent MSE:\n")
        f.write(str(mean_per_step_latent.tolist()) + "\n\n")
        f.write("Mean per-step pixel MSE:\n")
        f.write(str(mean_per_step_pixel.tolist()) + "\n")
    print("Saved summary to:", out_txt)

if __name__ == "__main__":
    main()