import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import csv

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "ae4.pth"
DYN_WEIGHTS = "dynamics_multistep_k5.pth"   

LATENT_DIM = 128
NUM_ACTIONS = 4
N_EPISODES = 10
OUT_CSV = "world_model_out/April6th/original_results.csv"

DATASETS = [
    "dataset/dataset1",
    "dataset/dataset2",
    "dataset/dataset3",
]

K_VALUES = [16, 32]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_episode(npz_path):
    data = np.load(npz_path)
    obs = data["obs"]
    actions = data["actions"]
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
    gt_frames = obs[t0:t0+K+1]
    act_seq   = actions[t0:t0+K]

    with torch.no_grad():
        gt_imgs = to_torch_img(gt_frames).to(DEVICE)
        z_gt = ae.encoder(gt_imgs)

    # smoothness
    z_norm = z_gt.norm(dim=1)
    delta_gt = z_gt[1:] - z_gt[:-1]
    delta_gt_norm = delta_gt.norm(dim=1)
    smooth_ratio = (delta_gt_norm.mean() / z_norm.mean()).item()

    # 1-step
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

    # rollout
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
        "smooth_ratio": smooth_ratio,
        "one_step_latent_mse": one_step_latent_mse,
        "one_step_pixel_mse": one_step_pixel_mse,
        "rollout_latent_mse": rollout_latent_mse,
        "rollout_pixel_mse": rollout_pixel_mse,
        "per_step_latent_mse": per_step_latent_mse,
        "per_step_pixel_mse": per_step_pixel_mse,
    }


def evaluate_dataset(dataset_dir, ae, dyn, K, n_episodes):
    files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not files:
        raise RuntimeError(f"No episodes found in {dataset_dir}")

    chosen_files = random.sample(files, min(n_episodes, len(files)))
    results = []

    for f in chosen_files:
        ep_path = os.path.join(dataset_dir, f)
        out = evaluate_one_episode(ep_path, ae, dyn, K)
        if out is not None:
            results.append(out)

    if not results:
        raise RuntimeError(f"No valid episodes for {dataset_dir} with K={K}")

    return {
        "smoothness": float(np.mean([r["smooth_ratio"] for r in results])),
        "one_step_latent": float(np.mean([r["one_step_latent_mse"] for r in results])),
        "one_step_pixel": float(np.mean([r["one_step_pixel_mse"] for r in results])),
        "rollout_latent": float(np.mean([r["rollout_latent_mse"] for r in results])),
        "rollout_pixel": float(np.mean([r["rollout_pixel_mse"] for r in results])),
    }


def main():
    os.makedirs("world_model_out", exist_ok=True)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=256).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    rows = []
    print("Evaluating all datasets...")

    for dataset_dir in DATASETS:
        dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
        for K in K_VALUES:
            print(f"Running {dataset_name}, K={K}")
            metrics = evaluate_dataset(dataset_dir, ae, dyn, K, N_EPISODES)

            row = {
                "Dataset": dataset_name,
                "K": K,
                "Smoothness": metrics["smoothness"],
                "1-step Latent": metrics["one_step_latent"],
                "1-step Pixel": metrics["one_step_pixel"],
                "Rollout Latent": metrics["rollout_latent"],
                "Rollout Pixel": metrics["rollout_pixel"],
            }
            rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "K",
                "Smoothness",
                "1-step Latent",
                "1-step Pixel",
                "Rollout Latent",
                "Rollout Pixel",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {OUT_CSV}\n")

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()