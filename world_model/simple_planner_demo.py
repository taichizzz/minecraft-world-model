import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "ae4.pth"
DYN_WEIGHTS = "dynamics_multistep_k5.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4
K = 16

DATASETS = [
    "dataset/dataset1",
    "dataset/dataset2",
    "dataset/dataset3",
]

OUT_DIR = "world_model_out/planner_demo"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_episode(path):
    data = np.load(path)
    return data["obs"], data["actions"]


def to_torch_img(frames):
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def make_candidate_actions(gt_actions):
    """
    Creates simple candidate action sequences.
    Action meanings depend on your collector, but the model only needs integers.
    """
    candidates = {}

    candidates["ground_truth"] = gt_actions.copy()

    candidates["all_0"] = np.zeros(K, dtype=np.int64)
    candidates["all_1"] = np.ones(K, dtype=np.int64)
    candidates["all_2"] = np.full(K, 2, dtype=np.int64)
    candidates["all_3"] = np.full(K, 3, dtype=np.int64)

    candidates["alternate_0_1"] = np.array([0, 1] * (K // 2), dtype=np.int64)
    candidates["alternate_2_3"] = np.array([2, 3] * (K // 2), dtype=np.int64)

    candidates["random"] = np.random.randint(0, NUM_ACTIONS, size=K, dtype=np.int64)

    return candidates


def rollout_candidate(ae, dyn, z0, action_seq):
    z_list = [z0.squeeze(0)]

    with torch.no_grad():
        z = z0
        for a in action_seq:
            a_t = torch.tensor([int(a)], dtype=torch.long, device=DEVICE)
            z = dyn(z, a_t)
            z_list.append(z.squeeze(0))

        z_rollout = torch.stack(z_list, dim=0)
        recon = ae.decoder(z_rollout)

    return z_rollout, recon


def img_row_from_tensor(imgs):
    imgs = imgs.detach().clamp(0, 1).cpu()
    imgs = imgs.permute(0, 2, 3, 1).numpy()
    return imgs


def run_one_dataset(dataset_dir, ae, dyn):
    files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ])

    ep_file = random.choice(files)
    ep_path = os.path.join(dataset_dir, ep_file)

    obs, actions = load_episode(ep_path)
    T = obs.shape[0]

    if T < K + 1:
        print("Episode too short, skipping:", ep_path)
        return

    t0 = random.randint(0, T - (K + 1))

    gt_frames = obs[t0:t0 + K + 1]
    gt_actions = actions[t0:t0 + K]

    gt_imgs = to_torch_img(gt_frames).to(DEVICE)

    with torch.no_grad():
        z_gt = ae.encoder(gt_imgs)
        recon_gt = ae.decoder(z_gt)

    z0 = z_gt[0].unsqueeze(0)

    candidates = make_candidate_actions(gt_actions)

    rows = []
    row_labels = []

    # ground truth row
    rows.append(img_row_from_tensor(gt_imgs))
    row_labels.append("true frames")

    # AE reconstruction row
    rows.append(img_row_from_tensor(recon_gt))
    row_labels.append("AE recon")

    scores = {}

    for name, action_seq in candidates.items():
        z_rollout, recon = rollout_candidate(ae, dyn, z0, action_seq)

        # diagnostic score against the actual future
        # In a real planner, this would be replaced by a goal/collision score.
        pixel_mse = F.mse_loss(recon, gt_imgs).item()
        latent_mse = F.mse_loss(z_rollout, z_gt).item()

        scores[name] = {
            "pixel_mse": pixel_mse,
            "latent_mse": latent_mse,
            "actions": action_seq.tolist(),
        }

        rows.append(img_row_from_tensor(recon))
        row_labels.append(name)

    best_name = min(scores.keys(), key=lambda n: scores[n]["pixel_mse"])

    dataset_name = os.path.basename(dataset_dir)
    print(f"\nDataset: {dataset_name}")
    print("Episode:", ep_file)
    print("t0:", t0)
    print("Best candidate by diagnostic pixel MSE:", best_name)

    for name, s in scores.items():
        print(
            f"{name:15s} | pixel={s['pixel_mse']:.6f} "
            f"| latent={s['latent_mse']:.6f} | actions={s['actions']}"
        )

    # plot
    n_rows = len(rows)
    n_cols = K + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.35 * n_rows))

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].imshow(rows[r][c])
            axes[r, c].axis("off")

            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=8)

            if r == 0:
                axes[r, c].set_title(f"t{c}", fontsize=7)

    plt.tight_layout()

    out_path = os.path.join(
        OUT_DIR,
        f"planner_demo_{dataset_name}_{ep_file.replace('.npz', '')}.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


def main():
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsMLP(
        latent_dim=LATENT_DIM,
        num_actions=NUM_ACTIONS,
        hidden=256
    ).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    for dataset_dir in DATASETS:
        run_one_dataset(dataset_dir, ae, dyn)


if __name__ == "__main__":
    main()