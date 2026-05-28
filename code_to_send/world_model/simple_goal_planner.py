import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn1.pth"
DYN_WEIGHTS = "dynamics_multistep_turn.pth"

LATENT_DIM = 128
NUM_ACTIONS = 4

DATASET_DIR = "dataset/dataset7"   # change to dataset2 / dataset3
K = 16
N_CANDIDATES = 128

OUT_DIR = "world_model_out/goal_planner"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def to_torch_img(frames):
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    return x


def load_random_episode():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    ep_file = random.choice(files)
    data = np.load(os.path.join(DATASET_DIR, ep_file))
    return ep_file, data["obs"], data["actions"]


def rollout(ae, dyn, z0, actions):
    z = z0
    z_list = [z0.squeeze(0)]

    with torch.no_grad():
        for a in actions:
            a_t = torch.tensor([int(a)], dtype=torch.long, device=DEVICE)
            z = dyn(z, a_t)
            z_list.append(z.squeeze(0))

        z_seq = torch.stack(z_list, dim=0)
        frames = ae.decoder(z_seq)

    return z_seq, frames


def img_tensor_to_numpy(x):
    x = x.detach().clamp(0, 1).cpu()
    return x.permute(0, 2, 3, 1).numpy()


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

    ep_file, obs, true_actions = load_random_episode()
    T = obs.shape[0]

    if T < K + 1:
        raise RuntimeError("Episode too short")

    t0 = random.randint(0, T - (K + 1))

    start_frame = obs[t0:t0 + 1]
    goal_frame = obs[t0 + K:t0 + K + 1]
    true_future = obs[t0:t0 + K + 1]
    true_action_seq = true_actions[t0:t0 + K]

    with torch.no_grad():
        start_img = to_torch_img(start_frame).to(DEVICE)
        goal_img = to_torch_img(goal_frame).to(DEVICE)
        true_imgs = to_torch_img(true_future).to(DEVICE)

        z_start = ae.encoder(start_img)
        z_goal = ae.encoder(goal_img)
        z_true = ae.encoder(true_imgs)

    candidates = []

    # include real recorded action sequence as reference
    candidates.append(("ground_truth", true_action_seq.copy()))

    # random candidate plans
    for i in range(N_CANDIDATES):
        actions = np.random.randint(0, NUM_ACTIONS, size=K, dtype=np.int64)
        candidates.append((f"cand_{i:03d}", actions))

    results = []

    for name, actions in candidates:
        z_seq, pred_frames = rollout(ae, dyn, z_start, actions)

        # planner score: final predicted latent close to goal latent
        final_latent = z_seq[-1].unsqueeze(0)
        goal_score = F.mse_loss(final_latent, z_goal).item()

        # diagnostic only: compare full rollout with true future
        true_mse = F.mse_loss(pred_frames, true_imgs).item()

        results.append({
            "name": name,
            "actions": actions,
            "goal_score": goal_score,
            "true_mse": true_mse,
            "frames": pred_frames,
        })

    results.sort(key=lambda r: r["goal_score"])

    print("Dataset:", DATASET_DIR)
    print("Episode:", ep_file)
    print("t0:", t0)
    print("Goal frame: t0 +", K)
    print("\nTop 5 candidates by goal latent score:")

    for r in results[:5]:
        print(
            f"{r['name']:12s} | goal_score={r['goal_score']:.6f} "
            f"| true_mse={r['true_mse']:.6f} | actions={r['actions'].tolist()}"
        )

    # visualize true future + top 5 imagined plans
    rows = []
    labels = []

    rows.append(img_tensor_to_numpy(true_imgs))
    labels.append("true future")

    for r in results[:5]:
        rows.append(img_tensor_to_numpy(r["frames"]))
        labels.append(r["name"])

    n_rows = len(rows)
    n_cols = K + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.35 * n_rows))

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].imshow(rows[r][c])
            axes[r, c].axis("off")

            if c == 0:
                axes[r, c].set_ylabel(labels[r], fontsize=8)

            if r == 0:
                axes[r, c].set_title(f"t{c}", fontsize=7)

    plt.tight_layout()

    dataset_name = os.path.basename(DATASET_DIR)
    out_path = os.path.join(
        OUT_DIR,
        f"goal_planner_{dataset_name}_{ep_file.replace('.npz', '')}.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()