import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= CONFIG =============
AE_WEIGHTS = "aeturn2.pth"
DYN_WEIGHTS = "dynamics_turn_1.pth"

LATENT_DIM = 128
NUM_ACTIONS = 3 

DATASET_DIR = "dataset/dataset2_turn"

# how far ahead the planner imagines per replan
PLAN_HORIZON = 4

# how many random plans to sample at each step
NUM_SAMPLES = 512

# how many steps the controller runs in total
CONTROL_STEPS = 16

# distance between start and goal frame in the recorded episode
GOAL_OFFSET = 16

# stop early if we get within this latent distance of the goal
DIST_THRESHOLD = 0.0005

# action sampling weights (matches data collection: 4=move, 3=turnL, 3=turnR)
ACTION_WEIGHTS = [4.0, 3.0, 3.0]

# weight on pixel-space goal matching (score = latent_mse + PIXEL_COST_WEIGHT * pixel_mse)
# 0.0 disables pixel term and falls back to pure-latent scoring.
PIXEL_COST_WEIGHT = 1.0

OUT_DIR = "world_model_out/April20th"
os.makedirs(OUT_DIR, exist_ok=True)

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

import time
SEED = int(time.time()) % 100000  # different every second
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using seed: {SEED}")


def to_torch_img(frames):
    x = torch.from_numpy(frames).float() / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def img_tensor_to_numpy(x):
    x = x.detach().clamp(0, 1).cpu()
    return x.permute(0, 2, 3, 1).numpy()

def load_random_episode():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    ep_file = random.choice(files)
    data = np.load(os.path.join(DATASET_DIR, ep_file))
    return ep_file, data["obs"], data["actions"]

def load_random_episode_with_movement():
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])

    for _ in range(50):
        ep_file = random.choice(files)
        data = np.load(os.path.join(DATASET_DIR, ep_file))
        obs = data["obs"]
        actions = data["actions"]
        T = obs.shape[0]

        if T < GOAL_OFFSET + 1:
            continue

        t0 = random.randint(0, T - GOAL_OFFSET - 1)

        start = obs[t0].astype(float)
        goal = obs[t0 + GOAL_OFFSET].astype(float)
        diff = np.abs(start - goal).mean()

        if diff > 20:
            return ep_file, obs, actions, t0

    raise RuntimeError("Could not find episode with sufficient movement")


def plan_one_action(z_current, z_goal, goal_img, dyn, ae,
                    num_actions, horizon, num_samples):
    """
    Sample many action sequences, roll them out, score by FINAL-step cost:
        latent_mse(z_final, z_goal) + PIXEL_COST_WEIGHT * pixel_mse(decode(z_final), goal_img)
    """
    weights = torch.tensor(ACTION_WEIGHTS, device=DEVICE)
    flat_actions = torch.multinomial(
        weights, num_samples * horizon, replacement=True
    )
    actions = flat_actions.view(num_samples, horizon).long()

    z = z_current.unsqueeze(0).expand(num_samples, -1).contiguous()
    with torch.no_grad():
        for t in range(horizon):
            a_t = actions[:, t]
            z = dyn(z, a_t)

        latent_scores = ((z - z_goal.unsqueeze(0)) ** 2).mean(dim=1)

        if PIXEL_COST_WEIGHT > 0.0:
            decoded = ae.decoder(z)  # (num_samples, 3, 64, 64)
            pixel_scores = ((decoded - goal_img) ** 2).mean(dim=(1, 2, 3))
            scores = latent_scores + PIXEL_COST_WEIGHT * pixel_scores
        else:
            scores = latent_scores
            pixel_scores = torch.zeros_like(latent_scores)

    best_idx = scores.argmin().item()
    best_first_action = actions[best_idx, 0].item()
    best_score = scores[best_idx].item()
    best_latent = latent_scores[best_idx].item()
    best_pixel = pixel_scores[best_idx].item()

    return best_first_action, best_score, best_latent, best_pixel


def main():
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM,
        num_actions=NUM_ACTIONS,
        hidden=512
    ).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    ep_file, obs, true_actions, t0 = load_random_episode_with_movement()

    # t0 = random.randint(0, obs.shape[0] - GOAL_OFFSET - 1)

    start_frame = obs[t0:t0 + 1]
    goal_frame = obs[t0 + GOAL_OFFSET:t0 + GOAL_OFFSET + 1]

    print(f"Episode: {ep_file}")
    print(f"Start at t0={t0}, goal at t0+{GOAL_OFFSET}={t0+GOAL_OFFSET}")
    print(f"True actions between: {true_actions[t0:t0+GOAL_OFFSET].tolist()}")

    with torch.no_grad():
        start_img = to_torch_img(start_frame).to(DEVICE)
        goal_img = to_torch_img(goal_frame).to(DEVICE)

        z_current = ae.encoder(start_img).squeeze(0)
        z_goal = ae.encoder(goal_img).squeeze(0)

    chosen_actions = []
    z_history = [z_current]
    dist_init = F.mse_loss(z_current, z_goal).item()
    distances_to_goal = [dist_init]

    print(f"\nInitial distance to goal: {dist_init:.6f}")
    print(f"Distance threshold: {DIST_THRESHOLD}\n")

    reached_goal = False

    for step in range(CONTROL_STEPS):
        # FIX 3: check threshold BEFORE planning (not after)
        current_dist = F.mse_loss(z_current, z_goal).item()
        if current_dist < DIST_THRESHOLD:
            print(f"Reached goal at step {step}, dist={current_dist:.6f}")
            reached_goal = True
            break

        action, score, lat_score, pix_score = plan_one_action(
            z_current, z_goal, goal_img, dyn, ae,
            num_actions=NUM_ACTIONS,
            horizon=PLAN_HORIZON,
            num_samples=NUM_SAMPLES
        )

        with torch.no_grad():
            a_tensor = torch.tensor([action], dtype=torch.long, device=DEVICE)
            z_next = dyn(z_current.unsqueeze(0), a_tensor).squeeze(0)

        chosen_actions.append(action)
        z_history.append(z_next)

        dist = F.mse_loss(z_next, z_goal).item()
        distances_to_goal.append(dist)

        z_current = z_next

        action_name = ["move", "turnL", "turnR"][action]
        print(
            f"step {step:02d} | action={action} ({action_name}) "
            f"| dist_to_goal={dist:.6f} | plan_score={score:.6f} "
            f"(lat={lat_score:.6f}, pix={pix_score:.6f})"
        )

    if not reached_goal:
        print(f"\nFinished {CONTROL_STEPS} steps without reaching goal")
        print(f"Final distance: {distances_to_goal[-1]:.6f}")
        print(f"Best distance achieved: {min(distances_to_goal):.6f} "
              f"at step {distances_to_goal.index(min(distances_to_goal))}")

    # decode trajectory
    z_seq = torch.stack(z_history, dim=0)
    with torch.no_grad():
        decoded = ae.decoder(z_seq)

    decoded_np = img_tensor_to_numpy(decoded)
    goal_real_np = goal_frame[0] / 255.0
    start_real_np = start_frame[0] / 255.0

    # plot trajectory comparison
    n_steps = len(z_history)
    fig, axes = plt.subplots(3, max(n_steps, 4), figsize=(1.5 * max(n_steps, 4), 4.8))

    for i in range(n_steps):
        # row 0: planner imagined trajectory
        axes[0, i].imshow(decoded_np[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t{i}", fontsize=7)

        # row 1: blank for now (will fill with reference info)
        axes[1, i].axis("off")

        # row 2: blank
        axes[2, i].axis("off")

    # row 1: real start frame on left, real goal frame on right
    axes[1, 0].imshow(start_real_np)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("real start", fontsize=7)

    axes[1, n_steps - 1].imshow(goal_real_np)
    axes[1, n_steps - 1].axis("off")
    axes[1, n_steps - 1].set_title("real goal", fontsize=7)

    # row 2: actual recorded frames from the episode (for comparison)
    real_traj = obs[t0:t0 + n_steps] / 255.0
    for i in range(min(n_steps, real_traj.shape[0])):
        axes[2, i].imshow(real_traj[i])
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("real frames →", fontsize=7)

    fig.suptitle(
        f"Row 1: planner imagined | Row 2: reference | Row 3: real episode\n"
        f"actions: {chosen_actions} | final dist: {distances_to_goal[-1]:.6f}",
        fontsize=9
    )
    plt.tight_layout()

    out_path = os.path.join(
        OUT_DIR,
        f"closed_loop_v2_{ep_file.replace('.npz', '')}_t{t0}.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved trajectory: {out_path}")

    # plot distance curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(distances_to_goal)), distances_to_goal, marker="o")
    plt.axhline(y=DIST_THRESHOLD, color='r', linestyle='--', label='threshold')
    plt.xlabel("Control step")
    plt.ylabel("Latent MSE distance to goal")
    plt.title("Distance to goal over closed-loop steps")
    plt.legend()
    plt.tight_layout()

    out_dist = os.path.join(
        OUT_DIR,
        f"closed_loop_v2_distance_{ep_file.replace('.npz', '')}_t{t0}.png"
    )
    plt.savefig(out_dist, dpi=200)
    plt.close()
    print(f"Saved distance curve: {out_dist}")


if __name__ == "__main__":
    main()