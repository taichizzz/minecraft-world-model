"""
Planner that uses a value head V(z) to find and approach the diamond block,
instead of matching a specific goal frame.

For each control step:
  - sample NUM_SAMPLES random action sequences of length PLAN_HORIZON
  - roll each one out in latent space with the dynamics model
  - score by V(z_final): higher = predicted closer to the diamond
  - pick the first action of the best sequence, step forward, repeat

No goal frame is needed. Starts from any frame in the dataset.
"""
import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP
from train_value_head import ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= CONFIG =============
AE_WEIGHTS = "aeturn3.pth"
DYN_WEIGHTS = "dynamics_turn_3.pth"
VALUE_WEIGHTS = "value_head_2.pth"

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 256

DATASET_DIR = "dataset/dataset2_turn"

PLAN_HORIZON = 16
NUM_SAMPLES = 512
CONTROL_STEPS = 32

ACTION_WEIGHTS = [4.0, 3.0, 3.0]

OUT_DIR = "world_model_out/April26th"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = int(time.time()) % 100000
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


def load_random_start():
    """Pick a random episode and a random starting frame, biased away from
    the very last frames (which are already near the diamond)."""
    files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    ep_file = random.choice(files)
    data = np.load(os.path.join(DATASET_DIR, ep_file))
    obs = data["obs"]
    T = obs.shape[0]
    # avoid starting in the last 10 frames to give the planner real work to do
    t0 = random.randint(0, max(1, T - 10))
    return ep_file, obs, t0


def plan_one_action(z_current, dyn, value, horizon, num_samples):
    """
    Sample action sequences, roll them out, pick the one whose final
    latent has the highest V (= predicted P(near diamond)).
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
        v_final = value(z)  # (num_samples,)

    best_idx = v_final.argmax().item()
    return actions[best_idx, 0].item(), v_final[best_idx].item()


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

    value = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE))
    value.eval()

    ep_file, obs, t0 = load_random_start()
    start_frame = obs[t0:t0 + 1]
    print(f"Episode: {ep_file}, start at t0={t0} (T={obs.shape[0]})")

    with torch.no_grad():
        start_img = to_torch_img(start_frame).to(DEVICE)
        z_current = ae.encoder(start_img).squeeze(0)
        v_init = value(z_current.unsqueeze(0)).item()

    chosen_actions = []
    z_history = [z_current]
    v_history = [v_init]
    print(f"Initial V(z) = {v_init:.4f}\n")

    for step in range(CONTROL_STEPS):
        action, v_pred = plan_one_action(
            z_current, dyn, value,
            horizon=PLAN_HORIZON,
            num_samples=NUM_SAMPLES,
        )

        with torch.no_grad():
            a_tensor = torch.tensor([action], dtype=torch.long, device=DEVICE)
            z_next = dyn(z_current.unsqueeze(0), a_tensor).squeeze(0)
            v_now = value(z_next.unsqueeze(0)).item()

        chosen_actions.append(action)
        z_history.append(z_next)
        v_history.append(v_now)
        z_current = z_next

        action_name = ["move", "turnL", "turnR"][action]
        print(
            f"step {step:02d} | action={action} ({action_name}) | "
            f"V_now={v_now:.4f} | V_pred@horizon={v_pred:.4f}"
        )

    # decode imagined trajectory
    z_seq = torch.stack(z_history, dim=0)
    with torch.no_grad():
        decoded = ae.decoder(z_seq)
    decoded_np = img_tensor_to_numpy(decoded)
    start_real_np = start_frame[0] / 255.0

    n_steps = len(z_history)
    fig, axes = plt.subplots(2, max(n_steps, 4), figsize=(1.2 * max(n_steps, 4), 3.5))
    for i in range(n_steps):
        axes[0, i].imshow(decoded_np[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t{i}\nV={v_history[i]:.2f}", fontsize=6)
        axes[1, i].axis("off")
    axes[1, 0].imshow(start_real_np)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("real start", fontsize=7)

    fig.suptitle(
        f"Planner maximizing V(z) | actions: {chosen_actions}",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = os.path.join(
        OUT_DIR, f"value_planner_{ep_file.replace('.npz', '')}_t{t0}.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved trajectory: {out_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(v_history)), v_history, marker="o")
    plt.xlabel("Control step")
    plt.ylabel("V(z)  —  P(near diamond)")
    plt.title("Value over closed-loop steps (higher = closer to diamond)")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_v = os.path.join(
        OUT_DIR, f"value_planner_v_{ep_file.replace('.npz', '')}_t{t0}.png"
    )
    plt.savefig(out_v, dpi=200)
    plt.close()
    print(f"Saved V curve: {out_v}")


if __name__ == "__main__":
    main()
