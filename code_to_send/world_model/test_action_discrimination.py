"""
Test whether DynamicsTurningMLP actually discriminates between actions.

For several random starting frames, compute:
  z_0            = AE.encode(frame)
  z_id           = z_0            (no dynamics)
  z_0_fwd        = dyn(z_0, a=0)  (forward)
  z_0_turnL      = dyn(z_0, a=1)  (turn left)
  z_0_turnR      = dyn(z_0, a=2)  (turn right)

Then:
  1. Save a visual grid: row per frame, columns = [original, AE recon,
     pred_fwd, pred_turnL, pred_turnR]. If columns 3-5 look identical, the
     dynamics ignores the action input.
  2. Print latent-space diagnostics:
       action_step = ||z_0_a - z_0||  for each a
         - should be on the order of consec_step from measure_latent_smoothness
       between_action = ||z_0_a - z_0_b||  for a != b
         - should be non-trivial; if ~0 the dynamics collapses over actions
       ratio between / action = how much actions diverge from each other
         relative to how much each moves the latent
"""
import os
import random
import numpy as np
import torch
from PIL import Image

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AE_WEIGHTS = "aeturn2.pth"
DYN_WEIGHTS = "dynamics_turn_1.pth"

LATENT_DIM = 128
NUM_ACTIONS = 3
ACTION_NAMES = ["fwd", "turnL", "turnR"]

DATASET_DIR = "dataset/dataset2_turn"
N_FRAMES = 6
OUT_DIR = "world_model_out/April22nd"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


def to_uint8(chw_tensor):
    x = chw_tensor.detach().clamp(0, 1).cpu()
    return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def make_row(imgs_uint8, pad=4):
    h, w, _ = imgs_uint8[0].shape
    grid_w = len(imgs_uint8) * w + (len(imgs_uint8) - 1) * pad
    canvas = np.zeros((h, grid_w, 3), dtype=np.uint8)
    x = 0
    for im in imgs_uint8:
        canvas[:, x:x + w, :] = im
        x += w + pad
    return canvas


def stack_rows(rows, pad=6):
    h = rows[0].shape[0]
    w = rows[0].shape[1]
    canvas = np.zeros((len(rows) * h + (len(rows) - 1) * pad, w, 3), dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + h, :, :] = r
        y += h + pad
    return canvas


def sample_frames(dataset_dir, n, rng):
    episode_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not episode_files:
        raise RuntimeError(f"No episodes in {dataset_dir}")

    frames = []
    chosen_meta = []
    tries = 0
    while len(frames) < n and tries < n * 10:
        ep = rng.choice(episode_files)
        data = np.load(os.path.join(dataset_dir, ep))
        obs = data["obs"]
        if obs.shape[0] < 1:
            tries += 1
            continue
        t = rng.randint(0, obs.shape[0] - 1)
        frames.append(obs[t])
        chosen_meta.append((ep, t))
        tries += 1

    return np.stack(frames, axis=0), chosen_meta


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=512
    ).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    frames_np, meta = sample_frames(DATASET_DIR, N_FRAMES, rng)
    print("Sampled frames:")
    for ep, t in meta:
        print(f"  {ep} @ t={t}")

    frames = torch.from_numpy(frames_np).float() / 255.0
    frames = frames.permute(0, 3, 1, 2).to(DEVICE)  # (N, 3, 64, 64)

    with torch.no_grad():
        z_0 = ae.encoder(frames)                    # (N, 128)
        recon_0 = ae.decoder(z_0)                   # (N, 3, 64, 64)

        preds = []
        preds_decoded = []
        for a in range(NUM_ACTIONS):
            a_t = torch.full((frames.shape[0],), a, dtype=torch.long, device=DEVICE)
            z_next = dyn(z_0, a_t)
            preds.append(z_next)
            preds_decoded.append(ae.decoder(z_next))

    # ---- latent diagnostics ----
    print("\n=== latent diagnostics ===")
    z_0_norm = z_0.norm(dim=1).mean().item()
    print(f"mean ||z_0||                      = {z_0_norm:.4f}")

    action_steps = {}
    for a in range(NUM_ACTIONS):
        step = (preds[a] - z_0).norm(dim=1).mean().item()
        action_steps[a] = step
        print(f"mean ||dyn(z_0, a={a}) - z_0||  ({ACTION_NAMES[a]:5s}) = {step:.4f}")

    print()
    between = {}
    for a in range(NUM_ACTIONS):
        for b in range(a + 1, NUM_ACTIONS):
            d = (preds[a] - preds[b]).norm(dim=1).mean().item()
            between[(a, b)] = d
            print(f"mean ||dyn(_, a={a}) - dyn(_, a={b})||  "
                  f"({ACTION_NAMES[a]:5s} vs {ACTION_NAMES[b]:5s}) = {d:.4f}")

    avg_action_step = sum(action_steps.values()) / len(action_steps)
    avg_between = sum(between.values()) / len(between)
    print(f"\navg action step                 = {avg_action_step:.4f}")
    print(f"avg between-action separation   = {avg_between:.4f}")
    if avg_action_step > 1e-8:
        print(f"between / action step ratio     = {avg_between / avg_action_step:.3f}")
        print("  ~0  -> all actions predict the same next latent (action collapse)")
        print("  ~1  -> actions are as different from each other as each is from z_0 (healthy)")
    else:
        print("  dynamics barely moves the latent at all (identity collapse)")

    # reference consec step was ~0.57 on aeturn1; re-measure with aeturn2 via
    # measure_latent_smoothness.py if you want a direct aeturn2-based baseline.
    print("\nReference: mean real consec step on aeturn1 was ~0.57 "
          "(re-run measure_latent_smoothness.py with aeturn2 for an exact match).")
    print("If action_step is << that value, dynamics predicts much less motion than real.")

    # ---- visualization ----
    rows = []
    for i in range(frames.shape[0]):
        cells = [
            to_uint8(frames[i]),
            to_uint8(recon_0[i]),
            to_uint8(preds_decoded[0][i]),
            to_uint8(preds_decoded[1][i]),
            to_uint8(preds_decoded[2][i]),
        ]
        rows.append(make_row(cells, pad=4))

    grid = stack_rows(rows, pad=6)

    out_path = os.path.join(OUT_DIR, "action_discrimination.png")
    Image.fromarray(grid).save(out_path)
    print(f"\nSaved: {out_path}")
    print("Column order: [original, AE recon, pred fwd, pred turnL, pred turnR]")
    print("If cols 3/4/5 look indistinguishable, the dynamics ignores the action.")


if __name__ == "__main__":
    main()
