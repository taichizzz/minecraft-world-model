"""
Step 1 of the hybrid (map + MPC) planner: spatial map builder.

Runs a closed-loop episode in Malmo using a SIMPLE reactive controller
(turn until V > threshold, then move; with a 'stuck' fallback) and
incrementally builds a 2D occupancy grid. The point is to validate three
things before we wire MPC back in:

    1. Position tracking from Malmo XPos/ZPos works as expected.
    2. V > threshold + ray-cast actually marks the diamond cell correctly.
    3. Bump detection (move sent but position unchanged) marks obstacles.

After the episode it plots the agent's belief map next to the ground-truth
env3 layout so we can eyeball whether the map matches reality.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- bootstrap so we can import siblings + MalmoPython + xsd schemas ---
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
for _d in [
    os.path.join(ROOT, "build", "Malmo", "src", "PythonWrapper", "Release"),
    os.path.join(ROOT, "build", "install", "Python_Examples"),
]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
if not os.environ.get("MALMO_XSD_PATH"):
    _schemas = os.path.join(ROOT, "Schemas")
    if os.path.isfile(os.path.join(_schemas, "Mission.xsd")):
        os.environ["MALMO_XSD_PATH"] = _schemas

import MalmoPython
from model import AutoEncoder
from train_value_head import ValueHead
from detect_diamond import diamond_pixel_mask, MIN_PIXELS as CYAN_MIN_PIXELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MISSION_FILE = "missions/env3.xml"
AE_WEIGHTS = "aeturn3.pth"
VALUE_WEIGHTS = "value_head_2.pth"

LATENT_DIM = 128
HIDDEN = 256

# action mapping must match the data-collection script
ACTIONS_STR = ["move 1", "turn 0.2", "turn -0.2"]
ACTIONS_NAME = ["move", "turnR", "turnL"]

V_THRESHOLD = 0.5      # mark diamond + walk forward when V > this
TURN_DIR = 1           # which direction to turn while searching (1=R, 2=L)
STUCK_TURN_AFTER = 3   # if we haven't moved for N steps, force turn
MAX_STEPS = 200
IMG_H, IMG_W = 64, 64

# debugging: dump every Nth frame so we can eyeball what the agent actually saw
DUMP_EVERY = 5

ROOM_HALF = 5
GRID_LO, GRID_HI = -ROOM_HALF, ROOM_HALF
GRID_SIZE = 2 * ROOM_HALF + 1   # 11x11

UNKNOWN, FREE, OBSTACLE, DIAMOND = 0, 1, 2, 3

OUT_DIR = "world_model_out/closed_loop"
os.makedirs(OUT_DIR, exist_ok=True)


# ============= HELPERS =============
def to_torch_img(frame):
    return (torch.from_numpy(frame).float().div_(255.0)
            .permute(2, 0, 1).unsqueeze(0))


def world_to_grid(x, z):
    """Continuous world (x, z) -> integer grid indices, clipped to bounds."""
    gx = int(np.clip(np.floor(x) - GRID_LO, 0, GRID_SIZE - 1))
    gz = int(np.clip(np.floor(z) - GRID_LO, 0, GRID_SIZE - 1))
    return gx, gz


def yaw_to_forward(yaw_deg):
    """Minecraft yaw: 0=+Z(south), 90=-X(west), 180=-Z(north), 270=+X(east).
    Returns forward direction (dx, dz) for one block forward."""
    rad = np.radians(yaw_deg)
    return -np.sin(rad), np.cos(rad)


def mark_current_free(grid, x, z):
    gx, gz = world_to_grid(x, z)
    if grid[gz, gx] != DIAMOND:   
        grid[gz, gx] = FREE


def maybe_mark_diamond(grid, x, z, yaw, diamond_visible):
    """If the diamond is visible right now (cyan pixels detected), ray-cast
    forward and mark the FARTHEST in-bounds unknown cell along the ray as
    the diamond cell. The diamond is at the far end of view, not the cell
    directly in front of us — the agent will refine as it gets closer."""
    if not diamond_visible:
        return
    dx, dz = yaw_to_forward(yaw)
    farthest = None
    for k in range(1, ROOM_HALF * 2 + 2):
        tx = x + dx * k
        tz = z + dz * k
        if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
            break
        tgx, tgz = world_to_grid(tx, tz)
        cell = grid[tgz, tgx]
        if cell == OBSTACLE:
            break                  # ray hits a wall; can't see beyond
        if cell == UNKNOWN:
            farthest = (tgx, tgz)
    if farthest is not None:
        tgx, tgz = farthest
        grid[tgz, tgx] = DIAMOND


def maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action,
                        cur_x, cur_z):
    """If the previous step was 'move' but position didn't change, the cell
    one block forward from prev pose is an obstacle."""
    if prev_action != 0 or prev_x is None:
        return
    if abs(cur_x - prev_x) > 0.05 or abs(cur_z - prev_z) > 0.05:
        return                     # we did move; not blocked
    dx, dz = yaw_to_forward(prev_yaw)
    tx, tz = prev_x + dx, prev_z + dz
    if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
        return
    tgx, tgz = world_to_grid(tx, tz)
    if grid[tgz, tgx] not in (DIAMOND, FREE):
        grid[tgz, tgx] = OBSTACLE

def reactive_action(diamond_visible, stuck_count, prev_action, rng):
    """Walk forward when diamond visible; otherwise search by turning.

    PRIORITY ORDER MATTERS:
      1. If stuck → escape (alternate turn/move) — even if the diamond is
         in view we have to find a free direction first, otherwise we'd
         bash into the same wall forever.
      2. If diamond visible (and not stuck) → walk toward it.
      3. Otherwise → search by turning.
    """
    if stuck_count >= STUCK_TURN_AFTER:
        # alternate: failed move -> turn to new direction;
        #            turn -> try move in that new direction.
        if prev_action == 0:
            return TURN_DIR
        else:
            return 0
    if diamond_visible:
        return 0
    return TURN_DIR if rng.random() < 0.7 else 0

def main():
    rng = np.random.default_rng()

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    value = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE))
    value.eval()
    print(f"Loaded AE={AE_WEIGHTS}  V={VALUE_WEIGHTS}")

    agent_host = MalmoPython.AgentHost()
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        mission_xml = f.read()
    mission = MalmoPython.MissionSpec(mission_xml, True)
    record = MalmoPython.MissionRecordSpec()

    ws = agent_host.getWorldState()
    while ws.is_mission_running:
        time.sleep(0.1)
        ws = agent_host.getWorldState()

    started = False
    for _ in range(5):
        try:
            agent_host.startMission(mission, record)
            started = True
            break
        except MalmoPython.MissionException as e:
            print("startMission retry:", e)
            time.sleep(2)
    if not started:
        raise RuntimeError("could not start mission")

    print("Waiting for mission to begin...")
    ws = agent_host.getWorldState()
    while not ws.has_mission_begun:
        time.sleep(0.1)
        ws = agent_host.getWorldState()

    while True:
        ws = agent_host.getWorldState()
        if not ws.is_mission_running:
            break
        if (len(ws.observations) > 0 and ws.observations[-1].text != "{}"
                and len(ws.video_frames) > 0):
            break
        time.sleep(0.05)
    print("Mission running. Building map.\n")

    grid = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.uint8)
    pos_list, yaw_list, v_list, action_list = [], [], [], []
    saved_frames = []   # for debugging — sampled subset
    prev_x = prev_z = prev_yaw = None
    prev_action = None
    stuck_count = 0

    for step in range(MAX_STEPS):
        ws = agent_host.getWorldState()
        if not ws.is_mission_running:
            print(f"Mission ended at step {step}")
            break
        if (len(ws.observations) == 0 or ws.observations[-1].text == "{}"
                or len(ws.video_frames) == 0):
            time.sleep(0.05)
            continue

        obs_json = json.loads(ws.observations[-1].text)
        if "XPos" not in obs_json or "ZPos" not in obs_json:
            continue
        x = obs_json["XPos"]
        z_pos = obs_json["ZPos"]
        yaw = obs_json.get("Yaw", 0.0) % 360.0

        frame = (np.frombuffer(ws.video_frames[-1].pixels, dtype=np.uint8)
                 .reshape(IMG_H, IMG_W, 3).copy())

        with torch.no_grad():
            img = to_torch_img(frame).to(DEVICE)
            z_t = ae.encoder(img).squeeze(0)
            v_now = value(z_t.unsqueeze(0)).item()

        # cyan-pixel detector is deterministic and is what V was trained
        # on; we use it as the primary "diamond visible" signal and keep
        # V around just for logging.
        cyan_count = int(diamond_pixel_mask(frame).sum())
        diamond_visible = cyan_count >= CYAN_MIN_PIXELS

        # --- update map from this observation ---
        maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action, x, z_pos)
        mark_current_free(grid, x, z_pos)
        maybe_mark_diamond(grid, x, z_pos, yaw, diamond_visible)

        # stuck detection: only count consecutive FAILED moves
        # (turning naturally doesn't change position; that isn't "stuck")
        if prev_action == 0:
            if (prev_x is not None
                    and abs(x - prev_x) < 0.05
                    and abs(z_pos - prev_z) < 0.05):
                stuck_count += 1
            else:
                stuck_count = 0

        # --- choose next action ---
        action_idx = reactive_action(diamond_visible, stuck_count, prev_action, rng)
        agent_host.sendCommand(ACTIONS_STR[action_idx])

        pos_list.append([x, z_pos])
        yaw_list.append(yaw)
        v_list.append(v_now)
        action_list.append(action_idx)
        if step % DUMP_EVERY == 0:
            saved_frames.append((step, x, z_pos, yaw, cyan_count, frame.copy()))

        marker = ""
        if diamond_visible:
            marker += "  DIAMOND"
        if stuck_count >= STUCK_TURN_AFTER:
            marker += "  STUCK"
        print(f"step {step:03d} | a={action_idx} ({ACTIONS_NAME[action_idx]:<5}) "
              f"| V={v_now:.3f} cyan={cyan_count:4d} "
              f"| pos=({x:5.2f},{z_pos:5.2f}) yaw={yaw:5.1f}{marker}")

        prev_x, prev_z, prev_yaw, prev_action = x, z_pos, yaw, action_idx
        time.sleep(0.25)

    print(f"\nfinal pos=({pos_list[-1][0]:.2f},{pos_list[-1][1]:.2f})  "
          f"V_max={max(v_list) if v_list else 0:.3f}  steps={len(pos_list)}")

    save_run(grid, pos_list, yaw_list, v_list, action_list)
    save_frame_grid(saved_frames)


def save_frame_grid(saved_frames):
    """Plot a grid of sampled frames so we can see what the agent saw, with
    pos/yaw/cyan-count annotations. Useful for debugging why cyan never fired."""
    if not saved_frames:
        return
    n = len(saved_frames)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (step, x, z, yaw, cyan, frame) in enumerate(saved_frames):
        axes[i].imshow(frame)
        axes[i].axis("off")
        axes[i].set_title(
            f"t={step} ({x:.1f},{z:.1f}) yaw={int(yaw)}\ncyan={cyan}",
            fontsize=7,
        )
    for j in range(n, len(axes)):
        axes[j].axis("off")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"map_step1_frames_{ts}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved frame grid: {out}")


# ============= VISUALIZATION =============
def save_run(grid, positions, yaws, values, actions):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(OUT_DIR, f"map_step1_{ts}.npz")
    np.savez(
        out_npz,
        grid=grid,
        positions=np.array(positions, dtype=np.float32),
        yaw=np.array(yaws, dtype=np.float32),
        v=np.array(values, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
    )
    print(f"Saved npz: {out_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # left: agent's belief map
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#cccccc", "#ffffff", "#222222", "#5be0d2"]
    )
    extent = [GRID_LO - 0.5, GRID_HI + 0.5, GRID_HI + 0.5, GRID_LO - 0.5]
    axes[0].imshow(grid, cmap=cmap, vmin=0, vmax=3, extent=extent, origin="upper")
    axes[0].set_title("Agent's belief map\n"
                      "gray=unknown, white=free, black=obstacle, cyan=diamond_seen",
                      fontsize=10)

    # right: ground truth from env3.xml
    OBSTACLES = [(-2, -1), (-1, 2), (2, 1),
                 (-3, 0), (0, 3),
                 (1, -2), (3, -1),
                 (-2, 3), (3, 2), (-1, -3)]
    GOALS = [(4, 4, "DIAMOND", "#5be0d2"),
             (-4, 4, "gold", "#fff570"),
             (4, -4, "emerald", "#50d250")]
    axes[1].add_patch(Rectangle((-ROOM_HALF - 0.5, -ROOM_HALF - 0.5),
                                2 * ROOM_HALF + 1, 2 * ROOM_HALF + 1,
                                facecolor="white", edgecolor="black", linewidth=3))
    for (x, z) in OBSTACLES:
        axes[1].add_patch(Rectangle((x - 0.5, z - 0.5), 1, 1,
                                    facecolor="#222", edgecolor="black"))
    for (x, z, name, color) in GOALS:
        axes[1].add_patch(Rectangle((x - 0.5, z - 0.5), 1, 1,
                                    facecolor=color, edgecolor="black", linewidth=2))
        axes[1].text(x, z, name, ha="center", va="center",
                     fontsize=8, weight="bold")
    axes[1].set_title("Ground truth (env3.xml)", fontsize=10)

    # overlay path on both panels
    pos = np.array(positions)
    for ax in axes:
        if len(pos) > 0:
            ax.plot(pos[:, 0], pos[:, 1], "r-", linewidth=1.0, alpha=0.6)
            ax.plot(pos[0, 0], pos[0, 1], "yo",
                    markersize=12, markeredgecolor="black", label="start")
            ax.plot(pos[-1, 0], pos[-1, 1], "rs",
                    markersize=12, markeredgecolor="black", label="end")
        ax.set_xlim(-ROOM_HALF - 1, ROOM_HALF + 1)
        ax.set_ylim(-ROOM_HALF - 1, ROOM_HALF + 1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"map_step1_{ts}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved map: {out_png}")


if __name__ == "__main__":
    main()
