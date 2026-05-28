"""
Step 1: reactive controller + occupancy-grid map builder.

Reactive policy (no MPC):
  - if V(current frame) > V_THRESHOLD: move forward
  - else: turn (consistent direction) until something appears
  - if blocked from moving N times in a row: force a turn
  - if turning M times without seeing anything: force a move (escape spinning)

Map (built live as the agent walks around):
  - 11x11 grid covering env3 floor (cells at integer block coords -5..+5)
  - state per cell: 0 unknown / 1 free / 2 obstacle
  - mark current cell free at every step
  - mark cell ahead obstacle if a `move` was attempted but position didn't change
  - record every position+yaw where V or cyan detector fired ("diamond seen here")

After the run we save:
  - the npz log (positions, yaws, V, cyan counts, cell_state, diamond_touched)
  - a side-by-side plot: agent's belief vs ground-truth env3 layout
This is purely diagnostic — no map-based planning yet. The point is to verify
the map matches reality before we layer A* + MPC on top in step 2.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
from detect_diamond import diamond_pixel_mask, MIN_PIXELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= CONFIG =============
MISSION_FILE = "missions/env3.xml"
AE_WEIGHTS = "aeturn3.pth"
VALUE_WEIGHTS = "value_head_2.pth"

LATENT_DIM = 128
HIDDEN = 256
IMG_H = IMG_W = 64
MAX_STEPS = 200

V_THRESHOLD = 0.30          # V above this -> move forward
STUCK_MOVE_LIMIT = 3        # if move fails this many times -> turn
TURN_STREAK_LIMIT = 8       # if turning this many times -> force move

# action mapping MUST match data collection
ACTIONS_STR = ["move 1", "turn 0.2", "turn -0.2"]
ACTIONS_NAME = ["move", "turnR", "turnL"]
A_MOVE, A_TURN_R, A_TURN_L = 0, 1, 2

# env3 layout (top-down)
GRID_MIN, GRID_MAX = -5, 5
GRID_SIZE = GRID_MAX - GRID_MIN + 1   # 11

# ground truth — only used for the side-by-side comparison plot
OBSTACLES_GT = [
    (-2, -1), (-1, 2), (2, 1),       # obsidian
    (-3, 0), (0, 3),                 # nether_brick
    (1, -2), (3, -1),                # quartz
    (-2, 3), (3, 2), (-1, -3),       # end_stone
]
GOALS_GT = [
    (4, 4, "DIAMOND",  "#5be0d2"),
    (-4, 4, "gold",    "#fff570"),
    (4, -4, "emerald", "#50d250"),
]

OUT_DIR = "world_model_out/closed_loop"
os.makedirs(OUT_DIR, exist_ok=True)


def to_torch_img(frame):
    return (torch.from_numpy(frame).float().div_(255.0)
            .permute(2, 0, 1).unsqueeze(0))


def world_to_cell(x, z):
    return int(np.floor(x)), int(np.floor(z))


def cell_in_grid(cx, cz):
    return GRID_MIN <= cx <= GRID_MAX and GRID_MIN <= cz <= GRID_MAX


def yaw_to_forward(yaw_deg):
    """Minecraft: yaw=0 -> +Z (south), 90 -> -X (west), 180 -> -Z (north), 270 -> +X (east)."""
    rad = np.radians(yaw_deg)
    return -np.sin(rad), np.cos(rad)


def main():
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    value = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE))
    value.eval()
    print(f"Loaded AE={AE_WEIGHTS}, V={VALUE_WEIGHTS}")

    # ----- start mission -----
    agent_host = MalmoPython.AgentHost()
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        mission_xml = f.read()
    mission = MalmoPython.MissionSpec(mission_xml, True)
    record = MalmoPython.MissionRecordSpec()

    ws = agent_host.getWorldState()
    while ws.is_mission_running:
        time.sleep(0.1)
        ws = agent_host.getWorldState()

    for retry in range(5):
        try:
            agent_host.startMission(mission, record)
            break
        except MalmoPython.MissionException as e:
            print("startMission retry:", e)
            time.sleep(2)

    print("Waiting for mission...")
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

    print("Mission running. Reactive controller + map.\n")

    # 0=unknown, 1=free, 2=obstacle
    cell_state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    obs_log, action_log = [], []
    pos_log, yaw_log, v_log, cyan_log = [], [], [], []
    diamond_obs = []   # (x, z, yaw, cyan_count)

    prev_pos = None
    prev_action = None
    move_fail_streak = 0
    turn_streak = 0
    turn_dir = A_TURN_L

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

        frame = np.frombuffer(ws.video_frames[-1].pixels, dtype=np.uint8)
        frame = frame.reshape(IMG_H, IMG_W, 3).copy()

        # ----- perception -----
        with torch.no_grad():
            img = to_torch_img(frame).to(DEVICE)
            z_t = ae.encoder(img).squeeze(0)
            v_now = value(z_t.unsqueeze(0)).item()
        cyan_count = int(diamond_pixel_mask(frame).sum())
        cyan_visible = cyan_count >= MIN_PIXELS

        # ----- map update -----
        cx, cz = world_to_cell(x, z_pos)
        if cell_in_grid(cx, cz):
            cell_state[cx - GRID_MIN, cz - GRID_MIN] = 1   # FREE

        # detect failed move = obstacle ahead
        moved = True
        if prev_pos is not None and prev_action == A_MOVE:
            if abs(x - prev_pos[0]) < 0.05 and abs(z_pos - prev_pos[1]) < 0.05:
                moved = False
                dx, dz = yaw_to_forward(yaw)
                ahead_cx = int(np.floor(x + dx))
                ahead_cz = int(np.floor(z_pos + dz))
                if cell_in_grid(ahead_cx, ahead_cz):
                    g = cell_state[ahead_cx - GRID_MIN, ahead_cz - GRID_MIN]
                    if g != 1:
                        cell_state[ahead_cx - GRID_MIN, ahead_cz - GRID_MIN] = 2

        if v_now > V_THRESHOLD or cyan_visible:
            diamond_obs.append((x, z_pos, yaw, cyan_count))

        # ----- reactive controller -----
        if prev_action == A_MOVE and not moved:
            move_fail_streak += 1
        else:
            move_fail_streak = 0

        if v_now > V_THRESHOLD:
            action_idx = A_MOVE
            turn_streak = 0
        elif move_fail_streak >= STUCK_MOVE_LIMIT:
            action_idx = turn_dir
            turn_streak += 1
            move_fail_streak = 0
        elif turn_streak >= TURN_STREAK_LIMIT:
            action_idx = A_MOVE
            turn_streak = 0
        else:
            action_idx = turn_dir
            turn_streak += 1

        cmd = ACTIONS_STR[action_idx]

        obs_log.append(frame)
        action_log.append(action_idx)
        pos_log.append([x, z_pos])
        yaw_log.append(yaw)
        v_log.append(v_now)
        cyan_log.append(cyan_count)

        print(f"step {step:03d} | a={action_idx} ({ACTIONS_NAME[action_idx]:<5}) "
              f"| V={v_now:.3f}  cyan={cyan_count:4d} "
              f"| pos=({x:.2f},{z_pos:.2f}) yaw={yaw:.1f} "
              f"| stuck={move_fail_streak} turn_streak={turn_streak}")

        agent_host.sendCommand(cmd)
        time.sleep(0.20)
        time.sleep(0.05)

        prev_pos = (x, z_pos)
        prev_action = action_idx

    # ----- finalize -----
    if not pos_log:
        print("no frames collected.")
        return
    fx, fz = pos_log[-1]
    diamond_touched = bool(abs(fx - 4.0) < 1.5 and abs(fz - 4.0) < 1.5)
    print(f"\nDiamond touched: {diamond_touched}  (final=({fx:.2f},{fz:.2f}))")
    print(f"Total steps: {len(obs_log)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(OUT_DIR, f"reactive_{ts}.npz")
    np.savez(
        out_npz,
        obs=np.array(obs_log, dtype=np.uint8),
        actions=np.array(action_log, dtype=np.int64),
        v=np.array(v_log, dtype=np.float32),
        positions=np.array(pos_log, dtype=np.float32),
        yaw=np.array(yaw_log, dtype=np.float32),
        cyan=np.array(cyan_log, dtype=np.int64),
        cell_state=cell_state,
        diamond_touched=np.array([diamond_touched], dtype=np.uint8),
    )
    print(f"Saved npz: {out_npz}")

    plot_map_and_truth(
        cell_state, np.array(pos_log), np.array(yaw_log), np.array(v_log),
        diamond_obs, diamond_touched,
        out_npz.replace(".npz", "_map.png"),
    )


def plot_map_and_truth(cell_state, pos, yaw, v, diamond_obs, diamond_touched, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

    for panel, ax in enumerate(axes):
        is_truth = (panel == 1)

        ax.add_patch(Rectangle(
            (-5.5, -5.5), 11, 11,
            facecolor="#fafafa", edgecolor="black", linewidth=2,
        ))

        if is_truth:
            for (ox, oz) in OBSTACLES_GT:
                ax.add_patch(Rectangle((ox - 0.5, oz - 0.5), 1, 1,
                                       facecolor="#888", edgecolor="black"))
            for (gx, gz, name, color) in GOALS_GT:
                ax.add_patch(Rectangle((gx - 0.5, gz - 0.5), 1, 1,
                                       facecolor=color, edgecolor="black", linewidth=2))
                ax.text(gx, gz, name, ha="center", va="center",
                        fontsize=8, weight="bold")
            ax.set_title("Ground truth env3", fontsize=11)
        else:
            for gx in range(GRID_SIZE):
                for gz in range(GRID_SIZE):
                    s = cell_state[gx, gz]
                    cx = gx + GRID_MIN
                    cz = gz + GRID_MIN
                    if s == 1:
                        ax.add_patch(Rectangle((cx - 0.5, cz - 0.5), 1, 1,
                                               facecolor="#cdf5cd",
                                               edgecolor="lightgray", linewidth=0.4))
                    elif s == 2:
                        ax.add_patch(Rectangle((cx - 0.5, cz - 0.5), 1, 1,
                                               facecolor="#cc6666",
                                               edgecolor="black", linewidth=1))
            for (dx, dz, dyaw, npix) in diamond_obs:
                fx_, fz_ = yaw_to_forward(dyaw)
                ax.arrow(dx, dz, fx_ * 0.8, fz_ * 0.8,
                         head_width=0.18, head_length=0.13,
                         fc="cyan", ec="cyan", alpha=0.7,
                         length_includes_head=True, zorder=8)
            ax.set_title(
                f"Agent's belief  |  diamond_touched={diamond_touched}\n"
                f"green=free, red=obstacle, blank=unknown, cyan arrow=diamond seen",
                fontsize=10,
            )

        # trail (both panels)
        ax.plot(pos[:, 0], pos[:, 1], color="gray", lw=0.6, alpha=0.5)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=v, cmap="viridis",
                        s=22, vmin=0, vmax=1, zorder=5,
                        edgecolors="black", linewidths=0.2)
        ax.plot(pos[0, 0], pos[0, 1], 'o', markersize=14,
                markerfacecolor="yellow", markeredgecolor="black", zorder=10,
                label="start")
        ax.plot(pos[-1, 0], pos[-1, 1], 's', markersize=14,
                markerfacecolor="red", markeredgecolor="black", zorder=10,
                label="end")

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("X (east →)")
        ax.set_ylabel("Z (south ↓)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("V_now (P diamond visible)")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved map: {out_path}")


if __name__ == "__main__":
    main()
