"""
Step 3 of the hybrid planner: MPC over the dynamics model, *not* A*.

Same map building as step 2 (free / obstacle / diamond / unknown via
position, bump detection and ray-cast), but the controller is replaced
with model-predictive control that actually uses the world model:

  - At each control step, enumerate all 3^H action sequences of length H.
  - For each sequence:
      (a) Simulate the agent's pose (x, z, yaw) kinematically. Move
          actions advance 1 block in the current heading; if the target
          cell is an OBSTACLE on the map, the move is treated as blocked
          (position unchanged, blocked-counter incremented). Turns rotate
          yaw by ±90°.
      (b) Roll the dynamics model forward to get the predicted latent at
          horizon end, and evaluate V on that latent.
  - Score each sequence by:
        - distance(end_pose, target_cell)        (smaller = better)
        + LAMBDA_V * V_pred                       (higher = better)
        - LAMBDA_BLOCK * blocked_moves            (smaller = better)
  - Send the FIRST action of the best sequence.

Target cell: closest DIAMOND mark if any, otherwise nearest frontier
(unknown adjacent to free). Same as step 2.
"""
import os
import sys
import time
import json
import random
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from detect_diamond import diamond_pixel_mask, diamond_blob_visible

# --- bootstrap ---
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
from dynamics_model import DynamicsTurningMLP
from detect_diamond import diamond_pixel_mask, MIN_PIXELS as CYAN_DATASET_THRESHOLD

CYAN_MIN_PIXELS = 15

# Stronger threshold for actually marking a DIAMOND cell on the map.
# CYAN_MIN_PIXELS is only a weak visibility/logging threshold.
# MARK_MIN_PIXELS should be higher to avoid marking glass/cyan-ish noise as diamond.
MARK_MIN_PIXELS = 80

# ── Edge rejection ──────────────────────────────────────────────────
# Ignore diamond blobs whose bounding box touches the frame edge.
# Partial views at borders are unreliable for direction estimation and
# often come from glass walls or incidental cyan.
EDGE_MARGIN_PX = 2   # pixels from frame border to consider "edge"

# ── Temporal confirmation ───────────────────────────────────────────
# Require the diamond to be visible for CONFIRM_FRAMES consecutive
# frames before committing a DIAMOND mark to the map. This filters
# out single-frame false positives from glass reflections etc.
CONFIRM_FRAMES = 2

# Print diamond localization details for debugging.
DEBUG_DIAMOND_EST = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Environment configs ──────────────────────────────────────────────
# Switch ENV to run in a different room. Everything else adapts.
ENV = "env3"      # "env1", "env2", or "env3"

_ENV_CONFIGS = {
    "env1": {
        "mission": "missions/env1random.xml",
        "occupied": {
            (4, 4),                             # diamond (goal)
            (-4, 4),                            # gold
            (4, -4),                            # emerald
        },
        "obstacles_gt": [],                     # no floor obstacles
    },
    "env2": {
        "mission": "missions/env2random.xml",
        "occupied": {
            (4, 4),                             # diamond (goal)
            (-4, 4),                            # gold
            (4, -4),                            # emerald
        },
        "obstacles_gt": [],                     # no floor obstacles
    },
    "env3": {
        "mission": "missions/env3random.xml",
        "occupied": {
            (-2, -1), (-1, 2), (2, 1),         # obsidian
            (-3, 0), (0, 3),                    # nether_brick
            (1, -2), (3, -1),                   # quartz_block
            (-2, 3), (3, 2), (-1, -3),          # end_stone
            (4, 4),                             # diamond (goal)
            (-4, 4),                            # gold
            (4, -4),                            # emerald
        },
        "obstacles_gt": [(-2, -1), (-1, 2), (2, 1), (-3, 0), (0, 3),
                         (1, -2), (3, -1), (-2, 3), (3, 2), (-1, -3)],
    },
}

_cfg = _ENV_CONFIGS[ENV]
MISSION_FILE = _cfg["mission"]
OCCUPIED_CELLS = _cfg["occupied"]


def pick_random_spawn(rng=None):
    rng = rng or random
    free = [(bx, bz)
            for bx in range(-4, 5)
            for bz in range(-4, 5)
            if (bx, bz) not in OCCUPIED_CELLS]
    bx, bz = rng.choice(free)
    yaw = rng.choice([0, 90, 180, 270])
    return bx + 0.5, bz + 0.5, yaw

AE_WEIGHTS = "ae_multienv.pth"
DYN_WEIGHTS = "dynamics_multienv.pth"
VALUE_WEIGHTS = "value_head_multienv.pth"

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 256

ACTIONS_STR = ["move 1", "turn 0.2", "turn -0.2"]
ACTIONS_NAME = ["move", "turnR", "turnL"]
TURN_DIR_FALLBACK = 1
STUCK_TURN_AFTER = 3
MAX_STEPS = 200
IMG_H, IMG_W = 64, 64

# MPC config
MPC_HORIZON = 4
FOV_DEG = 70.0          # for cyan-centroid ray cast direction

# ── Scoring weights (tweak these to change world-model influence) ──
W_DIST    = -100.0      # distance to target (negative = closer is better)
W_BLOCK   = -10.0       # blocked moves penalty
W_SUMDIST = -1.0        # cumulative path distance
W_V       = 50.0        # value prediction (positive = higher V is better)

# ── Distance-dependent V weight ────────────────────────────────────
# When the agent is very close to the diamond (dist ≤ 1), V can hurt:
# stepping ONTO the diamond makes cyan disappear → V drops → agent avoids
# the last step.  Fix: fade V weight to 0 as dist → 0.
#   effective_v_weight(d) = W_V * min(1.0, d / V_FADE_DIST)
# At dist=0: weight=0  (V is silenced, map distance decides)
# At dist≥V_FADE_DIST: weight=W_V (full world-model contribution)
V_FADE_DIST = 3.0       # distance at which V reaches full weight

ROOM_HALF = 5
GRID_LO, GRID_HI = -ROOM_HALF, ROOM_HALF
GRID_SIZE = 2 * ROOM_HALF + 1

UNKNOWN, FREE, OBSTACLE, DIAMOND = 0, 1, 2, 3

OUT_DIR = "world_model_out/closed_loop"
os.makedirs(OUT_DIR, exist_ok=True)


# ============= helpers =============
def to_torch_img(frame):
    return (torch.from_numpy(frame).float().div_(255.0)
            .permute(2, 0, 1).unsqueeze(0))


def world_to_grid(x, z):
    gx = int(np.clip(np.floor(x) - GRID_LO, 0, GRID_SIZE - 1))
    gz = int(np.clip(np.floor(z) - GRID_LO, 0, GRID_SIZE - 1))
    return gx, gz


def yaw_to_forward(yaw_deg):
    rad = np.radians(yaw_deg)
    return -np.sin(rad), np.cos(rad)


def mark_current_free(grid, x, z):
    gx, gz = world_to_grid(x, z)

    # Do not erase a DIAMOND hypothesis just because the agent is currently
    # standing in the same discretized grid cell.
    if grid[gz, gx] != DIAMOND:
        grid[gz, gx] = FREE


def maybe_mark_diamond(grid, x, z, yaw, frame, diamond_visible, blob,
                       confirm_counter=0):
    """
    Mark a possible DIAMOND cell using the connected diamond-colored blob.

    Important difference from the previous version:
    - Previous version used the centroid of ALL cyan pixels.
      That can be pulled toward glass walls or cyan-like noise.
    - This version uses the largest valid connected blob returned by
      diamond_blob_visible(frame), so the ray-cast direction is based on the
      actual compact target-like region.

    Temporal confirmation: diamond must be visible for CONFIRM_FRAMES
    consecutive frames before we commit a mark to the map.
    """
    if not diamond_visible or blob is None:
        return

    # ── Temporal confirmation ──
    if confirm_counter < CONFIRM_FRAMES:
        if DEBUG_DIAMOND_EST:
            print(f"  DIAMOND_EST WAIT confirm={confirm_counter}/{CONFIRM_FRAMES}")
        return

    area = int(blob.get("area", 0))

    # Do not create a map target from weak detections.
    if area < MARK_MIN_PIXELS:
        return

    # ── Edge rejection: skip blobs touching the frame border ──
    # bbox format from detect_diamond: (x1, y1, x2, y2) — top-left to bottom-right
    bbox = blob.get("bbox", None)
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        img_h, img_w_full = frame.shape[:2]
        if (bx1 < EDGE_MARGIN_PX or by1 < EDGE_MARGIN_PX
                or bx2 > (img_w_full - 1 - EDGE_MARGIN_PX)
                or by2 > (img_h - 1 - EDGE_MARGIN_PX)):
            if DEBUG_DIAMOND_EST:
                print(f"  DIAMOND_EST EDGE_REJECT bbox={bbox}")
            return

    cx_pixel = float(blob["cx"])
    img_w = frame.shape[1]
    img_center = (img_w - 1) / 2.0

    # Convert horizontal image position to yaw offset.
    # Left side of image -> negative offset.
    # Right side of image -> positive offset.
    angle_offset = ((cx_pixel - img_center) / img_center) * (FOV_DEG / 2.0)
    effective_yaw = (yaw + angle_offset) % 360.0

    MIN_RAY_K = 3
    dx, dz = yaw_to_forward(effective_yaw)

    farthest = None
    hit_obstacle_early = False

    for k in range(1, ROOM_HALF * 2 + 2):
        tx, tz = x + dx * k, z + dz * k

        if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
            break

        tgx, tgz = world_to_grid(tx, tz)
        cell = grid[tgz, tgx]

        if cell == OBSTACLE:
            if k < MIN_RAY_K:
                hit_obstacle_early = True
            break

        if cell == UNKNOWN and k >= MIN_RAY_K:
            farthest = (tgx, tgz)

    if farthest is not None and not hit_obstacle_early:
        # Keep only the newest strong DIAMOND hypothesis.
        # This prevents fake cyan regions from accumulating into a vertical strip.
        grid[grid == DIAMOND] = UNKNOWN
        grid[farthest[1], farthest[0]] = DIAMOND

    if DEBUG_DIAMOND_EST:
        print(
            f"  DIAMOND_EST area={area} "
            f"bbox={blob.get('bbox', None)} "
            f"aspect={blob.get('aspect', 0):.2f} "
            f"fill={blob.get('fill', 0):.2f} "
            f"cx={cx_pixel:.1f} "
            f"yaw={yaw:.1f} eff_yaw={effective_yaw:.1f} "
            f"marked={farthest}"
        )


def maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action, cur_x, cur_z):
    if prev_action != 0 or prev_x is None:
        return
    if abs(cur_x - prev_x) > 0.05 or abs(cur_z - prev_z) > 0.05:
        return
    dx, dz = yaw_to_forward(prev_yaw)
    tx, tz = prev_x + dx, prev_z + dz
    if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
        return
    tgx, tgz = world_to_grid(tx, tz)

    # Do not overwrite a DIAMOND hypothesis as OBSTACLE.
    if grid[tgz, tgx] != DIAMOND:
        grid[tgz, tgx] = OBSTACLE


def find_diamond_cells_sorted(grid, current_gxgz):
    """Return all DIAMOND-marked cells sorted by Manhattan distance to agent."""
    cgx, cgz = current_gxgz
    locs = np.argwhere(grid == DIAMOND)
    return sorted(
        ((int(gx), int(gz)) for gz, gx in locs),
        key=lambda c: abs(c[0] - cgx) + abs(c[1] - cgz),
    )


def find_frontier_cell(grid, start_gxgz):
    """BFS for the closest UNKNOWN cell adjacent to a FREE cell."""
    visited = {start_gxgz}
    queue = [start_gxgz]
    while queue:
        cur = queue.pop(0)
        gx, gz = cur
        for dgx, dgz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ngx, ngz = gx + dgx, gz + dgz
            if not (0 <= ngx < GRID_SIZE and 0 <= ngz < GRID_SIZE):
                continue
            n = (ngx, ngz)
            if n in visited:
                continue
            visited.add(n)
            cell = grid[ngz, ngx]
            if cell == OBSTACLE:
                continue
            if cell == UNKNOWN:
                return n
            if cell in (FREE, DIAMOND):
                queue.append(n)
    return None


# ============= MPC core =============
def mpc_plan(grid, x, z, yaw, frame, ae, dyn, value, target_cell):
    """Enumerate all action sequences of length MPC_HORIZON, score by
    kinematic distance + V_pred at horizon, return the first action of
    the best sequence."""
    H = MPC_HORIZON
    seqs = np.array(list(itertools.product(range(NUM_ACTIONS), repeat=H)),
                    dtype=np.int64)             # (N, H)
    N = seqs.shape[0]

    # ---- batched kinematic simulation ----
    xs = np.full(N, x, dtype=np.float32)
    zs = np.full(N, z, dtype=np.float32)
    yaws = np.full(N, yaw, dtype=np.float32)
    blocked = np.zeros(N, dtype=np.float32)

    # track distance to target after each step so we can reward sequences
    # that reach the goal *earlier* in the rollout, not just at the end.
    dist_per_step = np.zeros((N, H), dtype=np.float32)

    for h in range(H):
        a = seqs[:, h]                          # (N,)
        # turns first (vectorized)
        yaws = np.where(a == 1, (yaws + 90.0) % 360.0, yaws)
        yaws = np.where(a == 2, (yaws - 90.0) % 360.0, yaws)

        # moves: must check obstacle per-sequence
        is_move = (a == 0)
        if is_move.any():
            rad = np.radians(yaws)
            dx_v = -np.sin(rad)
            dz_v = np.cos(rad)
            new_xs = xs + dx_v
            new_zs = zs + dz_v

            # check OBSTACLE in target cell for each move
            for i in np.where(is_move)[0]:
                gx_t, gz_t = world_to_grid(float(new_xs[i]), float(new_zs[i]))
                if grid[gz_t, gx_t] == OBSTACLE:
                    blocked[i] += 1
                else:
                    xs[i] = new_xs[i]
                    zs[i] = new_zs[i]

        # record dist-to-target after this action
        gx_h = np.clip(np.floor(xs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
        gz_h = np.clip(np.floor(zs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
        dist_per_step[:, h] = (
            np.abs(gx_h - target_cell[0]) + np.abs(gz_h - target_cell[1])
        )

    # final cell of each sequence
    end_gx = np.clip(np.floor(xs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
    end_gz = np.clip(np.floor(zs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
    dist = np.abs(end_gx - target_cell[0]) + np.abs(end_gz - target_cell[1])
    sum_dist = dist_per_step.sum(axis=1)

    # ---- batched dyn rollout for V_pred ----
    with torch.no_grad():
        img = to_torch_img(frame).to(DEVICE)
        z0 = ae.encoder(img).squeeze(0)            # (latent_dim,)
        z_batch = z0.unsqueeze(0).expand(N, -1).contiguous()
        a_t = torch.from_numpy(seqs).long().to(DEVICE)   # (N, H)

        for h in range(H):
            z_batch = dyn(z_batch, a_t[:, h])

        v_pred = value(z_batch).cpu().numpy()      # (N,)

    # ---- score and pick ----
    # Distance-dependent V weight: fade V to 0 when close to target
    # so the map planner handles the final approach.
    v_weight = W_V * np.minimum(1.0, dist.astype(np.float32) / V_FADE_DIST)

    score = (
        W_DIST * dist.astype(np.float32)
        + W_BLOCK * blocked
        + W_SUMDIST * sum_dist
        + v_weight * v_pred
    )

    best_idx = int(np.argmax(score))
    best_action = int(seqs[best_idx, 0])

    return best_action, float(v_pred[best_idx]), float(dist[best_idx]), float(blocked[best_idx])


def planned_action(grid, current_gxgz, current_x, current_z, current_yaw,
                   frame, ae, dyn, value, stuck_count, prev_action):
    """Pick a target cell, run MPC, return action.

    Target priority: closest DIAMOND if any -> nearest frontier -> stuck escape.
    MPC uses the map for distance/obstacle scoring with V_pred as a
    meaningful tiebreaker (weight 15 inside mpc_plan).
    """
    cgx, cgz = current_gxgz

    diamond_cells = find_diamond_cells_sorted(grid, current_gxgz)
    target = None
    target_label = None

    for cell in diamond_cells:
        if cell == current_gxgz:
            continue
        target = cell
        target_label = "DIAMOND"
        break

    if target is None:
        frontier = find_frontier_cell(grid, current_gxgz)
        if frontier is not None and frontier != current_gxgz:
            target = frontier
            target_label = "frontier"

    if target is None:
        if stuck_count >= STUCK_TURN_AFTER and prev_action == 0:
            return TURN_DIR_FALLBACK, "no_target_turn", 0.0, 0.0
        return TURN_DIR_FALLBACK, "no_target", 0.0, 0.0

    action, v_pred, dist, blocked = mpc_plan(
        grid, current_x, current_z, current_yaw, frame, ae, dyn, value, target
    )

    reason = (f"MPC->{target_label} ({target[0]},{target[1]})  "
              f"V={v_pred:.2f} dist={dist:.0f} blk={blocked:.0f}")

    if stuck_count >= STUCK_TURN_AFTER:
        if action == 0 and prev_action == 0:
            return TURN_DIR_FALLBACK, "stuck_turn (override MPC)", v_pred, dist
        if action != 0 and prev_action != 0:
            return 0, "stuck_move (override MPC)", v_pred, dist

    return action, reason, v_pred, dist


# ============= main =============
def main():
    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()

    dyn = DynamicsTurningMLP(
        latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS, hidden=512
    ).to(DEVICE)
    dyn.load_state_dict(torch.load(DYN_WEIGHTS, map_location=DEVICE))
    dyn.eval()

    value = ValueHead(latent_dim=LATENT_DIM, hidden=HIDDEN).to(DEVICE)
    value.load_state_dict(torch.load(VALUE_WEIGHTS, map_location=DEVICE))
    value.eval()

    print(f"Loaded AE={AE_WEIGHTS}  DYN={DYN_WEIGHTS}  V={VALUE_WEIGHTS}")
    print(f"MPC horizon={MPC_HORIZON}, sequences=3^{MPC_HORIZON}={NUM_ACTIONS**MPC_HORIZON}")

    agent_host = MalmoPython.AgentHost()
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        mission_xml = f.read()
    sx, sz, syaw = pick_random_spawn()
    mission_xml = (mission_xml
                   .replace("{{START_X}}", f"{sx}")
                   .replace("{{START_Z}}", f"{sz}")
                   .replace("{{START_YAW}}", f"{syaw}"))
    print(f"Random spawn: x={sx}, z={sz}, yaw={syaw}")
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

    print("Mission running. MPC over dynamics model.\n")

    grid = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.uint8)
    grid[0, :] = OBSTACLE
    grid[GRID_SIZE - 1, :] = OBSTACLE
    grid[:, 0] = OBSTACLE
    grid[:, GRID_SIZE - 1] = OBSTACLE

    pos_list, yaw_list, v_list, action_list = [], [], [], []
    prev_x = prev_z = prev_yaw = None
    prev_action = None
    stuck_count = 0
    same_pos_count = 0          # counts ANY steps at the same position (incl. turns)
    diamond_confirm = 0         # consecutive frames with diamond visible

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

        vf = ws.video_frames[-1]
        frame_raw = (np.frombuffer(vf.pixels, dtype=np.uint8)
                     .reshape(vf.height, vf.width, 3))
        frame = np.array(Image.fromarray(frame_raw).resize(
            (IMG_W, IMG_H), Image.LANCZOS))

        cyan_count = int(diamond_pixel_mask(frame).sum())
        diamond_visible, blob = diamond_blob_visible(frame)

        # ── Temporal confirmation counter ──
        if diamond_visible and blob is not None and int(blob.get("area", 0)) >= MARK_MIN_PIXELS:
            diamond_confirm += 1
        else:
            diamond_confirm = 0

        # update map
        maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action, x, z_pos)
        mark_current_free(grid, x, z_pos)
        maybe_mark_diamond(grid, x, z_pos, yaw, frame, diamond_visible, blob,
                           confirm_counter=diamond_confirm)

        # stuck counter (failed moves only)
        if prev_action == 0:
            if (prev_x is not None
                    and abs(x - prev_x) < 0.05
                    and abs(z_pos - prev_z) < 0.05):
                stuck_count += 1
            else:
                stuck_count = 0

        # same-position counter: catches spinning in place (turn oscillation)
        if (prev_x is not None
                and abs(x - prev_x) < 0.05
                and abs(z_pos - prev_z) < 0.05):
            same_pos_count += 1
        else:
            same_pos_count = 0

        # if spinning in place for too long, force a move forward
        if same_pos_count >= 6 and prev_action != 0:
            action_idx = 0  # force move
            agent_host.sendCommand(ACTIONS_STR[0])
            same_pos_count = 0
            print(f"  [SPIN BREAK: forced move at step {step}]")

            pos_list.append([x, z_pos])
            yaw_list.append(yaw)
            v_list.append(0.0)
            action_list.append(0)

            prev_x, prev_z, prev_yaw, prev_action = x, z_pos, yaw, 0
            time.sleep(0.25)
            continue

        if stuck_count >= STUCK_TURN_AFTER * 4 and (grid == DIAMOND).any():
            grid[grid == DIAMOND] = UNKNOWN
            stuck_count = 0
            print(f"  [cleared stale DIAMOND marks at step {step}]")

        cur_grid = world_to_grid(x, z_pos)

        action_idx, reason, v_pred, dist = planned_action(
            grid, cur_grid, x, z_pos, yaw, frame,
            ae, dyn, value, stuck_count, prev_action
        )

        agent_host.sendCommand(ACTIONS_STR[action_idx])

        pos_list.append([x, z_pos])
        yaw_list.append(yaw)
        v_list.append(v_pred)
        action_list.append(action_idx)

        marker = ""
        if diamond_visible:
            marker += "  cyan↑"
        if stuck_count >= STUCK_TURN_AFTER:
            marker += "  STUCK"

        print(f"step {step:03d} | a={action_idx} ({ACTIONS_NAME[action_idx]:<5}) "
              f"| {reason:<48} | cyan={cyan_count:4d} "
              f"| pos=({x:5.2f},{z_pos:5.2f}) yaw={yaw:5.1f}{marker}")

        prev_x, prev_z, prev_yaw, prev_action = x, z_pos, yaw, action_idx

        time.sleep(0.25)

    if len(pos_list) > 0:
        print(f"\nfinal pos=({pos_list[-1][0]:.2f},{pos_list[-1][1]:.2f})  "
              f"steps={len(pos_list)}")
    else:
        print("\nNo valid steps collected.")

    save_run(grid, pos_list, yaw_list, v_list, action_list)


def save_run(grid, positions, yaws, values, actions):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(OUT_DIR, f"map_step3_{ts}.npz")

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

    cmap = plt.matplotlib.colors.ListedColormap(
        ["#cccccc", "#ffffff", "#222222", "#5be0d2"]
    )

    extent = [GRID_LO - 0.5, GRID_HI + 0.5, GRID_HI + 0.5, GRID_LO - 0.5]
    axes[0].imshow(grid, cmap=cmap, vmin=0, vmax=3, extent=extent, origin="upper")
    axes[0].set_title("Belief map (MPC nav)\n"
                      "gray=unknown, white=free, black=obstacle, cyan=diamond_seen",
                      fontsize=10)

    OBSTACLES_GT = _cfg["obstacles_gt"]
    GOALS = [(4, 4, "DIAMOND", "#5be0d2"),
             (-4, 4, "gold", "#fff570"),
             (4, -4, "emerald", "#50d250")]

    axes[1].add_patch(Rectangle((-ROOM_HALF - 0.5, -ROOM_HALF - 0.5),
                                2 * ROOM_HALF + 1, 2 * ROOM_HALF + 1,
                                facecolor="white", edgecolor="black", linewidth=3))

    for (xc, zc) in OBSTACLES_GT:
        axes[1].add_patch(Rectangle((xc - 0.5, zc - 0.5), 1, 1,
                                    facecolor="#222", edgecolor="black"))

    for (xc, zc, name, color) in GOALS:
        axes[1].add_patch(Rectangle((xc - 0.5, zc - 0.5), 1, 1,
                                    facecolor=color, edgecolor="black", linewidth=2))
        axes[1].text(xc, zc, name, ha="center", va="center",
                     fontsize=8, weight="bold")

    axes[1].set_title(f"Ground truth ({ENV})", fontsize=10)

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
    out_png = os.path.join(OUT_DIR, f"map_step3_{ts}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved map: {out_png}")


if __name__ == "__main__":
    main()