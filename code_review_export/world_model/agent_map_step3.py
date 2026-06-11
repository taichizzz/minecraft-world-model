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
EDGE_MARGIN_PX = 2    # pixels from frame border to consider "edge"

# ── Temporal confirmation ───────────────────────────────────────────
# Require the diamond to be visible for CONFIRM_FRAMES consecutive
# frames before committing a DIAMOND mark to the map. This filters
# out single-frame false positives from glass reflections etc.
CONFIRM_FRAMES = 2

# ── Stale diamond timeout ───────────────────────────────────────────
# If the diamond hasn't been re-confirmed (visible with area >= threshold)
# for this many steps, clear the DIAMOND mark from the map. Prevents
# chasing a wrong hypothesis while the agent keeps moving (not stuck).
DIAMOND_STALE_STEPS = 15

# ── Goal memory (env4 diagnosis: failures spend ~105/107 steps exploring,
# never holding a goal lock; sightings were discarded or marks evaporated) ──
# "evidence": a confirmed mark is cleared only by COUNTER-EVIDENCE (looking
#             at the cell from close range and seeing no cyan), since the
#             diamond is a static landmark. "timer": legacy 15-step decay.
GOAL_MEMORY = "evidence"
GOAL_SEE_RANGE = 3.5          # blocks: close enough that detector must fire
GOAL_SEE_HALF_FOV_DEG = 25.0  # looking-at-cell tolerance (camera FOV/2 - margin)
COUNTEREVIDENCE_FRAMES = 3    # consecutive should-see-it misses before clearing
HINT_TTL_STEPS = 30           # unconfirmed-glimpse soft target lifetime
SWEEP_AFTER_STEPS = 25        # sighting-less steps before a 360° look-around
K_AREA_DIST = 2000.0          # blob area ~ K/d^2  ->  d̂ = sqrt(K/area)

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
    # ── Generalization envs: different forms (layouts) and factors
    # (materials/lighting). Same 11x11 frame + diamond goal at (4,4) so the
    # trained world model can be tested directly. ──
    "env4": {
        "mission": "missions/env4random.xml",
        "occupied": {
            (-3, 1), (-1, -1), (1, 3), (0, 2), (1, -2), (2, 2), (3, 0),
            (3, -3), (-2, -3), (-3, 3),
            (4, 4), (-4, 4), (4, -4),
        },
        "obstacles_gt": [(-3, 1), (-1, -1), (1, 3), (0, 2), (1, -2),
                         (2, 2), (3, 0), (3, -3), (-2, -3), (-3, 3)],
    },
    "env5": {
        "mission": "missions/env5random.xml",
        "occupied": {
            (-2, -2), (2, 1), (-2, 2), (-3, -1), (0, 0), (1, 2), (2, -2), (3, 2),
            (4, 4), (-4, 4), (4, -4),
        },
        "obstacles_gt": [(-2, -2), (2, 1), (-2, 2), (-3, -1), (0, 0),
                         (1, 2), (2, -2), (3, 2)],
    },
    "env6": {
        "mission": "missions/env6random.xml",
        "occupied": {
            (-2, 1), (1, -2), (2, 2), (-1, -1), (0, 3),
            (4, 4), (-4, 4), (4, -4),
        },
        "obstacles_gt": [(-2, 1), (1, -2), (2, 2), (-1, -1), (0, 3)],
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

# Current best stack ("p2b_vc", see MODELS.md). Eval scripts may override.
AE_WEIGHTS = "ae_predictive.pth"
DYN_WEIGHTS = "dynamics_predictive_vc.pth"
VALUE_WEIGHTS = "value_head_dist_pred.pth"

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
GAMMA_V   = 0.95        # discount for per-step value sum: score += W_V * Σ γ^h V(z_h)

# ── Distance-dependent V weight ────────────────────────────────────
# When the agent is very close to the diamond (dist ≤ 1), V can hurt:
# stepping ONTO the diamond makes cyan disappear → V drops → agent avoids
# the last step.  Fix: fade V weight to 0 as dist → 0.
#   effective_v_weight(d) = W_V * min(1.0, d / V_FADE_DIST)
# At dist=0: weight=0  (V is silenced, map distance decides)
# At dist≥V_FADE_DIST: weight=W_V (full world-model contribution)
V_FADE_DIST = 3.0       # distance at which V reaches full weight

# ── Localization noise (Track C necessity proof) ───────────────────
# Models how the agent's SELF-REPORTED position degrades. Only the
# geometric terms (dist/blocked/sum_dist) and map-building consume this
# corrupted estimate; the world-model value term reads PIXELS, so it is
# immune. Success/SPL are ALWAYS scored on the TRUE position.
#
#   POS_NOISE_MODEL:
#     "drift"  random walk: offset += N(0,std) every step. Realistic
#              odometry/SLAM error that accumulates over time (default).
#     "bias"   one constant offset ~ N(0,std) per episode, persistent.
#     "jitter" fresh N(0,std) every step. Pathological — averages out and
#              flips grid cells each frame; kept only for comparison.
#   POS_NOISE_STD: std in blocks. 0.0 == off (nominal agent, no change).
POS_NOISE_MODEL = "drift"
POS_NOISE_STD = 0.0
_pos_noise_offset = np.array([0.0, 0.0])   # live per-episode localization error


def reset_pos_noise():
    """Reset the localization-error state at the start of each episode."""
    global _pos_noise_offset
    if POS_NOISE_STD > 0.0 and POS_NOISE_MODEL == "bias":
        _pos_noise_offset = np.random.normal(0.0, POS_NOISE_STD, size=2)
    else:
        _pos_noise_offset = np.array([0.0, 0.0])


def noisy_pos(x_true, z_true):
    """Position the PLANNER believes, per POS_NOISE_MODEL. Advances the
    drift/jitter state; 'bias' stays fixed from reset_pos_noise(). Metrics
    still use the true coordinates."""
    global _pos_noise_offset
    if POS_NOISE_STD <= 0.0:
        return x_true, z_true
    if POS_NOISE_MODEL == "drift":
        _pos_noise_offset = _pos_noise_offset + np.random.normal(0.0, POS_NOISE_STD, size=2)
    elif POS_NOISE_MODEL == "jitter":
        _pos_noise_offset = np.random.normal(0.0, POS_NOISE_STD, size=2)
    return (x_true + float(_pos_noise_offset[0]),
            z_true + float(_pos_noise_offset[1]))

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


def est_dist_from_area(area):
    """Coarse range estimate from blob size: apparent area ~ K/d^2."""
    d = float(np.sqrt(K_AREA_DIST / max(float(area), 1.0)))
    return float(np.clip(d, 1.0, 2 * ROOM_HALF - 1))


def bearing_diff_deg(x, z, yaw, cell):
    """|angle| between camera forward and the direction to cell center."""
    tx, tz = cell[0] + GRID_LO + 0.5, cell[1] + GRID_LO + 0.5   # grid -> world
    dx, dz = tx - x, tz - z
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return 0.0
    yaw_to = np.degrees(np.arctan2(-dx, dz)) % 360.0
    return abs(((yaw_to - yaw + 180.0) % 360.0) - 180.0)


def looking_at_cell(x, z, yaw, cell):
    """True if cell is within detector-reliable range AND inside the camera
    cone — i.e. the detector SHOULD fire if a diamond were really there.
    Interior obstacles are 1 block high, the camera sees over them, so no
    occlusion test is needed inside the room."""
    tx, tz = cell[0] + GRID_LO + 0.5, cell[1] + GRID_LO + 0.5
    if np.hypot(tx - x, tz - z) > GOAL_SEE_RANGE:
        return False
    return bearing_diff_deg(x, z, yaw, cell) <= GOAL_SEE_HALF_FOV_DEG


def blob_at_edge(blob, frame):
    """Blob bbox touches the frame border (bearing estimate unreliable)."""
    bbox = blob.get("bbox", None)
    if bbox is None:
        return False
    bx1, by1, bx2, by2 = bbox
    img_h, img_w = frame.shape[:2]
    return (bx1 < EDGE_MARGIN_PX or by1 < EDGE_MARGIN_PX
            or bx2 > (img_w - 1 - EDGE_MARGIN_PX)
            or by2 > (img_h - 1 - EDGE_MARGIN_PX))


def estimate_sighting_cell(grid, x, z, yaw, blob, frame, want_unknown):
    """Best-guess map cell for a sighted blob.

    Ray-cast along the blob bearing, but — unlike the old version — the ray
    passes OVER believed interior obstacles (they are 1 block high; the
    camera sees over them; the old 'break on OBSTACLE' silently discarded
    confirmed sightings in cluttered/explored rooms — the env4 failure).
    Among candidates, pick the one nearest the blob-size range estimate d̂
    (the old 'farthest unknown' overshot past the diamond).

    want_unknown=True  -> hard mark: only UNKNOWN/DIAMOND cells qualify
                          (a FREE cell was walked on, so it can't hold the
                          diamond block).
    want_unknown=False -> soft hint: any non-OBSTACLE cell qualifies.
    """
    cx_pixel = float(blob["cx"])
    img_w = frame.shape[1]
    img_center = (img_w - 1) / 2.0
    angle_offset = ((cx_pixel - img_center) / img_center) * (FOV_DEG / 2.0)
    effective_yaw = (yaw + angle_offset) % 360.0
    dx, dz = yaw_to_forward(effective_yaw)
    d_hat = est_dist_from_area(blob.get("area", 1))

    best = None
    for k in range(1, ROOM_HALF * 2 + 2):
        tx, tz = x + dx * k, z + dz * k
        if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
            break
        tgx, tgz = world_to_grid(tx, tz)
        c = grid[tgz, tgx]
        if want_unknown:
            ok = c in (UNKNOWN, DIAMOND)
        else:
            ok = c != OBSTACLE
        if ok:
            score = abs(k - d_hat)
            if best is None or score < best[0]:
                best = (score, (tgx, tgz))
    return None if best is None else best[1]


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
        return None, "not_visible"

    # ── Temporal confirmation ──
    if confirm_counter < CONFIRM_FRAMES:
        return None, "wait_confirm"

    area = int(blob.get("area", 0))

    # Do not create a map target from weak detections.
    if area < MARK_MIN_PIXELS:
        return None, "weak"

    # ── Edge rejection: skip blobs touching the frame border ──
    if blob_at_edge(blob, frame):
        return None, "edge"

    marked = estimate_sighting_cell(grid, x, z, yaw, blob, frame,
                                    want_unknown=True)
    if marked is not None:
        # Keep only the newest strong DIAMOND hypothesis.
        grid[grid == DIAMOND] = UNKNOWN
        grid[marked[1], marked[0]] = DIAMOND
        if DEBUG_DIAMOND_EST:
            print(f"  DIAMOND_EST area={area} dhat={est_dist_from_area(area):.1f} "
                  f"yaw={yaw:.1f} marked={marked}")
        return marked, "marked"
    return None, "no_candidate"


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


BFS_UNREACHABLE = 999.0


def bfs_dist_field(grid, target_cell):
    """Geodesic distance (in moves) from every cell to target_cell on the
    BELIEVED map: 4-connected BFS that does not pass through OBSTACLE cells.
    UNKNOWN is optimistically passable (needed to plan toward frontiers).

    Replaces straight-line Manhattan distance in MPC scoring: Manhattan
    measures through walls, so in cluttered rooms it parks the agent against
    the wall nearest the target (the env4 failure). The geodesic knows that
    cell is far in *path* terms. Unreachable cells get BFS_UNREACHABLE.
    """
    tgx, tgz = int(target_cell[0]), int(target_cell[1])
    field = np.full((GRID_SIZE, GRID_SIZE), BFS_UNREACHABLE, dtype=np.float32)
    if grid[tgz, tgx] == OBSTACLE:
        return field
    field[tgz, tgx] = 0.0
    queue = [(tgx, tgz)]
    while queue:
        cx, cz = queue.pop(0)
        d = field[cz, cx] + 1.0
        for nx, nz in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
            if not (0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE):
                continue
            if grid[nz, nx] == OBSTACLE or field[nz, nx] <= d:
                continue
            field[nz, nx] = d
            queue.append((nx, nz))
    return field


# ============= MPC core =============
def mpc_plan(grid, x, z, yaw, frame, ae, dyn, value, target_cell):
    """Enumerate all action sequences of length MPC_HORIZON, score by
    kinematic distance + V_pred at horizon, return the first action of
    the best sequence. Distance-to-target = BFS geodesic on the believed
    map (falls back to Manhattan if belief says target is unreachable)."""
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

    # geodesic distance field on the believed map; if the agent's own cell
    # can't reach the target in belief (sealed pocket / bad marks), fall
    # back to Manhattan so scoring stays informative rather than flat.
    dfield = bfs_dist_field(grid, target_cell)
    agx, agz = world_to_grid(float(x), float(z))
    if dfield[agz, agx] >= BFS_UNREACHABLE:
        jj, ii = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE),
                             indexing="ij")          # jj=gz rows, ii=gx cols
        dfield = (np.abs(ii - target_cell[0])
                  + np.abs(jj - target_cell[1])).astype(np.float32)

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

        # record geodesic dist-to-target after this action
        gx_h = np.clip(np.floor(xs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
        gz_h = np.clip(np.floor(zs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
        dist_per_step[:, h] = dfield[gz_h, gx_h]

    # final cell of each sequence
    end_gx = np.clip(np.floor(xs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
    end_gz = np.clip(np.floor(zs).astype(int) - GRID_LO, 0, GRID_SIZE - 1)
    dist = dfield[end_gz, end_gx]
    sum_dist = dist_per_step.sum(axis=1)

    # ---- batched dyn rollout: discounted V sum Σ γ^h V(z_h) ----
    with torch.no_grad():
        img = to_torch_img(frame).to(DEVICE)
        z0 = ae.encoder(img).squeeze(0)            # (latent_dim,)
        z_batch = z0.unsqueeze(0).expand(N, -1).contiguous()
        a_t = torch.from_numpy(seqs).long().to(DEVICE)   # (N, H)

        v_pred = np.zeros(N, dtype=np.float32)
        for h in range(H):
            z_batch = dyn(z_batch, a_t[:, h])
            v_pred += (GAMMA_V ** h) * value(z_batch).cpu().numpy()  # (N,)

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
                   frame, ae, dyn, value, stuck_count, prev_action,
                   hint_cell=None):
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

    # unconfirmed-glimpse soft target: go verify the sighting instead of
    # dropping the evidence (fails -> hint expires, exploration resumes)
    if target is None and hint_cell is not None and hint_cell != current_gxgz:
        hgx, hgz = hint_cell
        if grid[hgz, hgx] != OBSTACLE:
            target = hint_cell
            target_label = "HINT"

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


# ============= model loading =============
def load_models():
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
    return ae, dyn, value


# ============= single episode =============
DIAMOND_POS = (4.5, 4.5)

def run_episode(agent_host, ae, dyn, value, save=True, verbose=True, spawn=None):
    """Run one episode. Returns a stats dict."""
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        mission_xml = f.read()
    sx, sz, syaw = spawn if spawn is not None else pick_random_spawn()
    mission_xml = (mission_xml
                   .replace("{{START_X}}", f"{sx}")
                   .replace("{{START_Z}}", f"{sz}")
                   .replace("{{START_YAW}}", f"{syaw}"))
    if verbose:
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
            if verbose:
                print("startMission retry:", e)
            time.sleep(2)

    if not started:
        raise RuntimeError("could not start mission")

    if verbose:
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

    if verbose:
        print("Mission running. MPC over dynamics model.\n")

    grid = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.uint8)
    grid[0, :] = OBSTACLE
    grid[GRID_SIZE - 1, :] = OBSTACLE
    grid[:, 0] = OBSTACLE
    grid[:, GRID_SIZE - 1] = OBSTACLE

    pos_list, yaw_list, v_list, action_list = [], [], [], []
    truepos_list = []        # TRUE positions, for success/SPL under POS_NOISE_STD
    # step-outcome counters: expose WHY an episode timed out (diamond-chasing,
    # endless frontier exploration, no-target spinning, stuck overrides)
    nsteps = {"diamond": 0, "frontier": 0, "no_target": 0,
              "stuck_override": 0, "spin_break": 0, "hint": 0, "sweep": 0}
    goal_events = []               # (what, step, ...) — mark/clear/hint/sweep log
    hint_cell = None               # unconfirmed-glimpse soft target
    hint_step = -10**9
    steps_since_sighting = 10**9   # large -> sweep allowed early if needed
    goal_counterev = 0             # consecutive should-see-it detector misses
    sweep_queue = []               # pending 360° look-around turns
    prev_x = prev_z = prev_yaw = None
    prev_action = None
    stuck_count = 0
    same_pos_count = 0
    diamond_confirm = 0
    steps_since_diamond_seen = 0   # for stale diamond timeout
    reset_pos_noise()              # per-episode localization-error state

    for step in range(MAX_STEPS):
        ws = agent_host.getWorldState()

        if not ws.is_mission_running:
            if verbose:
                print(f"Mission ended at step {step}")
            break

        if (len(ws.observations) == 0 or ws.observations[-1].text == "{}"
                or len(ws.video_frames) == 0):
            time.sleep(0.05)
            continue

        obs_json = json.loads(ws.observations[-1].text)
        if "XPos" not in obs_json or "ZPos" not in obs_json:
            continue

        x_true = obs_json["XPos"]
        z_true = obs_json["ZPos"]
        # corrupt the planner's self-localization; metrics use *_true
        x, z_pos = noisy_pos(x_true, z_true)
        yaw = obs_json.get("Yaw", 0.0) % 360.0

        vf = ws.video_frames[-1]
        frame_raw = (np.frombuffer(vf.pixels, dtype=np.uint8)
                     .reshape(vf.height, vf.width, 3))
        frame = np.array(Image.fromarray(frame_raw).resize(
            (IMG_W, IMG_H), Image.LANCZOS))

        cyan_count = int(diamond_pixel_mask(frame).sum())
        diamond_visible, blob = diamond_blob_visible(frame)

        if diamond_visible and blob is not None and int(blob.get("area", 0)) >= MARK_MIN_PIXELS:
            diamond_confirm += 1
            steps_since_diamond_seen = 0
        else:
            diamond_confirm = 0
            steps_since_diamond_seen += 1
        if diamond_visible:
            steps_since_sighting = 0
        else:
            steps_since_sighting += 1

        # ── Goal forgetting ──
        if GOAL_MEMORY == "timer":
            # legacy: decay after N unseen steps (deletes static-landmark
            # knowledge merely because the agent turned away)
            if steps_since_diamond_seen >= DIAMOND_STALE_STEPS and (grid == DIAMOND).any():
                grid[grid == DIAMOND] = UNKNOWN
                if verbose:
                    print(f"  [STALE DIAMOND cleared at step {step}]")
        else:
            # evidence-based: clear only after COUNTEREVIDENCE_FRAMES frames
            # of looking straight at the marked cell, close, detector silent
            dcells = np.argwhere(grid == DIAMOND)
            if len(dcells) and not diamond_visible:
                mgz, mgx = int(dcells[0][0]), int(dcells[0][1])
                if looking_at_cell(x, z_pos, yaw, (mgx, mgz)):
                    goal_counterev += 1
                    if goal_counterev >= COUNTEREVIDENCE_FRAMES:
                        grid[grid == DIAMOND] = UNKNOWN
                        goal_events.append(("clear_evidence", step, (mgx, mgz)))
                        goal_counterev = 0
                else:
                    goal_counterev = 0
            else:
                goal_counterev = 0

        maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action, x, z_pos)
        mark_current_free(grid, x, z_pos)
        marked, mark_status = maybe_mark_diamond(
            grid, x, z_pos, yaw, frame, diamond_visible, blob,
            confirm_counter=diamond_confirm)
        if marked is not None:
            goal_events.append(("mark", step, marked))
            hint_cell = None                      # hard mark supersedes hint
        elif mark_status == "no_candidate":
            goal_events.append(("mark_fail_no_candidate", step))

        # ── Glimpse hint: keep sub-confirmation sightings as a soft target ──
        if (diamond_visible and blob is not None and not (grid == DIAMOND).any()
                and not blob_at_edge(blob, frame)):
            hc = estimate_sighting_cell(grid, x, z_pos, yaw, blob, frame,
                                        want_unknown=False)
            if hc is not None and hc != hint_cell:
                goal_events.append(("hint", step, hc))
            if hc is not None:
                hint_cell, hint_step = hc, step
        if hint_cell is not None and (step - hint_step > HINT_TTL_STEPS
                                      or hint_cell == world_to_grid(x, z_pos)):
            hint_cell = None                      # expired or reached: let go

        if prev_action == 0:
            if (prev_x is not None
                    and abs(x - prev_x) < 0.05
                    and abs(z_pos - prev_z) < 0.05):
                stuck_count += 1
            else:
                stuck_count = 0

        if (prev_x is not None
                and abs(x - prev_x) < 0.05
                and abs(z_pos - prev_z) < 0.05):
            same_pos_count += 1
        else:
            same_pos_count = 0

        if same_pos_count >= 6 and prev_action != 0:
            agent_host.sendCommand(ACTIONS_STR[0])
            same_pos_count = 0
            nsteps["spin_break"] += 1
            if verbose:
                print(f"  [SPIN BREAK: forced move at step {step}]")
            pos_list.append([x, z_pos])
            truepos_list.append([x_true, z_true])
            yaw_list.append(yaw)
            v_list.append(0.0)
            action_list.append(0)
            prev_x, prev_z, prev_yaw, prev_action = x, z_pos, yaw, 0
            time.sleep(0.25)
            continue

        if stuck_count >= STUCK_TURN_AFTER * 4 and (grid == DIAMOND).any():
            grid[grid == DIAMOND] = UNKNOWN
            stuck_count = 0
            if verbose:
                print(f"  [cleared stale DIAMOND marks at step {step}]")

        cur_grid = world_to_grid(x, z_pos)

        # ── 360° look-around: exploration covers cells, detection needs
        # ANGLES — sweep when nothing goal-related has been seen for a while
        have_goal_lead = (grid == DIAMOND).any() or hint_cell is not None
        if have_goal_lead:
            sweep_queue = []                      # lead found: abort sweep
        elif not sweep_queue and steps_since_sighting >= SWEEP_AFTER_STEPS:
            sweep_queue = [1, 1, 1, 1]            # four turnR = full circle
            steps_since_sighting = 0              # one sweep per drought
            goal_events.append(("sweep", step))

        if sweep_queue:
            action_idx = sweep_queue.pop(0)
            reason, v_pred, dist = "sweep (look-around)", 0.0, 0.0
        else:
            action_idx, reason, v_pred, dist = planned_action(
                grid, cur_grid, x, z_pos, yaw, frame,
                ae, dyn, value, stuck_count, prev_action,
                hint_cell=hint_cell
            )

        if reason.startswith("sweep"):
            nsteps["sweep"] += 1
        elif "override" in reason:
            nsteps["stuck_override"] += 1
        elif reason.startswith("MPC->DIAMOND"):
            nsteps["diamond"] += 1
        elif reason.startswith("MPC->HINT"):
            nsteps["hint"] += 1
        elif reason.startswith("MPC->frontier"):
            nsteps["frontier"] += 1
        elif reason.startswith("no_target"):
            nsteps["no_target"] += 1

        agent_host.sendCommand(ACTIONS_STR[action_idx])

        pos_list.append([x, z_pos])
        truepos_list.append([x_true, z_true])
        yaw_list.append(yaw)
        v_list.append(v_pred)
        action_list.append(action_idx)

        if verbose:
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

    # ── Compute stats ──
    steps = len(truepos_list)
    if steps > 0:
        final_x, final_z = truepos_list[-1]
        dist_to_diamond = abs(final_x - DIAMOND_POS[0]) + abs(final_z - DIAMOND_POS[1])
        success = dist_to_diamond < 2.0
        path_length = sum(
            np.sqrt((truepos_list[i][0] - truepos_list[i-1][0])**2
                    + (truepos_list[i][1] - truepos_list[i-1][1])**2)
            for i in range(1, steps)
        )
    else:
        final_x, final_z = sx, sz
        success = False
        path_length = 0.0
        dist_to_diamond = abs(sx - DIAMOND_POS[0]) + abs(sz - DIAMOND_POS[1])

    spawn_dist = abs(sx - DIAMOND_POS[0]) + abs(sz - DIAMOND_POS[1])

    if verbose:
        tag = "SUCCESS" if success else "FAIL"
        print(f"\n[{tag}] final=({final_x:.2f},{final_z:.2f})  "
              f"steps={steps}  path={path_length:.1f}  spawn_dist={spawn_dist:.1f}")

    if save:
        save_run(grid, pos_list, yaw_list, v_list, action_list)

    return {
        "success": success,
        "steps": steps,
        "path_length": path_length,
        "spawn": (sx, sz, syaw),
        "spawn_dist": spawn_dist,
        "final_pos": (final_x, final_z),
        "dist_to_diamond": dist_to_diamond,
        "step_kinds": dict(nsteps),
        "goal_events": goal_events[:300],
    }


# ============= main =============
def main():
    ae, dyn, value = load_models()
    print(f"MPC horizon={MPC_HORIZON}, sequences=3^{MPC_HORIZON}={NUM_ACTIONS**MPC_HORIZON}")

    agent_host = MalmoPython.AgentHost()
    result = run_episode(agent_host, ae, dyn, value)
    print(f"\nResult: {result}")


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