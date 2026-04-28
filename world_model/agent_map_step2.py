"""
Step 2 of the hybrid (map + MPC) planner: A* navigation on the occupancy map.

Builds the same map as step 1 but the controller now plans deliberately:
  - if any DIAMOND cell exists on the map -> A* path to it
  - else -> A* to the nearest FRONTIER (unknown cell adjacent to a free cell)
           so the agent purposefully explores instead of wandering
  - convert next-cell direction to required yaw; turn if not aligned, else move

If A* can't find a path or the agent is stuck against a wall, fall back to
a small reactive escape (alternate turn/move) just like step 1.
"""
import os
import sys
import time
import json
import heapq
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
from detect_diamond import diamond_pixel_mask, MIN_PIXELS as CYAN_DATASET_THRESHOLD

# Lower threshold for marking the diamond on the map at run time. The dataset-
# label threshold (30) is calibrated for daytime training data; live renders
# can be much dimmer, so we accept fainter signals here. Any cyan signal
# >= this many pixels is treated as "diamond is roughly in this direction".
CYAN_MIN_PIXELS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MISSION_FILE = "missions/env3.xml"
AE_WEIGHTS = "aeturn3.pth"
VALUE_WEIGHTS = "value_head_2.pth"

LATENT_DIM = 128
HIDDEN = 256

ACTIONS_STR = ["move 1", "turn 0.2", "turn -0.2"]
ACTIONS_NAME = ["move", "turnR", "turnL"]
TURN_DIR_FALLBACK = 1   # if A* fails, default turn direction
STUCK_TURN_AFTER = 3
MAX_STEPS = 200
IMG_H, IMG_W = 64, 64

ROOM_HALF = 5
GRID_LO, GRID_HI = -ROOM_HALF, ROOM_HALF
GRID_SIZE = 2 * ROOM_HALF + 1

UNKNOWN, FREE, OBSTACLE, DIAMOND = 0, 1, 2, 3

OUT_DIR = "world_model_out/closed_loop"
os.makedirs(OUT_DIR, exist_ok=True)


# ============= grid helpers =============
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


# map updates
def mark_current_free(grid, x, z):
    gx, gz = world_to_grid(x, z)
    if grid[gz, gx] != DIAMOND:
        grid[gz, gx] = FREE

def maybe_mark_diamond(grid, x, z, yaw, diamond_visible):
    if not diamond_visible:
        return
    dx, dz = yaw_to_forward(yaw)
    farthest = None
    for k in range(1, ROOM_HALF * 2 + 2):
        tx, tz = x + dx * k, z + dz * k
        if abs(tx) > ROOM_HALF + 0.5 or abs(tz) > ROOM_HALF + 0.5:
            break
        tgx, tgz = world_to_grid(tx, tz)
        cell = grid[tgz, tgx]
        if cell == OBSTACLE:
            break
        if cell == UNKNOWN:
            farthest = (tgx, tgz)
    if farthest is not None:
        grid[farthest[1], farthest[0]] = DIAMOND

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
    if grid[tgz, tgx] not in (DIAMOND, FREE):
        grid[tgz, tgx] = OBSTACLE

# astar implementation
UNKNOWN_COST = 3   # penalty for stepping through an unverified cell


def a_star(grid, start, goal):
    """4-connected A* on the occupancy grid.

    Cost model:
      - OBSTACLE: blocked (never traversed)
      - FREE / DIAMOND: cost 1 (verified cells)
      - UNKNOWN: cost UNKNOWN_COST (allowed but penalized, so A* prefers
        going through verified-free cells whenever possible — keeps frontier
        exploration meaningful, but still lets the agent commit toward a
        DIAMOND candidate that requires crossing some unknowns)

    Returns the list of (gx, gz) cells from the cell after start through
    goal (inclusive), or None if no path exists.
    """
    if start == goal:
        return []
    h = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    open_heap = [(h(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))
        gx, gz = cur
        for dgx, dgz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ngx, ngz = gx + dgx, gz + dgz
            if not (0 <= ngx < GRID_SIZE and 0 <= ngz < GRID_SIZE):
                continue
            cell = grid[ngz, ngx]
            if cell == OBSTACLE:
                continue
            step_cost = 1 if cell in (FREE, DIAMOND) else UNKNOWN_COST
            tg = g + step_cost
            n = (ngx, ngz)
            if tg < g_score.get(n, float("inf")):
                g_score[n] = tg
                came_from[n] = cur
                heapq.heappush(open_heap, (tg + h(n, goal), tg, n))
    return None


def find_diamond_cell(grid):
    locs = np.argwhere(grid == DIAMOND)
    if len(locs) == 0:
        return None
    gz, gx = locs[0]   # any diamond cell
    return (int(gx), int(gz))


def find_frontier_cell(grid, start_gxgz):
    """Closest UNKNOWN cell with a FREE neighbor (BFS from start through
    free/unknown cells). Returns None if there are no frontiers."""
    visited = set([start_gxgz])
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
                # this is a frontier (since it has a FREE neighbor: cur)
                return n
            if cell in (FREE, DIAMOND):
                queue.append(n)
    return None


# ============= action choice =============
def direction_to_yaw(dgx, dgz):
    """+Z south -> 0; -X west -> 90; -Z north -> 180; +X east -> 270."""
    if dgz == 1:  return 0
    if dgx == -1: return 90
    if dgz == -1: return 180
    if dgx == 1:  return 270
    raise ValueError(f"non-unit step: {dgx},{dgz}")


def yaw_diff_signed(current, desired):
    """Smallest signed angle desired - current in [-180, 180]."""
    return (desired - current + 540) % 360 - 180


def planned_action(grid, current_gxgz, current_yaw, stuck_count, prev_action):
    """Pick an action using A* on the current map. Falls back to a reactive
    escape if A* can't make progress (no path / stuck against wall)."""
    # find a goal: diamond if known, else nearest frontier
    goal = find_diamond_cell(grid)
    target_label = "DIAMOND"
    if goal is None:
        goal = find_frontier_cell(grid, current_gxgz)
        target_label = "frontier"
    if goal is None:
        # nothing to plan toward; just turn to look around
        return TURN_DIR_FALLBACK, "no_goal"

    path = a_star(grid, current_gxgz, goal)
    if not path:
        # no path — likely surrounded by obstacles or goal unreachable
        if stuck_count >= STUCK_TURN_AFTER:
            return (0 if prev_action != 0 else TURN_DIR_FALLBACK), "stuck_no_path"
        return TURN_DIR_FALLBACK, "no_path"

    next_cell = path[0]
    dgx = next_cell[0] - current_gxgz[0]
    dgz = next_cell[1] - current_gxgz[1]
    desired_yaw = direction_to_yaw(dgx, dgz)
    diff = yaw_diff_signed(current_yaw, desired_yaw)

    if abs(diff) < 1.0:
        # aligned — but if our last few moves failed, try reactive escape
        if stuck_count >= STUCK_TURN_AFTER and prev_action == 0:
            return TURN_DIR_FALLBACK, f"stuck->turn (->{target_label})"
        return 0, f"move (->{target_label})"

    # turn toward the desired yaw
    if diff > 0:
        return 1, f"turnR (->{target_label}, want yaw={desired_yaw})"
    else:
        return 2, f"turnL (->{target_label}, want yaw={desired_yaw})"


# ============= main loop =============
def main():
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
    print("Mission running. Building map + A* navigating.\n")

    grid = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.uint8)
    pos_list, yaw_list, v_list, action_list = [], [], [], []
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

        cyan_count = int(diamond_pixel_mask(frame).sum())
        diamond_visible = cyan_count >= CYAN_MIN_PIXELS

        # update map
        maybe_mark_obstacle(grid, prev_x, prev_z, prev_yaw, prev_action, x, z_pos)
        mark_current_free(grid, x, z_pos)
        maybe_mark_diamond(grid, x, z_pos, yaw, diamond_visible)

        # stuck counter (only failed moves)
        if prev_action == 0:
            if (prev_x is not None
                    and abs(x - prev_x) < 0.05
                    and abs(z_pos - prev_z) < 0.05):
                stuck_count += 1
            else:
                stuck_count = 0

        cur_grid = world_to_grid(x, z_pos)
        action_idx, reason = planned_action(
            grid, cur_grid, yaw, stuck_count, prev_action
        )
        agent_host.sendCommand(ACTIONS_STR[action_idx])

        pos_list.append([x, z_pos])
        yaw_list.append(yaw)
        v_list.append(v_now)
        action_list.append(action_idx)

        marker = ""
        if diamond_visible:
            marker += "  cyan↑"
        if stuck_count >= STUCK_TURN_AFTER:
            marker += "  STUCK"
        print(f"step {step:03d} | a={action_idx} ({ACTIONS_NAME[action_idx]:<5}) "
              f"| {reason:<28} | cyan={cyan_count:4d} "
              f"| pos=({x:5.2f},{z_pos:5.2f}) yaw={yaw:5.1f}{marker}")

        prev_x, prev_z, prev_yaw, prev_action = x, z_pos, yaw, action_idx
        time.sleep(0.25)

    print(f"\nfinal pos=({pos_list[-1][0]:.2f},{pos_list[-1][1]:.2f})  "
          f"steps={len(pos_list)}")

    save_run(grid, pos_list, yaw_list, v_list, action_list)


# ============= visualization (same as step 1) =============
def save_run(grid, positions, yaws, values, actions):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(OUT_DIR, f"map_step2_{ts}.npz")
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
    axes[0].set_title("Belief map (A* nav)\n"
                      "gray=unknown, white=free, black=obstacle, cyan=diamond_seen",
                      fontsize=10)

    OBSTACLES = [(-2, -1), (-1, 2), (2, 1), (-3, 0), (0, 3),
                 (1, -2), (3, -1), (-2, 3), (3, 2), (-1, -3)]
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
    out_png = os.path.join(OUT_DIR, f"map_step2_{ts}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved map: {out_png}")


if __name__ == "__main__":
    main()
