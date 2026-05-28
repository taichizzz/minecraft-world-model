"""
Closed-loop test of the value-head planner against the live Malmo env.

Mirrors the data-collection script's Malmo setup (same mission, same action
mapping, same sleep timings) but replaces random action sampling with MPC:
at each step we encode the current real frame, sample NUM_SAMPLES random
action sequences of length PLAN_HORIZON, roll each one out in latent space
with the dynamics model, and send the first action of whichever sequence
reaches the highest V(z_final) — i.e. is most likely to bring the diamond
into view.

Stops when the env terminates (diamond_block touched / time up) or after
MAX_STEPS planner decisions.

Run with Minecraft already launched and listening on the default port:
    py -3.9 world_model/run_planner_malmo.py
"""
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# so `from model import ...` works when run as `world_model/run_planner_malmo.py`
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# locate MalmoPython.pyd produced by the malmo build
ROOT = os.path.dirname(HERE)  # one level up from world_model/
_MALMO_PYD_DIRS = [
    os.path.join(ROOT, "build", "Malmo", "src", "PythonWrapper", "Release"),
    os.path.join(ROOT, "build", "install", "Python_Examples"),
]
for _d in _MALMO_PYD_DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

# MalmoPython needs MALMO_XSD_PATH set to the dir containing Mission.xsd etc.
if not os.environ.get("MALMO_XSD_PATH"):
    _schemas = os.path.join(ROOT, "Schemas")
    if os.path.isfile(os.path.join(_schemas, "Mission.xsd")):
        os.environ["MALMO_XSD_PATH"] = _schemas

import MalmoPython

from model import AutoEncoder
from dynamics_model import DynamicsTurningMLP
from train_value_head import ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= CONFIG =============
MISSION_FILE = "missions/env3.xml"   # same world the models trained on

AE_WEIGHTS = "aeturn3.pth"
DYN_WEIGHTS = "dynamics_turn_3.pth"
VALUE_WEIGHTS = "value_head_2.pth"

LATENT_DIM = 128
NUM_ACTIONS = 3
HIDDEN = 256

# action mapping MUST match the data-collection script exactly
ACTIONS_STR = [
    "move 1",      # 0 = move forward
    "turn 0.2",    # 1 = small turn RIGHT
    "turn -0.2",   # 2 = small turn LEFT
]
ACTIONS_NAME = ["move", "turnR", "turnL"]

PLAN_HORIZON = 4    # short horizon: dyn drift over 16 steps was hallucinating V=1
NUM_SAMPLES = 512
MAX_STEPS = 200
ACTION_WEIGHTS = [4.0, 3.0, 3.0]   # same prior as in training data

IMG_H, IMG_W = 64, 64

OUT_DIR = "world_model_out/closed_loop"
os.makedirs(OUT_DIR, exist_ok=True)


def to_torch_img(frame_uint8):
    """H,W,3 uint8 -> 1,3,H,W float32 in [0,1]"""
    return (torch.from_numpy(frame_uint8).float().div_(255.0)
            .permute(2, 0, 1).unsqueeze(0))


def plan_one_action(z_current, dyn, value, horizon, num_samples):
    weights = torch.tensor(ACTION_WEIGHTS, device=DEVICE)
    flat = torch.multinomial(weights, num_samples * horizon, replacement=True)
    actions = flat.view(num_samples, horizon).long()
    z = z_current.unsqueeze(0).expand(num_samples, -1).contiguous()
    with torch.no_grad():
        for t in range(horizon):
            z = dyn(z, actions[:, t])
        v_final = value(z)
    best = v_final.argmax().item()
    return actions[best, 0].item(), v_final[best].item()


def main():
    # ----- load models -----
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

    # ----- connect to Malmo -----
    agent_host = MalmoPython.AgentHost()
    with open(MISSION_FILE, "r", encoding="utf-8") as f:
        mission_xml = f.read()
    mission = MalmoPython.MissionSpec(mission_xml, True)
    record = MalmoPython.MissionRecordSpec()

    # wait for any leftover mission to die
    ws = agent_host.getWorldState()
    while ws.is_mission_running:
        time.sleep(0.1)
        ws = agent_host.getWorldState()

    started = False
    for retry in range(5):
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
        for err in ws.errors:
            print("err:", err.text)

    # wait for first obs + frame
    while True:
        ws = agent_host.getWorldState()
        if not ws.is_mission_running:
            break
        if (len(ws.observations) > 0 and ws.observations[-1].text != "{}"
                and len(ws.video_frames) > 0):
            break
        time.sleep(0.05)

    print("Mission running. Starting closed-loop control.\n")

    obs_list, action_list, v_list, vpred_list = [], [], [], []
    pos_list, yaw_list = [], []
    final_step = -1

    for step in range(MAX_STEPS):
        ws = agent_host.getWorldState()
        if not ws.is_mission_running:
            print(f"Mission ended at step {step}")
            break
        if len(ws.observations) == 0 or ws.observations[-1].text == "{}":
            time.sleep(0.05)
            continue
        if len(ws.video_frames) == 0:
            time.sleep(0.05)
            continue

        obs_json = json.loads(ws.observations[-1].text)
        if "XPos" not in obs_json or "ZPos" not in obs_json:
            continue
        x = obs_json["XPos"]
        z_pos = obs_json["ZPos"]
        yaw = obs_json.get("Yaw", 0.0)

        frame = np.frombuffer(ws.video_frames[-1].pixels, dtype=np.uint8)
        frame = frame.reshape(IMG_H, IMG_W, 3).copy()

        # encode + V on the real frame
        with torch.no_grad():
            img = to_torch_img(frame).to(DEVICE)
            z_t = ae.encoder(img).squeeze(0)
            v_now = value(z_t.unsqueeze(0)).item()

        # MPC: pick best first action
        action_idx, v_pred = plan_one_action(
            z_t, dyn, value, PLAN_HORIZON, NUM_SAMPLES
        )
        cmd = ACTIONS_STR[action_idx]

        obs_list.append(frame)
        action_list.append(action_idx)
        v_list.append(v_now)
        vpred_list.append(v_pred)
        pos_list.append([x, z_pos])
        yaw_list.append(yaw)
        final_step = step

        print(f"step {step:03d} | a={action_idx} ({ACTIONS_NAME[action_idx]:<5}) "
              f"| V_now={v_now:.3f}  V@H={v_pred:.3f} "
              f"| pos=({x:.2f},{z_pos:.2f}) yaw={yaw:.1f}")

        agent_host.sendCommand(cmd)
        time.sleep(0.2)   # let the action register
        time.sleep(0.05)  # match collection script timing

    # diamond_block sits at (4, 227, 4) in env3.xml. The agent quits on touch,
    # so if the final position is near (4, 4) we treat that as a real touch;
    # otherwise the mission ended by timeout.
    if pos_list:
        fx, fz = pos_list[-1]
        diamond_touched = bool(abs(fx - 4.0) < 1.5 and abs(fz - 4.0) < 1.5)
    else:
        diamond_touched = False
    print(f"\nDiamond touched: {diamond_touched}  (final pos = {pos_list[-1] if pos_list else 'n/a'})")
    print(f"Total steps:     {len(obs_list)}")

    if not obs_list:
        print("No frames; nothing to save.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(OUT_DIR, f"closed_loop_{ts}.npz")
    np.savez(
        out_npz,
        obs=np.array(obs_list, dtype=np.uint8),
        actions=np.array(action_list, dtype=np.int64),
        v=np.array(v_list, dtype=np.float32),
        v_pred=np.array(vpred_list, dtype=np.float32),
        positions=np.array(pos_list, dtype=np.float32),
        yaw=np.array(yaw_list, dtype=np.float32),
        diamond_touched=np.array([diamond_touched], dtype=np.uint8),
    )
    print(f"Saved npz:    {out_npz}")

    # V curves
    plt.figure(figsize=(7, 4))
    plt.plot(v_list, marker="o", label="V_now (real obs)")
    plt.plot(vpred_list, marker="x", linestyle="--", label="V_pred @ horizon")
    plt.xlabel("Control step")
    plt.ylabel("V(z)")
    plt.title(f"Closed-loop V over time | diamond_touched={diamond_touched}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out_v = os.path.join(OUT_DIR, f"closed_loop_v_{ts}.png")
    plt.savefig(out_v, dpi=150)
    plt.close()
    print(f"Saved V curve: {out_v}")

    # frames sampled across the trajectory
    n = len(obs_list)
    stride = max(1, n // 16)
    sampled = list(range(0, n, stride))[:16]
    cols = 4
    rows = (len(sampled) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, idx in enumerate(sampled):
        axes[i].imshow(obs_list[idx])
        axes[i].axis("off")
        axes[i].set_title(
            f"t={idx} {ACTIONS_NAME[action_list[idx]]}\nV={v_list[idx]:.2f}",
            fontsize=7,
        )
    for j in range(len(sampled), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    out_img = os.path.join(OUT_DIR, f"closed_loop_frames_{ts}.png")
    plt.savefig(out_img, dpi=120)
    plt.close()
    print(f"Saved frames:  {out_img}")


if __name__ == "__main__":
    main()
