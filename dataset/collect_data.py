import MalmoPython
import time
import json
import random
import numpy as np
import os

# ================= CONFIG =================
MISSION_FILE = "missions/simple_nav.xml"
SAVE_DIR = "dataset"
EPISODES = 200
MAX_STEPS = 200
IMG_W = 64
IMG_H = 64

# 8-way movement (Option A): diagonals are two commands in one step
ACTIONS = [
    ["movenorth 1"],                  # 0: N
    ["movesouth 1"],                  # 1: S
    ["moveeast 1"],                   # 2: E
    ["movewest 1"],                   # 3: W
    ["movenorth 1", "moveeast 1"],    # 4: NE
    ["movenorth 1", "movewest 1"],    # 5: NW
    ["movesouth 1", "moveeast 1"],    # 6: SE
    ["movesouth 1", "movewest 1"],    # 7: SW
]

# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)

agent_host = MalmoPython.AgentHost()

for episode in range(EPISODES):
    print(f"\n=== Episode {episode} ===")

    # Load mission XML
    with open(MISSION_FILE, "r") as f:
        mission_xml = f.read()

    mission = MalmoPython.MissionSpec(mission_xml, True)
    record = MalmoPython.MissionRecordSpec()

    # Ensure no mission is still running
    ws = agent_host.getWorldState()
    while ws.is_mission_running:
        t0 = time.time()
        while time.time() - t0 < 2.0:
            ws = agent_host.getWorldState()
            if not ws.is_mission_running:
                time.sleep(0.1)
            else:
                time.sleep(0.1)

    # Start mission safely
    max_retries = 5
    for retry in range(max_retries):
        try:
            agent_host.startMission(mission, record)
            break
        except MalmoPython.MissionException as e:
            if retry == max_retries - 1:
                raise
            print("Retrying startMission:", e)
            time.sleep(2)

    # Wait for mission start
    print("Waiting for mission to begin...")
    ws = agent_host.getWorldState()
    while not ws.has_mission_begun:
        time.sleep(0.1)
        ws = agent_host.getWorldState()
        for err in ws.errors:
            print("Mission error:", err.text)

    # Wait for first valid observation + frame
    while True:
        ws = agent_host.getWorldState()
        if not ws.is_mission_running:
            break
        if (
            len(ws.observations) > 0
            and ws.observations[-1].text != "{}"
            and len(ws.video_frames) > 0
        ):
            break
        time.sleep(0.05)

    obs_list = []
    action_list = []
    pos_list = []
    done_list = []

    # ================= MAIN LOOP =================
    for step in range(MAX_STEPS):
        ws = agent_host.getWorldState()

        if not ws.is_mission_running:
            break

        if len(ws.observations) == 0 or ws.observations[-1].text == "{}":
            time.sleep(0.05)
            continue

        if len(ws.video_frames) == 0:
            time.sleep(0.05)
            continue

        obs = json.loads(ws.observations[-1].text)

        if "XPos" not in obs or "ZPos" not in obs:
            continue

        x, z = obs["XPos"], obs["ZPos"]

        frame = ws.video_frames[-1].pixels
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = frame.reshape(IMG_H, IMG_W, 3)

        # ----- choose one of 8 actions -----
        action_idx = random.randrange(len(ACTIONS))
        for cmd in ACTIONS[action_idx]:
            agent_host.sendCommand(cmd)
            # Optional tiny delay if you feel diagonals sometimes "miss" one command:
            # time.sleep(0.01)

        obs_list.append(frame)
        action_list.append(action_idx)
        pos_list.append([x, z])
        done_list.append(0)

        time.sleep(0.05)

    if len(done_list) > 0:
        done_list[-1] = 1
    else:
        print("Warning: episode ended with no steps collected")

    np.savez(
        f"{SAVE_DIR}/episode_{episode:06d}.npz",
        obs=np.array(obs_list, dtype=np.uint8),
        actions=np.array(action_list, dtype=np.int64),
        positions=np.array(pos_list, dtype=np.float32),
        done=np.array(done_list, dtype=np.uint8),
    )

    print(f"Saved episode {episode}, steps={len(obs_list)}")

    time.sleep(1)

print("\n=== DATA COLLECTION DONE ===")
