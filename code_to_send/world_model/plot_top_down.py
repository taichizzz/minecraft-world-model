"""
Top-down visualization of a closed-loop run in env3.

Renders the env3 floor (walls, obstacles, goal blocks) and overlays the
agent's path. Trail dots are colored by V_now so you can see exactly where
the value head fired. Yaw arrows show heading.

Usage:
    py -3.9 world_model/plot_top_down.py                    # plot latest
    py -3.9 world_model/plot_top_down.py path/to/run.npz    # specific file
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# env3.xml layout (X, Z) — y omitted for top-down view
ROOM_HALF = 5  # walls at x=±5, z=±5

OBSTACLES = [
    # (x, z, color, name)
    (-2, -1, "#1a1a1a",  "obs"),    # obsidian
    (-1,  2, "#1a1a1a",  "obs"),
    ( 2,  1, "#1a1a1a",  "obs"),
    (-3,  0, "#5d2a14",  "neth"),   # nether_brick
    ( 0,  3, "#5d2a14",  "neth"),
    ( 1, -2, "#fafafa",  "qtz"),    # quartz
    ( 3, -1, "#fafafa",  "qtz"),
    (-2,  3, "#e0d8aa",  "ends"),   # end_stone
    ( 3,  2, "#e0d8aa",  "ends"),
    (-1, -3, "#e0d8aa",  "ends"),
]

GOALS = [
    ( 4,  4, "DIAMOND",  "#5be0d2"),
    (-4,  4, "gold",     "#fff570"),
    ( 4, -4, "emerald",  "#50d250"),
]


def render(npz_path, out_path):
    data = np.load(npz_path)
    pos = data["positions"]   # (N, 2) of (x, z)
    yaw = data["yaw"]         # (N,)
    v = data["v"]             # (N,)
    actions = data["actions"]
    diamond_touched = bool(data["diamond_touched"][0]) if "diamond_touched" in data else None

    fig, ax = plt.subplots(figsize=(9, 9))

    # outer walls
    ax.add_patch(Rectangle(
        (-ROOM_HALF - 0.5, -ROOM_HALF - 0.5), 2 * ROOM_HALF + 1, 2 * ROOM_HALF + 1,
        facecolor="#f4f4f4", edgecolor="black", linewidth=3,
    ))

    # obstacles
    for x, z, color, name in OBSTACLES:
        ax.add_patch(Rectangle(
            (x - 0.5, z - 0.5), 1, 1,
            facecolor=color, edgecolor="black", linewidth=1,
        ))

    # goal blocks
    for x, z, name, color in GOALS:
        ax.add_patch(Rectangle(
            (x - 0.5, z - 0.5), 1, 1,
            facecolor=color, edgecolor="black", linewidth=2,
        ))
        ax.text(x, z, name, ha="center", va="center", fontsize=8, weight="bold")

    # connect dots (faint)
    ax.plot(pos[:, 0], pos[:, 1], color="gray", linewidth=0.6, alpha=0.5, zorder=4)

    # trail colored by V_now
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=v, cmap="viridis",
                    s=40, zorder=5, vmin=0, vmax=1, edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="V_now (P diamond visible)")

    # heading arrows
    # Minecraft yaw: 0=south(+Z), 90=west(-X), 180=north(-Z), 270=east(+X)
    # forward (dx, dz) = (-sin(yaw_rad), cos(yaw_rad))
    stride = max(1, len(pos) // 25)
    for i in range(0, len(pos), stride):
        rad = np.radians(yaw[i])
        dx = -np.sin(rad) * 0.45
        dz = np.cos(rad) * 0.45
        ax.arrow(pos[i, 0], pos[i, 1], dx, dz,
                 head_width=0.18, head_length=0.13,
                 fc="red", ec="red", zorder=6, length_includes_head=True)

    # start / end markers
    ax.plot(pos[0, 0], pos[0, 1], 'o', markersize=18,
            markerfacecolor="yellow", markeredgecolor="black",
            zorder=10, label="start")
    ax.plot(pos[-1, 0], pos[-1, 1], 's', markersize=18,
            markerfacecolor="red", markeredgecolor="black",
            zorder=10, label="end")

    ax.set_xlim(-ROOM_HALF - 1, ROOM_HALF + 1)
    ax.set_ylim(-ROOM_HALF - 1, ROOM_HALF + 1)
    ax.invert_yaxis()  # so -Z (north) is at top
    ax.set_aspect("equal")
    ax.set_xlabel("X (east →)")
    ax.set_ylabel("Z (south ↓)")

    title = f"Closed-loop top-down: {os.path.basename(npz_path)}\n"
    title += f"steps={len(pos)}  "
    if diamond_touched is not None:
        title += f"diamond_touched={diamond_touched}  "
    title += f"V_max={v.max():.2f}  V_mean={v.mean():.2f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        files = sorted(glob.glob("world_model_out/closed_loop/closed_loop_*.npz"))
        if not files:
            print("No closed_loop npz found in world_model_out/closed_loop/")
            sys.exit(1)
        npz_path = files[-1]
        print(f"Using latest: {npz_path}")

    out_path = npz_path.replace(".npz", "_topdown.png")
    render(npz_path, out_path)


if __name__ == "__main__":
    main()
