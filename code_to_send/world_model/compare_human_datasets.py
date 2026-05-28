"""
Quick visual comparison: pull one frame from each human dataset so we can
see if they were collected in different environments.

Saves a 4-panel PNG so we can eyeball whether dataset_1_human and
dataset_2_human were recorded in different rooms than dataset_3_human*.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

DATASETS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human1",
    "dataset/dataset_3_human2",
]
OUT = "world_model_out/dataset_compare.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, d in enumerate(DATASETS):
    full = os.path.join(ROOT, d)
    files = sorted(glob.glob(os.path.join(full, "episode_*.npz")))
    if not files:
        continue

    # frame 0 of episode 0 (= spawn frame)
    e0 = np.load(files[0])
    obs0 = e0["obs"]
    axes[0, col].imshow(obs0[0])
    axes[0, col].set_title(f"{os.path.basename(d)}\nep0 frame0  pos={e0['positions'][0]}")
    axes[0, col].axis("off")

    # a middle frame from a later episode (= mid-trajectory in some room)
    eN = np.load(files[len(files) // 2])
    midt = eN["obs"].shape[0] // 2
    axes[1, col].imshow(eN["obs"][midt])
    axes[1, col].set_title(f"ep{len(files)//2} frame{midt}  pos={eN['positions'][midt]}")
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig(OUT, dpi=100)
print(f"saved: {OUT}")
