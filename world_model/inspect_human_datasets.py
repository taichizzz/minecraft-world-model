"""
Quick diagnostic to figure out what's in each dataset_*_human* folder.

Reports per dataset:
- number of episodes
- avg episode length
- keys stored in each npz
- if positions are available: unique starting (x, z, yaw) — tells us if
  the dataset has fixed-spawn or random-spawn episodes
- action distribution
"""
import os
import glob
import numpy as np
from collections import Counter

DATASETS = [
    "dataset/dataset_1_human",
    "dataset/dataset_2_human",
    "dataset/dataset_3_human1",
    "dataset/dataset_3_human2",
]


def inspect(d):
    files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
    print(f"\n=== {d}  ({len(files)} episodes) ===")
    if not files:
        return

    # peek at the first file to see what keys exist
    f0 = np.load(files[0])
    print(f"keys: {list(f0.keys())}")
    print(f"obs shape: {f0['obs'].shape if 'obs' in f0 else 'N/A'}")
    if 'actions' in f0:
        print(f"actions shape: {f0['actions'].shape}")
    f0.close()

    # iterate all files
    lengths = []
    start_positions = []
    all_actions = Counter()
    for fpath in files:
        data = np.load(fpath)
        if 'obs' in data:
            lengths.append(data['obs'].shape[0])
        # try to grab the spawn position from various possible keys
        for k in ('positions', 'pos', 'xz', 'state'):
            if k in data:
                arr = data[k]
                if arr.ndim >= 2 and arr.shape[0] > 0:
                    start_positions.append(tuple(arr[0]))
                break
        if 'actions' in data:
            for a in data['actions']:
                all_actions[int(a)] += 1
        data.close()

    print(f"episode length: min={min(lengths)}  max={max(lengths)}  "
          f"mean={np.mean(lengths):.1f}")

    if start_positions:
        unique_starts = set(tuple(np.round(p, 2)) for p in start_positions)
        print(f"unique start positions: {len(unique_starts)}")
        if len(unique_starts) <= 12:
            for p in sorted(unique_starts):
                print(f"  {p}")
        else:
            sample = list(sorted(unique_starts))[:6]
            print(f"  first 6: {sample}")
            print(f"  ...")
    else:
        print("no position info stored in npz")

    if all_actions:
        total = sum(all_actions.values())
        print("action distribution:")
        for a, c in sorted(all_actions.items()):
            print(f"  action {a}: {c}  ({100*c/total:.1f}%)")


if __name__ == "__main__":
    for d in DATASETS:
        full = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
        if os.path.isdir(full):
            inspect(d)
        else:
            print(f"\n[skip] {d} does not exist")
