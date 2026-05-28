import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataset_dir="dataset", K=8):
        self.dataset_dir = dataset_dir
        self.K = K
        self.items = []  # list of (npz_path, t0)

        files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.npz")))
        if not files:
            raise RuntimeError(f"No episode_*.npz found in {dataset_dir}")

        for fp in files:
            data = np.load(fp)
            T = int(data["actions"].shape[0])
            if T >= (K + 1):
                # valid start indices: 0 .. T-(K+1)
                for t0 in range(0, T - (K + 1) + 1):
                    self.items.append((fp, t0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, t0 = self.items[idx]
        data = np.load(fp)
        obs = data["obs"]          
        actions = data["actions"]  

        # frames t0..t0+K  => (K+1,H,W,3)
        frames = obs[t0 : t0 + self.K + 1]
        # actions t0..t0+K-1 => (K,)
        acts = actions[t0 : t0 + self.K]

        # to torch: (K+1,3,H,W) float [0,1]
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()

        acts = torch.from_numpy(acts.astype(np.int64))  # (K,)
        return frames, acts