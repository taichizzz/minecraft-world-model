import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class TransitionDataset(Dataset):
    def __init__(self, dataset_dir="dataset"):
        self.episodes = []
        self.items = []

        files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.npz")))
        if len(files) == 0:
            raise RuntimeError(f"No .npz found in {dataset_dir}")

        for fp in files:
            data = np.load(fp)
            obs = data["obs"]
            actions = data["actions"]

            episode_idx = len(self.episodes)
            self.episodes.append((obs, actions))

            T = actions.shape[0]
            for t in range(T - 1):
                self.items.append((episode_idx, t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ep_idx, t = self.items[idx]
        obs, actions = self.episodes[ep_idx]

        img_t = obs[t]
        img_tp1 = obs[t+1]
        a_t = int(actions[t])

        img_t = torch.from_numpy(img_t).float().permute(2,0,1) / 255.0
        img_tp1 = torch.from_numpy(img_tp1).float().permute(2,0,1) / 255.0
        a_t = torch.tensor(a_t, dtype=torch.long)

        return img_t, a_t, img_tp1