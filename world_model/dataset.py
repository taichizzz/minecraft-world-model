import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os

#just testing
class MinecraftTransitions(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted(glob.glob(f"{dataset_dir}/episode_*.npz"))

        if len(self.files) == 0:
            raise RuntimeError("No episode_*.npz files found")

        self.transitions = []  # list of (file_index, timestep)

        # Build transition index
        for fi, file in enumerate(self.files):
            data = np.load(file)
            obs = data["obs"]
            actions = data["actions"]
            done = data["done"]

            T = len(actions)

            for t in range(T - 1):
                if done[t] == 1:
                    continue
                self.transitions.append((fi, t))

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        fi, t = self.transitions[idx]

        data = np.load(self.files[fi])

        obs = data["obs"]
        actions = data["actions"]

        frame_t = obs[t]
        frame_tp1 = obs[t + 1]
        action_t = actions[t]

        # Convert to torch
        frame_t = torch.from_numpy(frame_t).float() / 255.0
        frame_tp1 = torch.from_numpy(frame_tp1).float() / 255.0

        frame_t = frame_t.permute(2, 0, 1).contiguous()
        frame_tp1 = frame_tp1.permute(2, 0, 1).contiguous()

        action_t = torch.tensor(action_t, dtype=torch.long)

        return frame_t, action_t, frame_tp1