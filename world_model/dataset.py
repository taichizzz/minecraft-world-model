import numpy as np
import torch
from torch.utils.data import Dataset
import glob

#just testing
class MinecraftDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted(glob.glob(f"{dataset_dir}/episode_*.npz"))
        self.frames = []

        for f in self.files:
            data = np.load(f)
            obs = data["obs"]  #(T, 64, 64, 3)
            self.frames.append(obs)

        self.frames = np.concatenate(self.frames, axis=0)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).contiguous()  # (3, 64, 64)
        return img
