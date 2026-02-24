import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsMLP(nn.Module):
    def __init__(self, latent_dim=128, num_actions=4, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, a):
        """
        z: (B, latent_dim)
        a: (B,) int64 actions
        """
        a_onehot = F.one_hot(a, num_classes=self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=1)
        return self.net(x)
