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

class DynamicsTurningMLP(nn.Module):
    def __init__(self, latent_dim=128, num_actions=3, hidden=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        self.input_proj = nn.Linear(latent_dim + num_actions, hidden)
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.output = nn.Linear(hidden, latent_dim)

    def forward(self, z, a):
        a_onehot = F.one_hot(a, num_classes=self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=1)
        h = F.relu(self.input_proj(x))
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.output(h)