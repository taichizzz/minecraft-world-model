import torch
import torch.nn as nn

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=128, num_actions=4, action_embed_dim=16, hidden=256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, a_t):
        a = self.action_embed(a_t)   # (B, action_embed_dim)
        x = torch.cat([z_t, a], dim=1)
        return self.net(x)