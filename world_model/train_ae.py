import torch
from torch.utils.data import DataLoader
from dataset import MinecraftDataset
from model import AutoEncoder
import torch.optim as optim
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MinecraftDataset("dataset")
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

model = AutoEncoder(latent_dim=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0.0

    for imgs in loader:
        imgs = imgs.to(DEVICE)

        recon, _ = model(imgs)

        # force recon to match input spatial size (62->64 etc.)
        if recon.shape[-2:] != imgs.shape[-2:]:
            recon = F.interpolate(recon, size=imgs.shape[-2:], mode="bilinear", align_corners=False)

        loss = F.mse_loss(recon, imgs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "ae.pth")
torch.save(model.encoder.state_dict(), "encoder.pth") 

print("Saved ae.pth")
print("Saved encoder.pth")
