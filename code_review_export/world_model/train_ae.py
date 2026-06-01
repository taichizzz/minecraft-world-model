import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import MinecraftImages
from model import AutoEncoder
from transition_dataset import TransitionDataset
import torch.optim as optim
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 0.01 

dataset1 = TransitionDataset("dataset/dataset1")
dataset2 = TransitionDataset("dataset/dataset2")
dataset3 = TransitionDataset("dataset/dataset3")
dataset6 = TransitionDataset("dataset/dataset6")
dataset2_turn = TransitionDataset("dataset/dataset2_turn")
dataset3_turn = TransitionDataset("dataset/dataset3_turn")
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset6, dataset2_turn, dataset3_turn])

print("dataset1 frames:", len(dataset1))
print("dataset2 frames:", len(dataset2))
print("dataset3 frames:", len(dataset3))
print("dataset6 frames:", len(dataset6))
print("dataset2_turn frames:", len(dataset2_turn))
print("dataset3_turn frames:", len(dataset3_turn))
print("combined frames:", len(combined_dataset))


loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, drop_last=True)

model = AutoEncoder(latent_dim=128).to(DEVICE)
model.load_state_dict(torch.load("aeturn1.pth", map_location=DEVICE))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("Starting training...")
print("Using device:", DEVICE)
for epoch in range(80):
    print("Epoch", epoch, "started")
    model.train()
    total_loss = 0.0

    # for batch in loader:
    # # batch could be a Tensor, or (Tensor,), or [Tensor]
    #     if isinstance(batch, (list, tuple)):
    #         imgs = batch[0]
    #     else:
    #         imgs = batch

    #     imgs = imgs.to(DEVICE)
    #     recon, _ = model(imgs)

    #     # force recon to match input spatial size (62->64 etc.)
    #     if recon.shape[-2:] != imgs.shape[-2:]:
    #         recon = F.interpolate(recon, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
    #     loss = F.mse_loss(recon, imgs)

    for img_t, _, img_tp1 in loader:
        img_t = img_t.to(DEVICE)
        img_tp1 = img_tp1.to(DEVICE)

        recon_t, z_t = model(img_t)
        recon_tp1, z_tp1 = model(img_tp1)

        recon_loss = F.mse_loss(recon_t, img_t) + F.mse_loss(recon_tp1, img_tp1)
        # loss = F.mse_loss(recon, imgs)
        smooth_loss = F.mse_loss(z_tp1, z_t)

        loss = recon_loss + BETA * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "aeturn2.pth")
torch.save(model.encoder.state_dict(), "encoderturn2.pth") 

print("Saved aeturn2.pth")
print("Saved encoderturn2.pth")