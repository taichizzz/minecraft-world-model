import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from model import AutoEncoder
from transition_dataset import TransitionDataset
from dynamics_model import DynamicsMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset"
AE_WEIGHTS = "ae.pth"   
LATENT_DIM = 128
BATCH = 128
EPOCHS = 20
LR = 1e-3

def main():
    # dataset
    print("Building dataset") #debugging
    ds = TransitionDataset(DATASET_DIR)
    print("Dataset size:", len(ds)) #debugging
    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=4, hidden=256).to(DEVICE)
    opt = optim.Adam(dyn.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        dyn.train()
        total = 0.0

        for img_t, a_t, img_tp1 in train_loader:
            img_t = img_t.to(DEVICE)
            img_tp1 = img_tp1.to(DEVICE)
            a_t = a_t.to(DEVICE)

            with torch.no_grad():
                z_t = ae.encoder(img_t)       # (B,128)
                z_tp1 = ae.encoder(img_tp1)   # (B,128)

            z_pred = dyn(z_t, a_t)
            loss = F.mse_loss(z_pred, z_tp1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        # val
        dyn.eval()
        vtotal = 0.0
        with torch.no_grad():
            for img_t, a_t, img_tp1 in val_loader:
                img_t = img_t.to(DEVICE)
                img_tp1 = img_tp1.to(DEVICE)
                a_t = a_t.to(DEVICE)
                z_t = ae.encoder(img_t)
                z_tp1 = ae.encoder(img_tp1)
                z_pred = dyn(z_t, a_t)
                vtotal += F.mse_loss(z_pred, z_tp1).item()

        print(f"Epoch {epoch:02d} train={total/len(train_loader):.6f} val={vtotal/len(val_loader):.6f}")

    torch.save(dyn.state_dict(), "dynamics.pth")
    print("Saved dynamics.pth")

if __name__ == "__main__":
    main()
