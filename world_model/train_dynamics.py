import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

from model import AutoEncoder
from transition_dataset import TransitionDataset
from dynamics_model import DynamicsMLP
from torch.utils.data import ConcatDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_DIR = "dataset"
AE_WEIGHTS = "ae3.pth"   
LATENT_DIM = 128
BATCH = 128
EPOCHS = 80
LR = 1e-3
GRAD_CLIP = 1.0
LOG_EVERY = 200

ds1 = TransitionDataset("dataset/dataset1")
ds2 = TransitionDataset("dataset/dataset2")

print("Transitions ds1:", len(ds1))
print("Transitions ds2:", len(ds2))

def split_dataset(ds, val_fraction=0.1):
    n = len(ds)
    n_val = max(1, int(val_fraction * n))
    n_train = n - n_val
    return random_split(ds, [n_train, n_val])

train1, val1 = split_dataset(ds1)
train2, val2 = split_dataset(ds2)

train_ds = ConcatDataset([train1, train2])
val_ds   = ConcatDataset([val1, val2])



def main():
    # dataset
    print("Building dataset") #debugging
    # ds = TransitionDataset(DATASET_DIR)
    # ds = ConcatDataset([train1, train2])
    # print("Dataset size:", len(ds)) #debugging
    # n = len(ds)
    # n_val = max(1, int(0.1 * n))
    # n_train = n - n_val
    # train_ds, val_ds = random_split(ds, [n_train, n_val])

    # train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    # val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    loader1 = DataLoader(train1, batch_size=BATCH//2, shuffle=True, drop_last=True)
    loader2 = DataLoader(train2, batch_size=BATCH//2, shuffle=True, drop_last=True)

    val_loader1 = DataLoader(val1, batch_size=BATCH//2, shuffle=False, drop_last=True)
    val_loader2 = DataLoader(val2, batch_size=BATCH//2, shuffle=False, drop_last=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, drop_last=True)

    ae = AutoEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=DEVICE))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    dyn = DynamicsMLP(latent_dim=LATENT_DIM, num_actions=4, hidden=256).to(DEVICE)
    opt = optim.Adam(dyn.parameters(), lr=LR)

    train_losses = []
    val_losses = []



    for epoch in range(EPOCHS):
        dyn.train()
        total = 0.0
        batches = 0

        # for img_t, a_t, img_tp1 in train_loader:
        #     img_t = img_t.to(DEVICE)
        #     img_tp1 = img_tp1.to(DEVICE)
        #     a_t = a_t.to(DEVICE)

        # since we separate two datasets
        for batch_i, (img_t, a_t, img_tp1) in enumerate(train_loader):
            img_t = img_t.to(DEVICE)
            img_tp1 = img_tp1.to(DEVICE)
            a_t = a_t.to(DEVICE)

            with torch.no_grad():
                z_t = ae.encoder(img_t)       # (B,128)
                z_tp1 = ae.encoder(img_tp1)   

                

            # original
            z_pred = dyn(z_t, a_t)   
            loss = F.mse_loss(z_pred, z_tp1)

            # try predicting the small change
            # delta = dyn(z_t, a_t)
            # z_pred = z_t + delta   
            # loss = F.mse_loss(z_pred, z_tp1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn.parameters(), GRAD_CLIP)
            opt.step()

            total += loss.item()
            batches += 1

            if batch_i % LOG_EVERY == 0:
                with torch.no_grad():
                    delta_true = (z_tp1 - z_t).norm(dim=1).mean()
                    delta_pred = (z_pred - z_t).norm(dim=1).mean()
                print(f"epoch {epoch:02d} batch {batch_i:04d} loss {loss.item():.6f}  Δtrue {delta_true.item():.4f}  Δpred {delta_pred.item():.4f}")

        train_loss = total / batches
        train_losses.append(train_loss)

        # val
        dyn.eval()
        vtotal = 0.0
        vbatches = 0
        with torch.no_grad():
            # for img_t, a_t, img_tp1 in val_loader:
            #     img_t = img_t.to(DEVICE)
            #     img_tp1 = img_tp1.to(DEVICE)
            #     a_t = a_t.to(DEVICE)

            for img_t, a_t, img_tp1 in val_loader:

                img_t = img_t.to(DEVICE)
                img_tp1 = img_tp1.to(DEVICE)
                a_t = a_t.to(DEVICE)
                z_t = ae.encoder(img_t)
                z_tp1 = ae.encoder(img_tp1)

                # original
                z_pred = dyn(z_t, a_t)

                # try predicting the small change
                # delta = dyn(z_t, a_t)
                # z_pred = z_t + delta

                vtotal += F.mse_loss(z_pred, z_tp1).item()
                vbatches += 1
                

        # print(f"Epoch {epoch:02d} train={total/len(train_loader):.6f} val={vtotal/len(val_loader):.6f}")

        # train_loss = total / len(train_loader)
        # val_loss = vtotal / len(val_loader)


        val_loss = vtotal / vbatches
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Dynamics Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("world_model_out/dynamics_loss_balanced2.png")
    plt.close()

    print("Saved loss curve to world_model_out/dynamics_loss_balanced2.png")

    torch.save(dyn.state_dict(), "dynamics_balanced2.pth")
    print("saved dynamics_balanced2.pth")

if __name__ == "__main__":
    main()

