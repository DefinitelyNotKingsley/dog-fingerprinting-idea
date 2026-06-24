import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
EMBED_DIM = 128
MARGIN = 0.3


class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype("float32")
        self.y = y
        self.dogs = sorted(set(y))
        self.by_dog = {d: np.where(y == d)[0] for d in self.dogs}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor_y = self.y[idx]
        pos_idx = np.random.choice(self.by_dog[anchor_y])

        neg_y = np.random.choice([d for d in self.dogs if d != anchor_y])
        neg_idx = np.random.choice(self.by_dog[neg_y])

        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.X[pos_idx]),
            torch.tensor(self.X[neg_idx]),
        )


class TripletMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def main():
    print("Device:", DEVICE)

    X = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy").astype("float32")
    y = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)

    print("X:", X.shape)
    print("dogs:", len(set(y)))

    ds = TripletDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = TripletMLP(X.shape[1], EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for a, p, n in loader:
            a = a.to(DEVICE)
            p = p.to(DEVICE)
            n = n.to(DEVICE)

            za = model(a)
            zp = model(p)
            zn = model(n)

            loss = loss_fn(za, zp, zn)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * a.size(0)

        print(f"Epoch {epoch}/{EPOCHS} loss={total / len(ds):.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "triplet_fusion3_mlp.pt")
    print("Saved triplet model.")


if __name__ == "__main__":
    main()
