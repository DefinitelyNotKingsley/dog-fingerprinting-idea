from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
EMBED_DIM = 256
SCALE = 30.0
MARGIN = 0.5


class FusionMLP(nn.Module):
    def __init__(self, input_dim=848, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, W).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return logits * self.scale


def standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std, mean, std


def main():
    print("Device:", DEVICE)

    X_train = np.load(DATA_DIR / "fusion_train.npy").astype("float32")
    X_test = np.load(DATA_DIR / "fusion_test.npy").astype("float32")
    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    X_train, X_test, mean, std = standardize(X_train, X_test)

    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y_test)

    num_classes = len(le.classes_)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    print("Dogs/classes:", num_classes)

    class_counts = np.bincount(y_train_int)
    sample_weights = 1.0 / class_counts[y_train_int]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train_int, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    model = FusionMLP(input_dim=X_train.shape[1], embed_dim=EMBED_DIM).to(DEVICE)
    head = ArcFaceHead(
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
        scale=SCALE,
        margin=MARGIN,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            emb = model(xb)
            logits = head(emb, yb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={avg_loss:.4f}")

    model.eval()

    with torch.no_grad():
        train_emb = model(torch.tensor(X_train).to(DEVICE)).cpu().numpy()
        test_emb = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

    torch.save(model.state_dict(), MODEL_DIR / "fusion_mlp_arcface.pt")
    torch.save(head.state_dict(), MODEL_DIR / "arcface_head.pt")

    np.save(RESULTS_DIR / "arcface_fusion_train_embeddings.npy", train_emb)
    np.save(RESULTS_DIR / "arcface_fusion_test_embeddings.npy", test_emb)
    np.save(RESULTS_DIR / "label_classes.npy", le.classes_)
    np.save(RESULTS_DIR / "feature_mean.npy", mean)
    np.save(RESULTS_DIR / "feature_std.npy", std)

    print("Saved model and embeddings.")
    print("Train embeddings:", train_emb.shape)
    print("Test embeddings:", test_emb.shape)


if __name__ == "__main__":
    main()
