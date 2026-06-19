from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

D2V_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/dog2vec")
MFCC_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/mfcc")
ECAPA_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/ecapa")

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


def fix_labels(labels):
    return np.array(["dog_7" if x in ["dog_7a", "dog_7b"] else x for x in labels])


def load_feature(base, split):
    X = np.load(base / f"{split}_embeddings.npy").astype("float32")
    y = np.load(base / f"{split}_labels.npy", allow_pickle=True)
    return X, y


def standardize(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


class AttentionFusion(nn.Module):
    def __init__(self, d2v_dim=768, mfcc_dim=80, ecapa_dim=192, hidden=256, embed_dim=256):
        super().__init__()

        self.d2v_proj = nn.Linear(d2v_dim, hidden)
        self.mfcc_proj = nn.Linear(mfcc_dim, hidden)
        self.ecapa_proj = nn.Linear(ecapa_dim, hidden)

        self.attn = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.out = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, embed_dim)
        )

    def forward(self, d2v, mfcc, ecapa):
        h1 = F.relu(self.d2v_proj(d2v))
        h2 = F.relu(self.mfcc_proj(mfcc))
        h3 = F.relu(self.ecapa_proj(ecapa))

        stacked = torch.stack([h1, h2, h3], dim=1)
        scores = self.attn(stacked).squeeze(-1)
        weights = F.softmax(scores, dim=1)

        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        emb = self.out(fused)
        return F.normalize(emb, p=2, dim=1), weights


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


def main():
    print("Device:", DEVICE)

    d2v_train, y_train = load_feature(D2V_DIR, "train")
    d2v_test, y_test = load_feature(D2V_DIR, "test")

    mfcc_train, _ = load_feature(MFCC_DIR, "train")
    mfcc_test, _ = load_feature(MFCC_DIR, "test")

    ecapa_train, _ = load_feature(ECAPA_DIR, "train")
    ecapa_test, _ = load_feature(ECAPA_DIR, "test")

    y_train = fix_labels(y_train)
    y_test = fix_labels(y_test)

    d2v_train, d2v_test = standardize(d2v_train, d2v_test)
    mfcc_train, mfcc_test = standardize(mfcc_train, mfcc_test)
    ecapa_train, ecapa_test = standardize(ecapa_train, ecapa_test)

    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y_test)

    print("Train clips:", len(y_train))
    print("Test clips:", len(y_test))
    print("Dogs/classes:", len(le.classes_))

    class_counts = np.bincount(y_train_int)
    sample_weights = 1.0 / class_counts[y_train_int]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    ds = TensorDataset(
        torch.tensor(d2v_train),
        torch.tensor(mfcc_train),
        torch.tensor(ecapa_train),
        torch.tensor(y_train_int, dtype=torch.long)
    )

    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    model = AttentionFusion().to(DEVICE)
    head = ArcFaceHead(EMBED_DIM, len(le.classes_), SCALE, MARGIN).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        total = 0

        for d2v, mfcc, ecapa, yb in loader:
            d2v = d2v.to(DEVICE)
            mfcc = mfcc.to(DEVICE)
            ecapa = ecapa.to(DEVICE)
            yb = yb.to(DEVICE)

            emb, weights = model(d2v, mfcc, ecapa)
            logits = head(emb, yb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * yb.size(0)

        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={total / len(ds):.4f}")

    model.eval()

    with torch.no_grad():
        train_emb, train_w = model(
            torch.tensor(d2v_train).to(DEVICE),
            torch.tensor(mfcc_train).to(DEVICE),
            torch.tensor(ecapa_train).to(DEVICE)
        )
        test_emb, test_w = model(
            torch.tensor(d2v_test).to(DEVICE),
            torch.tensor(mfcc_test).to(DEVICE),
            torch.tensor(ecapa_test).to(DEVICE)
        )

    np.save(RESULTS_DIR / "attention_train_embeddings.npy", train_emb.cpu().numpy())
    np.save(RESULTS_DIR / "attention_test_embeddings.npy", test_emb.cpu().numpy())
    np.save(RESULTS_DIR / "attention_train_weights.npy", train_w.cpu().numpy())
    np.save(RESULTS_DIR / "attention_test_weights.npy", test_w.cpu().numpy())
    np.save(RESULTS_DIR / "attention_label_classes.npy", le.classes_)

    np.save(DATA_DIR / "attention_train_labels.npy", y_train)
    np.save(DATA_DIR / "attention_test_labels.npy", y_test)

    torch.save(model.state_dict(), MODEL_DIR / "attention_fusion_arcface.pt")
    torch.save(head.state_dict(), MODEL_DIR / "attention_arcface_head.pt")

    print("Saved attention fusion embeddings and model.")
    print("Train embeddings:", train_emb.shape)
    print("Test embeddings:", test_emb.shape)
    print("Average attention weights [Dog2Vec, MFCC, ECAPA]:")
    print(test_w.mean(dim=0).cpu().numpy())


if __name__ == "__main__":
    main()
