from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "fusion3_train.npy"
X_TEST_PATH = DATA_DIR / "fusion3_test.npy"
Y_TRAIN_PATH = DATA_DIR / "fusion3_train_labels.npy"
Y_TEST_PATH = DATA_DIR / "fusion3_test_labels.npy"

OUT_ALL = RESULTS_DIR / "repeated_holdout_all_results.csv"
OUT_SUMMARY = RESULTS_DIR / "repeated_holdout_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 5
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_K = 5
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
EMBED_DIM = 256
SCALE = 30.0
MARGIN = 0.5
THRESHOLDS = np.arange(0.10, 0.95, 0.05)


class FusionMLP(nn.Module):
    def __init__(self, input_dim=1040, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


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


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def standardize(train, other):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (train - mean) / std, (other - mean) / std, mean, std


def make_profiles(X, y, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}

    for dog in sorted(set(y)):
        idx = np.where(y == dog)[0]
        if len(idx) < k:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        c = X[chosen].mean(axis=0)
        c = c / np.linalg.norm(c)
        profiles[dog] = c

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])
    return labels, mat


def train_model(X_train, y_train, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_int = le.fit_transform(y_train)

    class_counts = np.bincount(y_int)
    sample_weights = 1.0 / class_counts[y_int]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_int, dtype=torch.long),
    )

    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    model = FusionMLP(input_dim=X_train.shape[1], embed_dim=EMBED_DIM).to(DEVICE)
    head = ArcFaceHead(EMBED_DIM, len(le.classes_), SCALE, MARGIN).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        total = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            emb = model(xb)
            logits = head(emb, yb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)

        print(f"    epoch {epoch:03d}/{EPOCHS} loss={total / len(ds):.4f}")

    return model


def embed(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()


def main():
    print("Device:", DEVICE)

    X_train_full = np.load(X_TRAIN_PATH).astype("float32")
    X_test_full = np.load(X_TEST_PATH).astype("float32")
    y_train_full = np.load(Y_TRAIN_PATH, allow_pickle=True)
    y_test_full = np.load(Y_TEST_PATH, allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train_full)))
    all_rows = []

    for run in range(N_RUNS):
        seed = 42 + run
        rng = np.random.default_rng(seed)

        unknown_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unknown_dogs

        train_mask = np.array([y in known_dogs for y in y_train_full])
        known_test_mask = np.array([y in known_dogs for y in y_test_full])
        unknown_test_mask = np.array([y in unknown_dogs for y in y_test_full])

        X_train = X_train_full[train_mask]
        y_train = y_train_full[train_mask]

        X_known_test = X_test_full[known_test_mask]
        y_known_test = y_test_full[known_test_mask]

        X_unknown_test = X_test_full[unknown_test_mask]

        X_train_std, X_known_test_std, mean, std = standardize(X_train, X_known_test)
        X_unknown_test_std = (X_unknown_test - mean) / std

        print("\n==============================")
        print(f"Run {run + 1}/{N_RUNS}")
        print("Known dogs:", len(known_dogs))
        print("Unknown dogs:", len(unknown_dogs))
        print("Unknown IDs:", sorted(unknown_dogs))
        print("Train clips:", X_train.shape[0])
        print("Known test clips:", X_known_test.shape[0])
        print("Unknown test clips:", X_unknown_test.shape[0])

        model = train_model(X_train_std, y_train, seed)

        train_emb = normalize(embed(model, X_train_std))
        known_emb = normalize(embed(model, X_known_test_std))
        unknown_emb = normalize(embed(model, X_unknown_test_std))

        profile_labels, profile_mat = make_profiles(train_emb, y_train, ENROLL_K, seed)

        X_eval = np.vstack([known_emb, unknown_emb])
        y_eval = np.concatenate([
            y_known_test,
            np.array(["unknown"] * len(unknown_emb)),
        ])

        sims = cosine_similarity(X_eval, profile_mat)
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(X_eval)), best_idx]
        closest = np.array([profile_labels[i] for i in best_idx])

        for threshold in THRESHOLDS:
            pred = np.where(best_scores < threshold, "unknown", closest)

            known_mask = y_eval != "unknown"
            unknown_mask = y_eval == "unknown"

            row = {
                "run": run + 1,
                "seed": seed,
                "enroll_k": ENROLL_K,
                "threshold": round(float(threshold), 2),
                "known_dogs": len(known_dogs),
                "unknown_dogs": len(unknown_dogs),
                "known_test_clips": int(known_mask.sum()),
                "unknown_test_clips": int(unknown_mask.sum()),
                "overall_accuracy": accuracy_score(y_eval, pred),
                "known_id_accuracy": accuracy_score(y_eval[known_mask], pred[known_mask]),
                "unknown_recall": np.mean(pred[unknown_mask] == "unknown"),
                "false_unknown_rate": np.mean(pred[known_mask] == "unknown"),
                "unknown_ids": ",".join(sorted(unknown_dogs)),
            }
            all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(OUT_ALL, index=False)

    all_df = pd.DataFrame(all_rows)
    summary = (
        all_df
        .groupby("threshold")
        [["overall_accuracy", "known_id_accuracy", "unknown_recall", "false_unknown_rate"]]
        .agg(["mean", "std"])
    )

    summary.to_csv(OUT_SUMMARY)

    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)

    print("\nBest thresholds by mean overall accuracy:")
    mean_df = all_df.groupby("threshold").agg({
        "overall_accuracy": ["mean", "std"],
        "known_id_accuracy": ["mean", "std"],
        "unknown_recall": ["mean", "std"],
        "false_unknown_rate": ["mean", "std"],
    })

    print(mean_df.sort_values(("overall_accuracy", "mean"), ascending=False).head(10))


if __name__ == "__main__":
    main()
