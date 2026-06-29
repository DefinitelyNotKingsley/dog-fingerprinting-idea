import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 20          # start with 20; later use 100
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
EMBED_DIM = 256
SCALE = 30.0
MARGIN = 0.5

OUT_ALL = RESULTS_DIR / "arcface_fusion3_strict_all.csv"
OUT_SUMMARY = RESULTS_DIR / "arcface_fusion3_strict_summary.csv"


class FusionMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
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


def standardize(train, other):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return (train - mean) / std, (other - mean) / std


def train_arcface(X_train, y_train_raw, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    model = FusionMLP(X_train.shape[1], EMBED_DIM).to(DEVICE)
    head = ArcFaceHead(EMBED_DIM, len(le.classes_), SCALE, MARGIN).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    X_train = X_train.astype("float32")
    y_train = y_train.astype("int64")

    n = len(X_train)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()

        perm = np.random.permutation(n)
        total = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]

            xb = torch.tensor(X_train[idx]).to(DEVICE)
            yb = torch.tensor(y_train[idx]).long().to(DEVICE)

            z = model(xb)
            logits = head(z, yb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(idx)

        print(f"  Epoch {epoch}/{EPOCHS} loss={total / n:.4f}")

    return model


def embed(model, X):
    model.eval()
    outs = []

    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i+1024].astype("float32")).to(DEVICE)
            z = model(xb).cpu().numpy()
            outs.append(z)

    return np.vstack(outs)


def make_profiles(X_enroll, y_enroll, dogs, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}

    for dog in sorted(dogs):
        idx = np.where(y_enroll == dog)[0]
        idx = rng.permutation(idx)

        if len(idx) < k:
            continue

        chosen = idx[:k]
        profile = X_enroll[chosen].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-12)
        profiles[dog] = profile

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])

    return labels, mat


def main():
    print("Device:", DEVICE)

    X_train_full = np.load(DATA_DIR / "fusion3_train.npy").astype("float32")
    X_test_full = np.load(DATA_DIR / "fusion3_test.npy").astype("float32")

    y_train_full = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test_full = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train_full)))
    rows = []

    for run in range(N_RUNS):
        seed = 6000 + run
        rng = np.random.default_rng(seed)

        unseen_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unseen_dogs

        print("\n==============================")
        print(f"Run {run+1}/{N_RUNS}")
        print("Known dogs:", len(known_dogs))
        print("Unseen dogs:", sorted(unseen_dogs))

        known_train_mask = np.array([y in known_dogs for y in y_train_full])

        X_known_train_raw = X_train_full[known_train_mask]
        y_known_train = y_train_full[known_train_mask]

        # Standardize using known training dogs only
        mean = X_known_train_raw.mean(axis=0, keepdims=True)
        std = X_known_train_raw.std(axis=0, keepdims=True)
        std[std == 0] = 1

        X_known_train = (X_known_train_raw - mean) / std
        X_train_std = (X_train_full - mean) / std
        X_test_std = (X_test_full - mean) / std

        print("Training clips:", len(X_known_train))

        model = train_arcface(X_known_train, y_known_train, seed)

        unseen_train_mask = np.array([y in unseen_dogs for y in y_train_full])
        unseen_test_mask = np.array([y in unseen_dogs for y in y_test_full])

        X_enroll = embed(model, X_train_std[unseen_train_mask])
        y_enroll = y_train_full[unseen_train_mask]

        X_probe_all = embed(model, X_test_std[unseen_test_mask])
        y_probe_all = y_test_full[unseen_test_mask]

        for k in ENROLL_KS:
            profile_labels, profile_mat = make_profiles(
                X_enroll,
                y_enroll,
                unseen_dogs,
                k,
                seed + k,
            )

            valid = np.array([y in set(profile_labels) for y in y_probe_all])
            X_probe = X_probe_all[valid]
            y_probe = y_probe_all[valid]

            sims = cosine_similarity(X_probe, profile_mat)
            best_idx = np.argmax(sims, axis=1)
            best_scores = sims[np.arange(len(X_probe)), best_idx]
            pred = np.array([profile_labels[i] for i in best_idx])

            acc = accuracy_score(y_probe, pred)

            print(f"  enroll_k={k} | test_barks={len(y_probe)} | acc={acc:.4f}")

            rows.append({
                "run": run + 1,
                "seed": seed,
                "model": "arcface_fusion3_retrained_strict",
                "enroll_k": k,
                "known_train_dogs": len(known_dogs),
                "unseen_dogs": len(profile_labels),
                "test_barks": len(y_probe),
                "accuracy": acc,
                "mean_best_similarity": float(np.mean(best_scores)),
                "std_best_similarity": float(np.std(best_scores)),
                "unseen_ids": ",".join(sorted(unseen_dogs)),
                "protocol": "strict_retrain_arcface_known_only",
            })

        pd.DataFrame(rows).to_csv(OUT_ALL, index=False)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["model", "enroll_k"])
        [["accuracy", "test_barks", "mean_best_similarity"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nRetrained ArcFace strict summary:")
    print(summary)
    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
