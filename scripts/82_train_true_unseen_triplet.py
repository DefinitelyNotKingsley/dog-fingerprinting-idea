import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 100
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
EMBED_DIM = 128
MARGIN = 0.3

OUT = RESULTS_DIR / "true_unseen_triplet_strict_results.csv"


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


def train_triplet(X_train, y_train, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = TripletDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = TripletMLP(X_train.shape[1], EMBED_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

            za = model(a)
            zp = model(p)
            zn = model(n)

            loss = loss_fn(za, zp, zn)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * a.size(0)

        print(f"  Epoch {epoch}/{EPOCHS} loss={total / len(ds):.4f}")

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


def make_profiles_from_enrollment(X_enroll, y_enroll, dogs, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}

    for dog in sorted(dogs):
        idx = np.where(y_enroll == dog)[0]
        idx = rng.permutation(idx)

        if len(idx) < k:
            continue

        enroll_idx = idx[:k]

        profile = X_enroll[enroll_idx].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-12)

        profiles[dog] = profile

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])

    return labels, mat


def main():
    print("Device:", DEVICE)

    X_train_full = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy").astype("float32")
    X_test_full = np.load(RESULTS_DIR / "arcface_fusion3_test_embeddings.npy").astype("float32")

    y_train_full = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test_full = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train_full)))
    rows = []

    for run in range(N_RUNS):
        seed = 3000 + run
        rng = np.random.default_rng(seed)

        unseen_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unseen_dogs

        print("\n==============================")
        print(f"Run {run+1}/{N_RUNS}")
        print("Known dogs for training:", len(known_dogs))
        print("Unseen dogs for enrollment/test:", len(unseen_dogs))
        print("Unseen IDs:", sorted(unseen_dogs))

        train_mask = np.array([y in known_dogs for y in y_train_full])

        X_known_train = X_train_full[train_mask]
        y_known_train = y_train_full[train_mask]

        print("Training clips:", len(X_known_train))

        model = train_triplet(X_known_train, y_known_train, seed)

        # Strict protocol:
        # unseen train split = enrollment only
        # unseen test split = probe/test only
        unseen_train_mask = np.array([y in unseen_dogs for y in y_train_full])
        unseen_test_mask = np.array([y in unseen_dogs for y in y_test_full])

        X_unseen_enroll = embed(model, X_train_full[unseen_train_mask])
        y_unseen_enroll = y_train_full[unseen_train_mask]

        X_unseen_probe = embed(model, X_test_full[unseen_test_mask])
        y_unseen_probe = y_test_full[unseen_test_mask]

        for k in ENROLL_KS:
            profile_labels, profile_mat = make_profiles_from_enrollment(
                X_unseen_enroll,
                y_unseen_enroll,
                unseen_dogs,
                k,
                seed + k,
            )

            # Only test dogs that successfully got a profile
            valid_probe_mask = np.array([y in set(profile_labels) for y in y_unseen_probe])
            X_probe = X_unseen_probe[valid_probe_mask]
            y_probe = y_unseen_probe[valid_probe_mask]

            sims = cosine_similarity(X_probe, profile_mat)
            best_idx = np.argmax(sims, axis=1)
            best_scores = sims[np.arange(len(X_probe)), best_idx]
            pred = np.array([profile_labels[i] for i in best_idx])

            acc = accuracy_score(y_probe, pred)

            print(f"  enroll_k={k} | test_barks={len(y_probe)} | acc={acc:.4f}")

            rows.append({
                "run": run + 1,
                "seed": seed,
                "enroll_k": k,
                "known_train_dogs": len(known_dogs),
                "unseen_dogs": len(profile_labels),
                "test_barks": len(y_probe),
                "accuracy": acc,
                "mean_best_similarity": float(np.mean(best_scores)),
                "std_best_similarity": float(np.std(best_scores)),
                "unseen_ids": ",".join(sorted(unseen_dogs)),
                "protocol": "strict_enroll_train_probe_test",
            })

        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("\nStrict true unseen fingerprinting summary:")
    print(df.groupby("enroll_k")[["accuracy", "test_barks", "mean_best_similarity"]].agg(["mean", "std"]))

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
