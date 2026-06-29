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

N_RUNS = 20
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

EPOCHS = 20
BATCH_P = 32
BATCH_K = 4
LR = 1e-3
EMBED_DIM = 128

OUT_ALL = RESULTS_DIR / "feature_ablation_proxy_strict_all.csv"
OUT_SUMMARY = RESULTS_DIR / "feature_ablation_proxy_strict_summary.csv"


class MetricMLP(nn.Module):
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


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, z, labels):
        proxies = F.normalize(self.proxies, p=2, dim=1)
        sim = z @ proxies.T

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        pos_exp = torch.exp(-self.alpha * (sim - self.margin)) * one_hot
        neg_exp = torch.exp(self.alpha * (sim + self.margin)) * (1 - one_hot)

        pos_term = torch.log1p(pos_exp.sum(dim=0))
        neg_term = torch.log1p(neg_exp.sum(dim=0))

        valid_pos = one_hot.sum(dim=0) > 0

        return pos_term[valid_pos].mean() + neg_term.mean()


def standardize_from_known(X_train_full, X_test_full, known_train_mask):
    known_train = X_train_full[known_train_mask]

    mean = known_train.mean(axis=0, keepdims=True)
    std = known_train.std(axis=0, keepdims=True)
    std[std == 0] = 1

    return (X_train_full - mean) / std, (X_test_full - mean) / std


def balanced_batch_indices(y, p=BATCH_P, k=BATCH_K, steps=None, seed=0):
    rng = np.random.default_rng(seed)

    classes = np.array(sorted(set(y)))
    by_class = {c: np.where(y == c)[0] for c in classes}

    if steps is None:
        steps = max(1, len(y) // (p * k))

    for _ in range(steps):
        chosen_classes = rng.choice(classes, size=min(p, len(classes)), replace=False)
        batch = []

        for c in chosen_classes:
            idx = by_class[c]
            replace = len(idx) < k
            batch.extend(rng.choice(idx, size=k, replace=replace))

        rng.shuffle(batch)
        yield np.array(batch)


def train_proxy_anchor(X_train, y_train_raw, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    model = MetricMLP(X_train.shape[1], EMBED_DIM).to(DEVICE)
    loss_fn = ProxyAnchorLoss(
        num_classes=len(le.classes_),
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    X_train = X_train.astype("float32")
    steps = max(1, len(X_train) // (BATCH_P * BATCH_K))

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total = 0.0
        n_steps = 0

        for batch_idx in balanced_batch_indices(y_train, seed=seed + epoch, steps=steps):
            xb = torch.tensor(X_train[batch_idx]).to(DEVICE)
            yb = torch.tensor(y_train[batch_idx]).long().to(DEVICE)

            z = model(xb)
            loss = loss_fn(z, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n_steps += 1

        print(f"    epoch {epoch}/{EPOCHS} loss={total / max(n_steps, 1):.4f}")

    return model


def embed(model, X):
    model.eval()
    outs = []

    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i + 1024].astype("float32")).to(DEVICE)
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


def evaluate(model, X_train_full, X_test_full, y_train, y_test, unseen_dogs, k, seed):
    unseen_train_mask = np.array([y in unseen_dogs for y in y_train])
    unseen_test_mask = np.array([y in unseen_dogs for y in y_test])

    X_enroll = embed(model, X_train_full[unseen_train_mask])
    y_enroll = y_train[unseen_train_mask]

    X_probe_all = embed(model, X_test_full[unseen_test_mask])
    y_probe_all = y_test[unseen_test_mask]

    profile_labels, profile_mat = make_profiles(
        X_enroll,
        y_enroll,
        unseen_dogs,
        k,
        seed,
    )

    valid = np.array([y in set(profile_labels) for y in y_probe_all])
    X_probe = X_probe_all[valid]
    y_probe = y_probe_all[valid]

    sims = cosine_similarity(X_probe, profile_mat)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(X_probe)), best_idx]
    pred = np.array([profile_labels[i] for i in best_idx])

    acc = accuracy_score(y_probe, pred)

    return {
        "accuracy": acc,
        "test_barks": len(y_probe),
        "unseen_dogs": len(profile_labels),
        "mean_best_similarity": float(np.mean(best_scores)),
        "std_best_similarity": float(np.std(best_scores)),
    }


def main():
    print("Device:", DEVICE)

    X_train_fusion3 = np.load(DATA_DIR / "fusion3_train.npy").astype("float32")
    X_test_fusion3 = np.load(DATA_DIR / "fusion3_test.npy").astype("float32")

    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    print("Fusion3 train:", X_train_fusion3.shape)
    print("Fusion3 test:", X_test_fusion3.shape)

    assert X_train_fusion3.shape[1] == 1040, "Expected fusion3 to be 1040-D"

    feature_sets = {
        "dog2vec": (slice(0, 768),),
        "mfcc": (slice(768, 848),),
        "ecapa": (slice(848, 1040),),
        "dog2vec_mfcc": (slice(0, 848),),
        "dog2vec_ecapa": (slice(0, 768), slice(848, 1040)),
        "mfcc_ecapa": (slice(768, 1040),),
        "dog2vec_mfcc_ecapa": (slice(0, 1040),),
    }

    all_dogs = np.array(sorted(set(y_train)))
    rows = []

    for run in range(N_RUNS):
        seed = 7000 + run
        rng = np.random.default_rng(seed)

        unseen_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unseen_dogs

        known_train_mask = np.array([y in known_dogs for y in y_train])

        print("\n==============================")
        print(f"Run {run + 1}/{N_RUNS}")
        print("Known dogs:", len(known_dogs))
        print("Unseen dogs:", sorted(unseen_dogs))

        for feature_name, slices in feature_sets.items():
            print(f"\n  Feature set: {feature_name}")

            X_train_raw = np.concatenate(
                [X_train_fusion3[:, s] for s in slices],
                axis=1,
            )
            X_test_raw = np.concatenate(
                [X_test_fusion3[:, s] for s in slices],
                axis=1,
            )

            X_train_std, X_test_std = standardize_from_known(
                X_train_raw,
                X_test_raw,
                known_train_mask,
            )

            model = train_proxy_anchor(
                X_train_std[known_train_mask],
                y_train[known_train_mask],
                seed,
            )

            for k in ENROLL_KS:
                metrics = evaluate(
                    model,
                    X_train_std,
                    X_test_std,
                    y_train,
                    y_test,
                    unseen_dogs,
                    k,
                    seed + k,
                )

                row = {
                    "run": run + 1,
                    "seed": seed,
                    "feature_set": feature_name,
                    "loss": "proxy_anchor",
                    "enroll_k": k,
                    "known_train_dogs": len(known_dogs),
                    "protocol": "strict_enroll_train_probe_test",
                    "unseen_ids": ",".join(sorted(unseen_dogs)),
                }
                row.update(metrics)
                rows.append(row)

                print(
                    f"    enroll_k={k} | "
                    f"test_barks={metrics['test_barks']} | "
                    f"acc={metrics['accuracy']:.4f}"
                )

            pd.DataFrame(rows).to_csv(OUT_ALL, index=False)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["feature_set", "loss", "enroll_k"])
        [["accuracy", "test_barks", "mean_best_similarity"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nFeature ablation summary:")
    print(summary)

    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
