import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 20          # use 20 first; later change to 100
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

EPOCHS = 20         # use 20 first; later change to 30
BATCH_P = 32        # dogs per batch
BATCH_K = 4         # clips per dog per batch
LR = 1e-3
EMBED_DIM = 128

METHODS = [
    "triplet",
    "supcon",
    "circle",
    "multi_similarity",
    "proxy_anchor",
]

OUT_ALL = RESULTS_DIR / "metric_loss_strict_all.csv"
OUT_SUMMARY = RESULTS_DIR / "metric_loss_strict_summary.csv"


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
        pos_loss = pos_term[valid_pos].mean()
        neg_loss = neg_term.mean()

        return pos_loss + neg_loss


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


def batch_hard_triplet_loss(z, labels, margin=0.3):
    dist = torch.cdist(z, z, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=z.device)

    pos_mask = same & ~eye
    neg_mask = ~same

    hardest_pos = dist.masked_fill(~pos_mask, -1e9).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, 1e9).min(dim=1).values

    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)

    return loss.mean()


def supervised_contrastive_loss(z, labels, temperature=0.07):
    sim = z @ z.T / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = mask.sum(dim=1)
    valid = pos_count > 0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def circle_loss(z, labels, margin=0.25, gamma=80):
    sim = z @ z.T
    labels = labels.view(-1, 1)
    same = torch.eq(labels, labels.T)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=z.device)

    pos_mask = same & ~eye
    neg_mask = ~same

    losses = []

    for i in range(sim.shape[0]):
        sp = sim[i][pos_mask[i]]
        sn = sim[i][neg_mask[i]]

        if len(sp) == 0 or len(sn) == 0:
            continue

        ap = torch.clamp_min(-sp.detach() + 1 + margin, 0.0)
        an = torch.clamp_min(sn.detach() + margin, 0.0)

        delta_p = 1 - margin
        delta_n = margin

        logit_p = -gamma * ap * (sp - delta_p)
        logit_n = gamma * an * (sn - delta_n)

        loss_i = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    return torch.stack(losses).mean()


def multi_similarity_loss(z, labels, alpha=2.0, beta=50.0, base=0.5):
    sim = z @ z.T
    labels = labels.view(-1, 1)
    same = torch.eq(labels, labels.T)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=z.device)

    pos_mask = same & ~eye
    neg_mask = ~same

    losses = []

    for i in range(sim.shape[0]):
        pos = sim[i][pos_mask[i]]
        neg = sim[i][neg_mask[i]]

        if len(pos) == 0 or len(neg) == 0:
            continue

        pos_loss = (1.0 / alpha) * torch.log1p(torch.exp(-alpha * (pos - base)).sum())
        neg_loss = (1.0 / beta) * torch.log1p(torch.exp(beta * (neg - base)).sum())

        losses.append(pos_loss + neg_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    return torch.stack(losses).mean()


def train_model(X_train, y_train_raw, method, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    model = MetricMLP(X_train.shape[1], EMBED_DIM).to(DEVICE)

    extra_params = []
    proxy_loss_fn = None

    if method == "proxy_anchor":
        proxy_loss_fn = ProxyAnchorLoss(
            num_classes=len(le.classes_),
            embed_dim=EMBED_DIM,
        ).to(DEVICE)
        extra_params = list(proxy_loss_fn.parameters())

    opt = torch.optim.AdamW(
        list(model.parameters()) + extra_params,
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

            if method == "triplet":
                loss = batch_hard_triplet_loss(z, yb)
            elif method == "supcon":
                loss = supervised_contrastive_loss(z, yb)
            elif method == "circle":
                loss = circle_loss(z, yb)
            elif method == "multi_similarity":
                loss = multi_similarity_loss(z, yb)
            elif method == "proxy_anchor":
                loss = proxy_loss_fn(z, yb)
            else:
                raise ValueError(f"Unknown method: {method}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n_steps += 1

        print(f"    {method} epoch {epoch}/{EPOCHS} loss={total / max(n_steps, 1):.4f}")

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


def evaluate_strict(model, X_train_full, X_test_full, y_train_full, y_test_full, unseen_dogs, k, seed):
    unseen_train_mask = np.array([y in unseen_dogs for y in y_train_full])
    unseen_test_mask = np.array([y in unseen_dogs for y in y_test_full])

    X_enroll = embed(model, X_train_full[unseen_train_mask])
    y_enroll = y_train_full[unseen_train_mask]

    X_probe = embed(model, X_test_full[unseen_test_mask])
    y_probe = y_test_full[unseen_test_mask]

    profile_labels, profile_mat = make_profiles(
        X_enroll,
        y_enroll,
        unseen_dogs,
        k,
        seed,
    )

    valid = np.array([y in set(profile_labels) for y in y_probe])
    X_probe = X_probe[valid]
    y_probe = y_probe[valid]

    sims = cosine_similarity(X_probe, profile_mat)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(X_probe)), best_idx]
    pred = np.array([profile_labels[i] for i in best_idx])

    acc = accuracy_score(y_probe, pred)

    return {
        "unseen_dogs_with_profiles": len(profile_labels),
        "test_barks": len(y_probe),
        "accuracy": acc,
        "mean_best_similarity": float(np.mean(best_scores)),
        "std_best_similarity": float(np.std(best_scores)),
    }


def main():
    print("Device:", DEVICE)

    X_train_full = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy").astype("float32")
    X_test_full = np.load(RESULTS_DIR / "arcface_fusion3_test_embeddings.npy").astype("float32")

    y_train_full = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test_full = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train_full)))
    rows = []

    for run in range(N_RUNS):
        seed = 4000 + run
        rng = np.random.default_rng(seed)

        unseen_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unseen_dogs

        train_mask = np.array([y in known_dogs for y in y_train_full])
        X_known_train = X_train_full[train_mask]
        y_known_train = y_train_full[train_mask]

        print("\n==============================")
        print(f"Run {run + 1}/{N_RUNS}")
        print("Known training dogs:", len(known_dogs))
        print("Unseen dogs:", sorted(unseen_dogs))
        print("Training clips:", len(X_known_train))

        for method in METHODS:
            print(f"\n  Training method: {method}")
            model = train_model(X_known_train, y_known_train, method, seed)

            for k in ENROLL_KS:
                metrics = evaluate_strict(
                    model,
                    X_train_full,
                    X_test_full,
                    y_train_full,
                    y_test_full,
                    unseen_dogs,
                    k,
                    seed + k,
                )

                row = {
                    "run": run + 1,
                    "seed": seed,
                    "method": method,
                    "enroll_k": k,
                    "known_train_dogs": len(known_dogs),
                    "unseen_dogs_requested": len(unseen_dogs),
                    "unseen_ids": ",".join(sorted(unseen_dogs)),
                    "protocol": "strict_enroll_train_probe_test",
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
        df.groupby(["method", "enroll_k"])
        [["accuracy", "test_barks", "mean_best_similarity"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nBig strict benchmark summary:")
    print(summary)

    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
