import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "full_eval_suite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 100          # start with 20; later change to 100
ENROLLED_DOGS_PER_RUN = 10
UNKNOWN_REJECT_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

EPOCHS = 20
BATCH_P = 32
BATCH_K = 4
LR = 1e-3
EMBED_DIM = 128

METHODS = ["triplet", "supcon", "proxy_anchor"]
FAR_LEVELS = [0.001, 0.01, 0.1]
THRESHOLDS = np.arange(0.05, 0.96, 0.01)


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
    known = X_train_full[known_train_mask]
    mean = known.mean(axis=0, keepdims=True)
    std = known.std(axis=0, keepdims=True)
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
            batch.extend(rng.choice(idx, size=k, replace=len(idx) < k))

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
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


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

    return -((mask * log_prob).sum(dim=1)[valid] / pos_count[valid]).mean()


def train_model(X_train, y_train_raw, method, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    model = MetricMLP(X_train.shape[1], EMBED_DIM).to(DEVICE)

    extra_params = []
    proxy_loss_fn = None

    if method == "proxy_anchor":
        proxy_loss_fn = ProxyAnchorLoss(len(le.classes_), EMBED_DIM).to(DEVICE)
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
            elif method == "proxy_anchor":
                loss = proxy_loss_fn(z, yb)
            else:
                raise ValueError(method)

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
            outs.append(model(xb).cpu().numpy())

    return np.vstack(outs)


def make_profiles(X_enroll, y_enroll, dogs, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}

    for dog in sorted(dogs):
        idx = np.where(y_enroll == dog)[0]
        idx = rng.permutation(idx)

        if len(idx) < k:
            continue

        profile = X_enroll[idx[:k]].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-12)
        profiles[dog] = profile

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])

    return labels, mat


def eer(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    P = labels.sum()
    N = len(labels) - P

    fars, frrs = [], []

    for t in scores:
        pred = scores >= t
        fa = np.sum((pred == 1) & (labels == 0))
        fr = np.sum((pred == 0) & (labels == 1))
        fars.append(fa / N if N > 0 else 0)
        frrs.append(fr / P if P > 0 else 0)

    fars = np.array(fars)
    frrs = np.array(frrs)
    i = np.argmin(np.abs(fars - frrs))
    return float((fars[i] + frrs[i]) / 2)


def min_dcf(pos_scores, neg_scores, p_target=0.01, c_miss=1, c_fa=1):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    P = labels.sum()
    N = len(labels) - P
    best = float("inf")

    for t in scores:
        pred = scores >= t
        fn = np.sum((pred == 0) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))

        pmiss = fn / P if P > 0 else 0
        pfa = fp / N if N > 0 else 0

        dcf = c_miss * pmiss * p_target + c_fa * pfa * (1 - p_target)
        best = min(best, dcf)

    default = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(best / default)


def dir_at_far(correct_scores, unknown_scores, far):
    if len(unknown_scores) == 0:
        return np.nan

    threshold = np.quantile(unknown_scores, 1 - far)
    return float(np.mean(correct_scores >= threshold))


def evaluate_full(model, X_train_std, X_test_std, y_train, y_test,
                  enrolled_dogs, unknown_dogs, k, seed):
    enrolled_train_mask = np.array([y in enrolled_dogs for y in y_train])
    enrolled_test_mask = np.array([y in enrolled_dogs for y in y_test])
    unknown_test_mask = np.array([y in unknown_dogs for y in y_test])

    X_enroll = embed(model, X_train_std[enrolled_train_mask])
    y_enroll = y_train[enrolled_train_mask]

    X_known_probe = embed(model, X_test_std[enrolled_test_mask])
    y_known_probe = y_test[enrolled_test_mask]

    X_unknown_probe = embed(model, X_test_std[unknown_test_mask])
    y_unknown_probe = y_test[unknown_test_mask]

    profile_labels, profile_mat = make_profiles(X_enroll, y_enroll, enrolled_dogs, k, seed)

    valid_known = np.array([y in set(profile_labels) for y in y_known_probe])
    X_known_probe = X_known_probe[valid_known]
    y_known_probe = y_known_probe[valid_known]

    known_sims = cosine_similarity(X_known_probe, profile_mat)
    known_best_idx = np.argmax(known_sims, axis=1)
    known_best_scores = known_sims[np.arange(len(X_known_probe)), known_best_idx]
    known_pred = np.array([profile_labels[i] for i in known_best_idx])

    known_correct = known_pred == y_known_probe
    rank1_acc = accuracy_score(y_known_probe, known_pred)

    true_label_to_col = {d: i for i, d in enumerate(profile_labels)}
    true_cols = np.array([true_label_to_col[y] for y in y_known_probe])
    pos_scores = known_sims[np.arange(len(X_known_probe)), true_cols]

    neg_mask = np.ones_like(known_sims, dtype=bool)
    neg_mask[np.arange(len(X_known_probe)), true_cols] = False
    neg_scores = known_sims[neg_mask]

    unknown_best_scores = np.array([])
    if len(X_unknown_probe) > 0:
        unknown_sims = cosine_similarity(X_unknown_probe, profile_mat)
        unknown_best_scores = unknown_sims.max(axis=1)

    combined_neg = np.concatenate([neg_scores, unknown_best_scores])

    eer_value = eer(pos_scores, combined_neg)
    mindcf_value = min_dcf(pos_scores, combined_neg)

    correct_scores = known_best_scores[known_correct]

    dir_metrics = {
        f"DIR@FAR={far}": dir_at_far(correct_scores, unknown_best_scores, far)
        for far in FAR_LEVELS
    }

    threshold_rows = []
    for t in THRESHOLDS:
        known_accept = known_best_scores >= t
        unknown_reject = unknown_best_scores < t if len(unknown_best_scores) > 0 else np.array([])

        known_id_acc_at_t = np.mean((known_pred == y_known_probe) & known_accept)
        false_unknown_rate = np.mean(~known_accept)

        unknown_recall = np.mean(unknown_reject) if len(unknown_reject) > 0 else np.nan
        far = np.mean(~unknown_reject) if len(unknown_reject) > 0 else np.nan

        y_open = np.concatenate([
            np.ones(len(known_best_scores)),
            np.zeros(len(unknown_best_scores)),
        ])
        pred_open = np.concatenate([
            known_accept.astype(int),
            (unknown_best_scores >= t).astype(int),
        ])

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_open,
            pred_open,
            average="binary",
            zero_division=0,
        )

        threshold_rows.append({
            "threshold": float(t),
            "known_id_accuracy_at_threshold": float(known_id_acc_at_t),
            "false_unknown_rate": float(false_unknown_rate),
            "unknown_recall": float(unknown_recall),
            "far": float(far),
            "binary_precision_known_vs_unknown": float(precision),
            "binary_recall_known_vs_unknown": float(recall),
            "binary_f1_known_vs_unknown": float(f1),
        })

    confusion_rows = []
    for true, pred, score in zip(y_known_probe, known_pred, known_best_scores):
        confusion_rows.append({
            "true_dog": true,
            "pred_dog": pred,
            "score": float(score),
            "correct": bool(true == pred),
        })

    per_dog_rows = []
    for dog in sorted(set(y_known_probe)):
        m = y_known_probe == dog
        per_dog_rows.append({
            "dog_id": dog,
            "num_probes": int(m.sum()),
            "accuracy": float(np.mean(known_pred[m] == y_known_probe[m])),
            "mean_best_score": float(np.mean(known_best_scores[m])),
        })

    score_rows = []
    score_rows.extend([{"score_type": "positive_true_profile", "score": float(s)} for s in pos_scores])
    score_rows.extend([{"score_type": "negative_wrong_profile", "score": float(s)} for s in neg_scores])
    score_rows.extend([{"score_type": "unknown_best_profile", "score": float(s)} for s in unknown_best_scores])

    metrics = {
        "rank1_accuracy": float(rank1_acc),
        "EER_percent": float(eer_value * 100),
        "minDCF": float(mindcf_value),
        "num_enrolled_dogs": len(profile_labels),
        "num_known_probes": len(y_known_probe),
        "num_unknown_reject_probes": len(y_unknown_probe),
        "mean_positive_score": float(np.mean(pos_scores)),
        "mean_negative_score": float(np.mean(neg_scores)),
        "mean_unknown_score": float(np.mean(unknown_best_scores)) if len(unknown_best_scores) else np.nan,
    }
    metrics.update(dir_metrics)

    return metrics, threshold_rows, confusion_rows, per_dog_rows, score_rows


def main():
    print("Device:", DEVICE)

    X_train_raw = np.load(DATA_DIR / "fusion3_train.npy").astype("float32")
    X_test_raw = np.load(DATA_DIR / "fusion3_test.npy").astype("float32")

    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train)))

    master_rows = []
    threshold_all = []
    confusion_all = []
    per_dog_all = []
    scores_all = []

    for run in range(N_RUNS):
        seed = 10000 + run
        rng = np.random.default_rng(seed)

        selected = rng.choice(
            all_dogs,
            size=ENROLLED_DOGS_PER_RUN + UNKNOWN_REJECT_DOGS_PER_RUN,
            replace=False,
        )

        enrolled_dogs = set(selected[:ENROLLED_DOGS_PER_RUN])
        unknown_dogs = set(selected[ENROLLED_DOGS_PER_RUN:])
        known_train_dogs = set(all_dogs) - enrolled_dogs - unknown_dogs

        known_train_mask = np.array([y in known_train_dogs for y in y_train])

        X_train_std, X_test_std = standardize_from_known(
            X_train_raw,
            X_test_raw,
            known_train_mask,
        )

        X_known_train = X_train_std[known_train_mask]
        y_known_train = y_train[known_train_mask]

        print("\n==============================")
        print(f"Run {run+1}/{N_RUNS}")
        print("Train dogs:", len(known_train_dogs))
        print("Enrolled dogs:", sorted(enrolled_dogs))
        print("Unknown reject dogs:", sorted(unknown_dogs))
        print("Training clips:", len(X_known_train))

        for method in METHODS:
            print(f"\n  Training loss: {method}")
            model = train_model(X_known_train, y_known_train, method, seed)

            for k in ENROLL_KS:
                metrics, threshold_rows, confusion_rows, per_dog_rows, score_rows = evaluate_full(
                    model=model,
                    X_train_std=X_train_std,
                    X_test_std=X_test_std,
                    y_train=y_train,
                    y_test=y_test,
                    enrolled_dogs=enrolled_dogs,
                    unknown_dogs=unknown_dogs,
                    k=k,
                    seed=seed + k,
                )

                base = {
                    "run": run + 1,
                    "seed": seed,
                    "feature_set": "dog2vec_mfcc_ecapa",
                    "loss": method,
                    "enroll_k": k,
                    "protocol": "strict_train_enroll_probe_unknown_reject",
                    "train_dogs": len(known_train_dogs),
                    "enrolled_ids": ",".join(sorted(enrolled_dogs)),
                    "unknown_reject_ids": ",".join(sorted(unknown_dogs)),
                }

                row = dict(base)
                row.update(metrics)
                master_rows.append(row)

                for r in threshold_rows:
                    rr = dict(base)
                    rr.update(r)
                    threshold_all.append(rr)

                for r in confusion_rows:
                    rr = dict(base)
                    rr.update(r)
                    confusion_all.append(rr)

                for r in per_dog_rows:
                    rr = dict(base)
                    rr.update(r)
                    per_dog_all.append(rr)

                for r in score_rows:
                    rr = dict(base)
                    rr.update(r)
                    scores_all.append(rr)

                print(
                    f"    k={k} | rank1={metrics['rank1_accuracy']:.4f} | "
                    f"EER={metrics['EER_percent']:.2f}% | minDCF={metrics['minDCF']:.4f} | "
                    f"DIR@0.01={metrics['DIR@FAR=0.01']:.4f}"
                )

                pd.DataFrame(master_rows).to_csv(OUT_DIR / "master_metrics_all.csv", index=False)
                pd.DataFrame(threshold_all).to_csv(OUT_DIR / "threshold_curves_all.csv", index=False)
                pd.DataFrame(confusion_all).to_csv(OUT_DIR / "confusion_predictions_all.csv", index=False)
                pd.DataFrame(per_dog_all).to_csv(OUT_DIR / "per_dog_accuracy_all.csv", index=False)
                pd.DataFrame(scores_all).to_csv(OUT_DIR / "score_distributions_all.csv", index=False)

    master = pd.DataFrame(master_rows)

    summary = (
        master.groupby(["feature_set", "loss", "enroll_k"])
        [[
            "rank1_accuracy",
            "EER_percent",
            "minDCF",
            "DIR@FAR=0.001",
            "DIR@FAR=0.01",
            "DIR@FAR=0.1",
            "mean_positive_score",
            "mean_negative_score",
            "mean_unknown_score",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(OUT_DIR / "master_metrics_summary.csv", index=False)

    conf = pd.DataFrame(confusion_all)
    top_confusions = (
        conf[conf["true_dog"] != conf["pred_dog"]]
        .groupby(["loss", "enroll_k", "true_dog", "pred_dog"])
        .size()
        .reset_index(name="count")
        .sort_values(["loss", "enroll_k", "count"], ascending=[True, True, False])
    )
    top_confusions.to_csv(OUT_DIR / "top_confused_pairs.csv", index=False)

    per_dog = pd.DataFrame(per_dog_all)
    per_dog_summary = (
        per_dog.groupby(["loss", "enroll_k", "dog_id"])
        [["accuracy", "num_probes", "mean_best_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_dog_summary.to_csv(OUT_DIR / "per_dog_accuracy_summary.csv", index=False)

    print("\nFinal full evaluation summary:")
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(OUT_DIR / "master_metrics_all.csv")
    print(OUT_DIR / "master_metrics_summary.csv")
    print(OUT_DIR / "threshold_curves_all.csv")
    print(OUT_DIR / "confusion_predictions_all.csv")
    print(OUT_DIR / "top_confused_pairs.csv")
    print(OUT_DIR / "per_dog_accuracy_all.csv")
    print(OUT_DIR / "per_dog_accuracy_summary.csv")
    print(OUT_DIR / "score_distributions_all.csv")


if __name__ == "__main__":
    main()
