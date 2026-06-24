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

X_TRAIN_PATH = RESULTS_DIR / "triplet_fusion3_train_embeddings.npy"
X_TEST_PATH = RESULTS_DIR / "triplet_fusion3_test_embeddings.npy"
Y_TRAIN_PATH = DATA_DIR / "train_labels.npy"
Y_TEST_PATH = DATA_DIR / "test_labels.npy"

OUT_ALL = RESULTS_DIR / "triplet_fusion3_all_results.csv"
OUT_SUMMARY = RESULTS_DIR / "triplet_fusion3_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_RUNS = 5
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_K_LIST = [1, 3, 5]
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
EMBED_DIM = 256
SCALE = 30.0
MARGIN = 0.5

THRESHOLDS = np.arange(-1.0, 1.0, 0.001)
REPORT_THRESHOLDS = np.arange(0.10, 0.95, 0.05)
DIR_FAR_TARGETS = [0.001, 0.01, 0.1]


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
        c = c / (np.linalg.norm(c) + 1e-12)
        profiles[dog] = c

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])
    return labels, mat


def compute_open_set_scores(known_emb, y_known, unknown_emb, profile_labels, profile_mat):
    X_eval = np.vstack([known_emb, unknown_emb])
    y_eval = np.concatenate([
        y_known,
        np.array(["unknown"] * len(unknown_emb)),
    ])

    sims = cosine_similarity(X_eval, profile_mat)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(X_eval)), best_idx]
    closest = np.array([profile_labels[i] for i in best_idx])

    return y_eval, best_scores, closest


def compute_eer_from_best_scores(y_eval, best_scores):
    known_mask = y_eval != "unknown"
    unknown_mask = y_eval == "unknown"

    fars = []
    frrs = []

    for t in THRESHOLDS:
        frr = np.mean(best_scores[known_mask] < t)
        far = np.mean(best_scores[unknown_mask] >= t)

        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2

    return eer, THRESHOLDS[idx], fars[idx], frrs[idx]


def compute_mindcf(y_eval, best_scores, p_target=0.01, c_miss=1, c_fa=1):
    known_mask = y_eval != "unknown"
    unknown_mask = y_eval == "unknown"

    best_dcf = float("inf")
    best_t = None

    for t in THRESHOLDS:
        frr = np.mean(best_scores[known_mask] < t)
        far = np.mean(best_scores[unknown_mask] >= t)

        dcf = c_miss * frr * p_target + c_fa * far * (1 - p_target)

        if dcf < best_dcf:
            best_dcf = dcf
            best_t = t

    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    return best_dcf / default_dcf, best_t


def compute_dir_at_far(y_eval, best_scores, closest, far_target):
    known_mask = y_eval != "unknown"
    unknown_mask = y_eval == "unknown"

    best_dir = 0.0
    best_t = None
    best_far = None

    for t in THRESHOLDS:
        far = np.mean(best_scores[unknown_mask] >= t)

        if far <= far_target:
            accepted_known = best_scores[known_mask] >= t
            correct_id = closest[known_mask] == y_eval[known_mask]
            dir_score = np.mean(accepted_known & correct_id)

            if dir_score > best_dir:
                best_dir = dir_score
                best_t = t
                best_far = far

    return best_dir, best_t, best_far


def train_model(X_train, y_train, seed):
    # DogECAPA embeddings are already trained.
    # This dummy function keeps the original script structure.
    return None

def embed(model, X):
    return X


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

        X_train_std = X_train
        X_known_test_std = X_known_test
        X_unknown_test_std = X_unknown_test

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

        for enroll_k in ENROLL_K_LIST:
            profile_labels, profile_mat = make_profiles(train_emb, y_train, enroll_k, seed)

            y_eval, best_scores, closest = compute_open_set_scores(
                known_emb,
                y_known_test,
                unknown_emb,
                profile_labels,
                profile_mat,
            )

            known_mask = y_eval != "unknown"
            unknown_mask = y_eval == "unknown"

            eer, eer_t, far_at_eer, frr_at_eer = compute_eer_from_best_scores(
                y_eval,
                best_scores,
            )
            mindcf, mindcf_t = compute_mindcf(y_eval, best_scores)

            dir_results = {}
            for far_target in DIR_FAR_TARGETS:
                dir_score, dir_t, actual_far = compute_dir_at_far(
                    y_eval,
                    best_scores,
                    closest,
                    far_target,
                )
                dir_results[f"DIR@FAR={far_target}"] = dir_score
                dir_results[f"DIR_threshold@FAR={far_target}"] = dir_t
                dir_results[f"actual_FAR@{far_target}"] = actual_far

            for threshold in REPORT_THRESHOLDS:
                pred = np.where(best_scores < threshold, "unknown", closest)

                row = {
                    "run": run + 1,
                    "seed": seed,
                    "enroll_k": enroll_k,
                    "threshold": round(float(threshold), 2),
                    "known_dogs": len(known_dogs),
                    "unknown_dogs": len(unknown_dogs),
                    "known_test_clips": int(known_mask.sum()),
                    "unknown_test_clips": int(unknown_mask.sum()),
                    "overall_accuracy": accuracy_score(y_eval, pred),
                    "known_id_accuracy": accuracy_score(y_eval[known_mask], pred[known_mask]),
                    "unknown_recall": np.mean(pred[unknown_mask] == "unknown"),
                    "false_unknown_rate": np.mean(pred[known_mask] == "unknown"),
                    "EER": eer,
                    "EER_percent": eer * 100,
                    "EER_threshold": eer_t,
                    "FAR_at_EER": far_at_eer,
                    "FRR_at_EER": frr_at_eer,
                    "minDCF": mindcf,
                    "minDCF_threshold": mindcf_t,
                    "unknown_ids": ",".join(sorted(unknown_dogs)),
                }

                row.update(dir_results)
                all_rows.append(row)

            pd.DataFrame(all_rows).to_csv(OUT_ALL, index=False)

    all_df = pd.DataFrame(all_rows)

    metric_cols = [
        "overall_accuracy",
        "known_id_accuracy",
        "unknown_recall",
        "false_unknown_rate",
        "EER",
        "EER_percent",
        "minDCF",
        "DIR@FAR=0.001",
        "DIR@FAR=0.01",
        "DIR@FAR=0.1",
    ]

    summary = (
        all_df
        .groupby(["enroll_k", "threshold"])[metric_cols]
        .agg(["mean", "std"])
    )

    summary.to_csv(OUT_SUMMARY)

    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)

    print("\nBest thresholds by mean overall accuracy:")
    mean_df = all_df.groupby(["enroll_k", "threshold"])[metric_cols].mean()
    print(mean_df.sort_values("overall_accuracy", ascending=False).head(15))

    print("\nTriplet Fusion3 summary by enrollment size:")
    compact = all_df.groupby("enroll_k")[
        ["EER_percent", "minDCF", "DIR@FAR=0.001", "DIR@FAR=0.01", "DIR@FAR=0.1"]
    ].agg(["mean", "std"])

    print(compact)


if __name__ == "__main__":
    main()
