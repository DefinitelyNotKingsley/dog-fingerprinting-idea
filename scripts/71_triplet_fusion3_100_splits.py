from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

OUT_ALL = RESULTS_DIR / "triplet_fusion3_100splits_all.csv"
OUT_SUMMARY = RESULTS_DIR / "triplet_fusion3_100splits_summary.csv"

N_RUNS = 100
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]
THRESHOLDS = np.arange(0.10, 0.95, 0.05)
FAR_LEVELS = [0.001, 0.01, 0.1]


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


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


def eer(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    fars = []
    frrs = []

    P = labels.sum()
    N = len(labels) - P

    for t in scores:
        pred = scores >= t
        fa = np.sum((pred == 1) & (labels == 0))
        fr = np.sum((pred == 0) & (labels == 1))

        fars.append(fa / N if N > 0 else 0)
        frrs.append(fr / P if P > 0 else 0)

    fars = np.array(fars)
    frrs = np.array(frrs)
    i = np.argmin(np.abs(fars - frrs))
    return (fars[i] + frrs[i]) / 2


def min_dcf(pos_scores, neg_scores, p_target=0.01, c_miss=1, c_fa=1):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    best = float("inf")
    P = labels.sum()
    N = len(labels) - P

    for t in scores:
        pred = scores >= t
        fn = np.sum((pred == 0) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))

        pmiss = fn / P if P > 0 else 0
        pfa = fp / N if N > 0 else 0

        dcf = c_miss * pmiss * p_target + c_fa * pfa * (1 - p_target)
        best = min(best, dcf)

    default = min(c_miss * p_target, c_fa * (1 - p_target))
    return best / default


def dir_at_far(known_correct_scores, unknown_scores, far_level):
    if len(unknown_scores) == 0:
        return 0.0

    threshold = np.quantile(unknown_scores, 1 - far_level)
    return np.mean(known_correct_scores >= threshold)


def main():
    X_train = normalize(np.load(RESULTS_DIR / "triplet_fusion3_train_embeddings.npy"))
    X_test = normalize(np.load(RESULTS_DIR / "triplet_fusion3_test_embeddings.npy"))

    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_dogs = np.array(sorted(set(y_train)))
    rows = []

    for run in range(N_RUNS):
        seed = 1000 + run
        rng = np.random.default_rng(seed)

        unknown_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))
        known_dogs = set(all_dogs) - unknown_dogs

        train_mask = np.array([y in known_dogs for y in y_train])
        known_test_mask = np.array([y in known_dogs for y in y_test])
        unknown_test_mask = np.array([y in unknown_dogs for y in y_test])

        X_known_train = X_train[train_mask]
        y_known_train = y_train[train_mask]

        X_known_test = X_test[known_test_mask]
        y_known_test = y_test[known_test_mask]

        X_unknown_test = X_test[unknown_test_mask]

        print(f"Run {run+1}/{N_RUNS} | known_test={len(X_known_test)} unknown_test={len(X_unknown_test)}")

        for k in ENROLL_KS:
            profile_labels, profile_mat = make_profiles(X_known_train, y_known_train, k, seed)

            known_sims = cosine_similarity(X_known_test, profile_mat)
            known_best_idx = np.argmax(known_sims, axis=1)
            known_best_scores = known_sims[np.arange(len(X_known_test)), known_best_idx]
            known_pred = np.array([profile_labels[i] for i in known_best_idx])
            known_correct = known_pred == y_known_test

            unknown_sims = cosine_similarity(X_unknown_test, profile_mat)
            unknown_best_scores = unknown_sims.max(axis=1)

            pos_scores = known_best_scores[known_correct]
            neg_scores = np.concatenate([
                known_best_scores[~known_correct],
                unknown_best_scores,
            ])

            eer_value = eer(pos_scores, neg_scores)
            mindcf_value = min_dcf(pos_scores, neg_scores)

            dir_scores = {
                f"DIR@FAR={far}": dir_at_far(pos_scores, unknown_best_scores, far)
                for far in FAR_LEVELS
            }

            for threshold in THRESHOLDS:
                pred_known = np.where(known_best_scores < threshold, "unknown", known_pred)
                pred_unknown = np.where(unknown_best_scores < threshold, "unknown", "known_dog")

                y_eval = np.concatenate([
                    y_known_test,
                    np.array(["unknown"] * len(X_unknown_test)),
                ])

                pred_eval = np.concatenate([
                    pred_known,
                    pred_unknown,
                ])

                row = {
                    "run": run + 1,
                    "seed": seed,
                    "enroll_k": k,
                    "threshold": round(float(threshold), 2),
                    "overall_accuracy": accuracy_score(y_eval, pred_eval),
                    "known_id_accuracy": accuracy_score(y_known_test, pred_known),
                    "unknown_recall": np.mean(pred_unknown == "unknown"),
                    "false_unknown_rate": np.mean(pred_known == "unknown"),
                    "EER_percent": eer_value * 100,
                    "minDCF": mindcf_value,
                    "unknown_ids": ",".join(sorted(unknown_dogs)),
                }
                row.update(dir_scores)
                rows.append(row)

        pd.DataFrame(rows).to_csv(OUT_ALL, index=False)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["enroll_k", "threshold"])
        [["overall_accuracy", "known_id_accuracy", "unknown_recall", "false_unknown_rate"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT_SUMMARY)

    final = (
        df.groupby("enroll_k")
        [["EER_percent", "minDCF", "DIR@FAR=0.001", "DIR@FAR=0.01", "DIR@FAR=0.1"]]
        .agg(["mean", "std"])
    )

    print("\n100-split final summary:")
    print(final)
    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
