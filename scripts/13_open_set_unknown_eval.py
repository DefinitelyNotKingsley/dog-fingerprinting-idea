from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
OUT = RESULTS_DIR / "open_set_unknown_fusion3_results.csv"
SEED = 42

UNKNOWN_DOG_FRAC = 0.20
ENROLL_K = 5


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def make_profiles(X_train, y_train, known_dogs, k):
    rng = np.random.default_rng(SEED)
    profiles = {}

    for dog in sorted(known_dogs):
        idx = np.where(y_train == dog)[0]
        if len(idx) < k:
            continue

        chosen = rng.choice(idx, size=k, replace=False)
        centroid = X_train[chosen].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        profiles[dog] = centroid

    labels = list(profiles.keys())
    mat = np.vstack([profiles[d] for d in labels])
    return labels, mat


def main():
    rng = np.random.default_rng(SEED)

    X_train = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy")
    X_test = np.load(RESULTS_DIR / "arcface_fusion3_test_embeddings.npy")
    y_train = np.load(DATA_DIR / "fusion3_train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "fusion3_test_labels.npy", allow_pickle=True)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    all_dogs = np.array(sorted(set(y_train)))
    num_unknown = int(len(all_dogs) * UNKNOWN_DOG_FRAC)

    unknown_dogs = set(rng.choice(all_dogs, size=num_unknown, replace=False))
    known_dogs = set(all_dogs) - unknown_dogs

    labels, profile_mat = make_profiles(X_train, y_train, known_dogs, ENROLL_K)

    known_test_mask = np.array([dog in known_dogs for dog in y_test])
    unknown_test_mask = np.array([dog in unknown_dogs for dog in y_test])

    X_eval = np.vstack([X_test[known_test_mask], X_test[unknown_test_mask]])

    y_eval = np.concatenate([
        y_test[known_test_mask],
        np.array(["unknown"] * unknown_test_mask.sum())
    ])

    sims = cosine_similarity(X_eval, profile_mat)
    best_idx = np.argmax(sims, axis=1)
    best_scores = sims[np.arange(len(X_eval)), best_idx]
    closest_known = np.array([labels[i] for i in best_idx])

    rows = []

    for threshold in np.arange(0.10, 0.95, 0.05):
        y_pred = np.where(best_scores < threshold, "unknown", closest_known)

        known_mask = y_eval != "unknown"
        unknown_mask = y_eval == "unknown"

        known_id_acc = accuracy_score(y_eval[known_mask], y_pred[known_mask])
        unknown_recall = np.mean(y_pred[unknown_mask] == "unknown")
        false_unknown_rate = np.mean(y_pred[known_mask] == "unknown")
        overall_acc = accuracy_score(y_eval, y_pred)

        rows.append({
            "enroll_k": ENROLL_K,
            "threshold": round(float(threshold), 2),
            "known_dogs": len(known_dogs),
            "unknown_dogs": len(unknown_dogs),
            "known_test_clips": int(known_mask.sum()),
            "unknown_test_clips": int(unknown_mask.sum()),
            "overall_accuracy": overall_acc,
            "known_id_accuracy": known_id_acc,
            "unknown_recall": unknown_recall,
            "false_unknown_rate": false_unknown_rate,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("Known dogs:", len(known_dogs))
    print("Unknown dogs:", len(unknown_dogs))
    print("Known test clips:", int(known_test_mask.sum()))
    print("Unknown test clips:", int(unknown_test_mask.sum()))
    print("\nSaved:", OUT)
    print("\nBest thresholds by overall accuracy:")
    print(df.sort_values("overall_accuracy", ascending=False).head(10))


if __name__ == "__main__":
    main()
