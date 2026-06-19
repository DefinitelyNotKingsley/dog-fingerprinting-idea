from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
OUT = RESULTS_DIR / "unknown_rejection_fusion3_results.csv"
SEED = 42


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def make_profiles(X_train, y_train, k):
    rng = np.random.default_rng(SEED)
    profiles = {}

    for dog in sorted(set(y_train)):
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


def evaluate_threshold(y_true, y_pred_known, best_scores, threshold):
    y_pred = []

    for pred, score in zip(y_pred_known, best_scores):
        if score < threshold:
            y_pred.append("unknown")
        else:
            y_pred.append(pred)

    y_pred = np.array(y_pred)

    known_mask = y_true != "unknown"
    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask])
    known_f1 = f1_score(y_true[known_mask], y_pred[known_mask], average="macro")

    return known_acc, known_f1


def main():
    X_train = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy")
    X_test = np.load(RESULTS_DIR / "arcface_fusion3_test_embeddings.npy")
    y_train = np.load(DATA_DIR / "fusion3_train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "fusion3_test_labels.npy", allow_pickle=True)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    rows = []

    for k in [1, 3, 5, 10, 20]:
        labels, profile_mat = make_profiles(X_train, y_train, k)

        sims = cosine_similarity(X_test, profile_mat)
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(X_test)), best_idx]
        y_pred_known = np.array([labels[i] for i in best_idx])

        base_acc = accuracy_score(y_test, y_pred_known)
        base_f1 = f1_score(y_test, y_pred_known, average="macro")

        for threshold in np.arange(0.10, 0.95, 0.05):
            known_acc, known_f1 = evaluate_threshold(
                y_test,
                y_pred_known,
                best_scores,
                threshold
            )

            reject_rate = np.mean(best_scores < threshold)

            rows.append({
                "k": k,
                "threshold": round(float(threshold), 2),
                "base_accuracy_no_reject": base_acc,
                "base_macro_f1_no_reject": base_f1,
                "known_accuracy_after_reject": known_acc,
                "known_macro_f1_after_reject": known_f1,
                "reject_rate": reject_rate,
            })

        print(f"k={k} | no-reject acc={base_acc:.4f} | no-reject macro_f1={base_f1:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\nSaved:", OUT)
    print("\nTop rows:")
    print(df.head())


if __name__ == "__main__":
    main()
