from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

OUT = RESULTS_DIR / "attention_fingerprint_results.csv"

SEED = 42


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def evaluate(X_train, y_train, X_test, y_test, k):
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

    profile_labels = list(profiles.keys())
    profile_matrix = np.vstack([profiles[d] for d in profile_labels])

    sims = cosine_similarity(X_test, profile_matrix)

    pred_idx = np.argmax(sims, axis=1)

    y_pred = np.array([profile_labels[i] for i in pred_idx])

    return {
        "k": k,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }


def main():

    X_train = np.load(
        RESULTS_DIR / "attention_train_embeddings.npy"
    )

    X_test = np.load(
        RESULTS_DIR / "attention_test_embeddings.npy"
    )

    y_train = np.load(
        DATA_DIR / "attention_train_labels.npy",
        allow_pickle=True
    )

    y_test = np.load(
        DATA_DIR / "attention_test_labels.npy",
        allow_pickle=True
    )

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    rows = []

    for k in [1,3,5,10,20]:

        r = evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            k
        )

        rows.append(r)

        print(
            f"k={k} | "
            f"acc={r['accuracy']:.4f} | "
            f"macro_f1={r['macro_f1']:.4f}"
        )

    pd.DataFrame(rows).to_csv(
        OUT,
        index=False
    )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
