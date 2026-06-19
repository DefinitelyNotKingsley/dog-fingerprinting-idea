from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "fingerprint_eval_fusion3_results.csv"
SEED = 42


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def eval_k(X_train, y_train, X_test, y_test, k):
    rng = np.random.default_rng(SEED)
    profiles = {}

    for dog in sorted(set(y_train)):
        idx = np.where(y_train == dog)[0]
        if len(idx) < k:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        profiles[dog] = normalize(X_train[chosen].mean(axis=0, keepdims=True))[0]

    labels = list(profiles.keys())
    profile_mat = np.vstack([profiles[d] for d in labels])

    sims = cosine_similarity(X_test, profile_mat)
    pred = np.array([labels[i] for i in np.argmax(sims, axis=1)])

    return {
        "enrollment_barks": k,
        "num_profiles": len(labels),
        "num_test": len(y_test),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
    }


def main():
    X_train = np.load(RESULTS_DIR / "arcface_fusion3_train_embeddings.npy")
    X_test = np.load(RESULTS_DIR / "arcface_fusion3_test_embeddings.npy")
    y_train = np.load(DATA_DIR / "fusion3_train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "fusion3_test_labels.npy", allow_pickle=True)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    rows = []
    for k in [1, 3, 5, 10, 20]:
        r = eval_k(X_train, y_train, X_test, y_test, k)
        rows.append(r)
        print(f"k={k} | acc={r['accuracy']:.4f} | macro_f1={r['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
