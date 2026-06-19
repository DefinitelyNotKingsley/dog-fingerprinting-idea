from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
OUT = RESULTS_DIR / "holdout10_open_set_results.csv"
SEED = 42
K = 5

def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

def make_profiles(X, y, k):
    rng = np.random.default_rng(SEED)
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

X_train = normalize(np.load(RESULTS_DIR / "holdout10_train_embeddings.npy"))
X_known_test = normalize(np.load(RESULTS_DIR / "holdout10_known_test_embeddings.npy"))
X_unknown = normalize(np.load(RESULTS_DIR / "holdout10_unknown_test_embeddings.npy"))

y_train = np.load(DATA_DIR / "holdout10/known_train_y.npy", allow_pickle=True)
y_known_test = np.load(DATA_DIR / "holdout10/known_test_y.npy", allow_pickle=True)

labels, profile_mat = make_profiles(X_train, y_train, K)

X_eval = np.vstack([X_known_test, X_unknown])
y_eval = np.concatenate([y_known_test, np.array(["unknown"] * len(X_unknown))])

sims = cosine_similarity(X_eval, profile_mat)
best_idx = np.argmax(sims, axis=1)
best_scores = sims[np.arange(len(X_eval)), best_idx]
closest = np.array([labels[i] for i in best_idx])

rows = []
for threshold in np.arange(0.10, 0.95, 0.05):
    pred = np.where(best_scores < threshold, "unknown", closest)

    known_mask = y_eval != "unknown"
    unknown_mask = y_eval == "unknown"

    rows.append({
        "k": K,
        "threshold": round(float(threshold), 2),
        "overall_accuracy": accuracy_score(y_eval, pred),
        "known_id_accuracy": accuracy_score(y_eval[known_mask], pred[known_mask]),
        "unknown_recall": np.mean(pred[unknown_mask] == "unknown"),
        "false_unknown_rate": np.mean(pred[known_mask] == "unknown"),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("Saved:", OUT)
print(df.sort_values("overall_accuracy", ascending=False).head(10).to_string(index=False))
