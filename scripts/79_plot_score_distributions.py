from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

X_train = np.load(RESULTS_DIR / "triplet_fusion3_train_embeddings.npy")
X_test = np.load(RESULTS_DIR / "triplet_fusion3_test_embeddings.npy")
y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

X_train = normalize(X_train)
X_test = normalize(X_test)

def make_profiles(X, y, known_dogs, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}
    for dog in sorted(known_dogs):
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

def collect_scores(unknown_count, enroll_k=5, seed=1000):
    rng = np.random.default_rng(seed)
    all_dogs = np.array(sorted(set(y_train)))

    unknown_dogs = set(rng.choice(all_dogs, size=unknown_count, replace=False))
    known_dogs = set(all_dogs) - unknown_dogs

    profile_labels, profile_mat = make_profiles(
        X_train,
        y_train,
        known_dogs,
        enroll_k,
        seed,
    )

    known_mask = np.array([y in known_dogs for y in y_test])
    unknown_mask = np.array([y in unknown_dogs for y in y_test])

    X_known = X_test[known_mask]
    X_unknown = X_test[unknown_mask]

    known_scores = cosine_similarity(X_known, profile_mat).max(axis=1)
    unknown_scores = cosine_similarity(X_unknown, profile_mat).max(axis=1)

    return known_scores, unknown_scores

configs = [
    {
        "title": "145 Known / 10 Unknown",
        "unknown_count": 10,
        "threshold": 0.65,
        "outfile": "score_dist_145known_10unknown.png",
    },
    {
        "title": "100 Known / 55 Unknown",
        "unknown_count": 55,
        "threshold": 0.85,
        "outfile": "score_dist_100known_55unknown.png",
    },
    {
        "title": "55 Known / 100 Unknown",
        "unknown_count": 100,
        "threshold": 0.90,
        "outfile": "score_dist_55known_100unknown.png",
    },
]

for cfg in configs:
    known_scores, unknown_scores = collect_scores(
        cfg["unknown_count"],
        enroll_k=5,
        seed=1000,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(known_scores, bins=50, alpha=0.6, density=True, label="Known dog queries")
    plt.hist(unknown_scores, bins=50, alpha=0.6, density=True, label="Unknown dog queries")
    plt.axvline(cfg["threshold"], linestyle="--", linewidth=2, label=f"Threshold = {cfg['threshold']}")

    plt.title(cfg["title"])
    plt.xlabel("Maximum cosine similarity to enrolled dog profiles")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    out = RESULTS_DIR / cfg["outfile"]
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)
    print("Known score mean/std:", known_scores.mean(), known_scores.std())
    print("Unknown score mean/std:", unknown_scores.mean(), unknown_scores.std())
    print()

print("Done.")
