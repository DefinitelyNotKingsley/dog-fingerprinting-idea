from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

OUT_DIR = RESULTS_DIR / "confusion_analysis"
OUT_DIR.mkdir(exist_ok=True)

ENROLL_K = 5
THRESHOLD = 0.65
UNKNOWN_DOGS = 10
SEED = 1000
TOP_N = 30

def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

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

X_train = normalize(np.load(RESULTS_DIR / "triplet_fusion3_train_embeddings.npy"))
X_test = normalize(np.load(RESULTS_DIR / "triplet_fusion3_test_embeddings.npy"))

y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

all_dogs = np.array(sorted(set(y_train)))
rng = np.random.default_rng(SEED)

unknown_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS, replace=False))
known_dogs = set(all_dogs) - unknown_dogs

profile_labels, profile_mat = make_profiles(
    X_train,
    y_train,
    known_dogs,
    ENROLL_K,
    SEED,
)

known_mask = np.array([y in known_dogs for y in y_test])
unknown_mask = np.array([y in unknown_dogs for y in y_test])

X_known = X_test[known_mask]
y_known = y_test[known_mask]

X_unknown = X_test[unknown_mask]
y_unknown = y_test[unknown_mask]

known_sims = cosine_similarity(X_known, profile_mat)
known_best_idx = known_sims.argmax(axis=1)
known_scores = known_sims[np.arange(len(X_known)), known_best_idx]
known_pred_raw = np.array([profile_labels[i] for i in known_best_idx])
known_pred = np.where(known_scores >= THRESHOLD, known_pred_raw, "unknown")

unknown_sims = cosine_similarity(X_unknown, profile_mat)
unknown_best_idx = unknown_sims.argmax(axis=1)
unknown_scores = unknown_sims[np.arange(len(X_unknown)), unknown_best_idx]
unknown_pred_raw = np.array([profile_labels[i] for i in unknown_best_idx])
unknown_pred = np.where(unknown_scores >= THRESHOLD, unknown_pred_raw, "unknown")

# Open-set summary
known_correct = np.sum(known_pred == y_known)
known_wrong_known = np.sum((known_pred != y_known) & (known_pred != "unknown"))
known_rejected = np.sum(known_pred == "unknown")

unknown_correct_reject = np.sum(unknown_pred == "unknown")
unknown_false_accept = np.sum(unknown_pred != "unknown")

summary = pd.DataFrame([
    {
        "Group": "Known dog queries",
        "Correct known ID": known_correct,
        "Wrong known ID": known_wrong_known,
        "Rejected as unknown": known_rejected,
        "Total": len(y_known),
    },
    {
        "Group": "Unknown dog queries",
        "Correct known ID": 0,
        "Wrong known ID": unknown_false_accept,
        "Rejected as unknown": unknown_correct_reject,
        "Total": len(y_unknown),
    },
])

summary.to_csv(OUT_DIR / "open_set_confusion_summary.csv", index=False)
print("\nOpen-set confusion summary:")
print(summary.to_string(index=False))

# Known-only dog confusion matrix, top N by support
counts = pd.Series(y_known).value_counts()
top_dogs = counts.head(TOP_N).index.tolist()

mask_top = np.array([y in top_dogs and p in top_dogs for y, p in zip(y_known, known_pred_raw)])

y_top = y_known[mask_top]
p_top = known_pred_raw[mask_top]

cm = confusion_matrix(y_top, p_top, labels=top_dogs)

cm_norm = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm_norm / row_sums

plt.figure(figsize=(12, 10))
plt.imshow(cm_norm, aspect="auto")
plt.colorbar(label="Row-normalized proportion")
plt.xticks(range(len(top_dogs)), top_dogs, rotation=90, fontsize=6)
plt.yticks(range(len(top_dogs)), top_dogs, fontsize=6)
plt.xlabel("Predicted dog")
plt.ylabel("True dog")
plt.title(f"Known-Dog Confusion Matrix | Top {TOP_N} Dogs | k={ENROLL_K} | threshold={THRESHOLD}")
plt.tight_layout()
plt.savefig(OUT_DIR / "known_dog_confusion_matrix_top30.png", dpi=300)
plt.close()

# Top confusions
confusions = []
for true_dog in top_dogs:
    for pred_dog in top_dogs:
        if true_dog == pred_dog:
            continue
        n = np.sum((y_top == true_dog) & (p_top == pred_dog))
        if n > 0:
            confusions.append({
                "true_dog": true_dog,
                "predicted_dog": pred_dog,
                "count": int(n),
            })

conf_df = pd.DataFrame(confusions).sort_values("count", ascending=False)
conf_df.to_csv(OUT_DIR / "top_known_dog_confusions.csv", index=False)

print("\nSaved:")
print(OUT_DIR / "open_set_confusion_summary.csv")
print(OUT_DIR / "known_dog_confusion_matrix_top30.png")
print(OUT_DIR / "top_known_dog_confusions.csv")

print("\nTop confusions:")
print(conf_df.head(20).to_string(index=False))
