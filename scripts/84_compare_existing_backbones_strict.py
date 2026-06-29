from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_RUNS = 100
UNKNOWN_DOGS_PER_RUN = 10
ENROLL_KS = [1, 3, 5]

BACKBONES = {
    # NOTE: arcface_fusion3 has identity leakage if it was trained on all 155 dogs.
    # Keep it here only as a diagnostic / upper-bound comparison, not as a strict final result.
    "arcface_fusion3_leaky_upper_bound": (
        RESULTS_DIR / "arcface_fusion3_train_embeddings.npy",
        RESULTS_DIR / "arcface_fusion3_test_embeddings.npy",
    ),

    "speechbrain_ecapa": (
        RESULTS_DIR / "speechbrain_ecapa_train_embeddings.npy",
        RESULTS_DIR / "speechbrain_ecapa_test_embeddings.npy",
    ),

    "dog_finetuned_ecapa": (
        RESULTS_DIR / "dog_finetuned_ecapa_train_embeddings.npy",
        RESULTS_DIR / "dog_finetuned_ecapa_test_embeddings.npy",
    ),

    "dogecapa": (
        RESULTS_DIR / "dogecapa_train_embeddings.npy",
        RESULTS_DIR / "dogecapa_test_embeddings.npy",
    ),

    "dogecapa_asp": (
        RESULTS_DIR / "dogecapa_asp_train_embeddings.npy",
        RESULTS_DIR / "dogecapa_asp_test_embeddings.npy",
    ),

    "dogecapa_tiny": (
        RESULTS_DIR / "dogecapa_tiny_train_embeddings.npy",
        RESULTS_DIR / "dogecapa_tiny_test_embeddings.npy",
    ),

    "fusion3_plus_speechbrain_ecapa": (
        DATA_DIR / "fusion3_plus_speechbrain_ecapa_train.npy",
        DATA_DIR / "fusion3_plus_speechbrain_ecapa_test.npy",
    ),
}

OUT_ALL = RESULTS_DIR / "existing_backbones_strict_all.csv"
OUT_SUMMARY = RESULTS_DIR / "existing_backbones_strict_summary.csv"


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def make_profiles(X_enroll, y_enroll, dogs, k, seed):
    rng = np.random.default_rng(seed)
    profiles = {}

    for dog in sorted(dogs):
        idx = np.where(y_enroll == dog)[0]
        idx = rng.permutation(idx)

        if len(idx) < k:
            continue

        chosen = idx[:k]
        profile = X_enroll[chosen].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-12)
        profiles[dog] = profile

    labels = list(profiles.keys())

    if len(labels) == 0:
        return [], np.empty((0, X_enroll.shape[1]), dtype="float32")

    mat = np.vstack([profiles[d] for d in labels])
    return labels, mat


def eval_backbone(X_train_full, X_test_full, y_train_full, y_test_full):
    all_dogs = np.array(sorted(set(y_train_full)))
    rows = []

    for run in range(N_RUNS):
        seed = 5000 + run
        rng = np.random.default_rng(seed)

        unseen_dogs = set(rng.choice(all_dogs, size=UNKNOWN_DOGS_PER_RUN, replace=False))

        unseen_train_mask = np.array([y in unseen_dogs for y in y_train_full])
        unseen_test_mask = np.array([y in unseen_dogs for y in y_test_full])

        X_enroll = X_train_full[unseen_train_mask]
        y_enroll = y_train_full[unseen_train_mask]

        X_probe_all = X_test_full[unseen_test_mask]
        y_probe_all = y_test_full[unseen_test_mask]

        for k in ENROLL_KS:
            profile_labels, profile_mat = make_profiles(
                X_enroll,
                y_enroll,
                unseen_dogs,
                k,
                seed + k,
            )

            if len(profile_labels) == 0:
                rows.append({
                    "run": run + 1,
                    "seed": seed,
                    "enroll_k": k,
                    "unseen_dogs": 0,
                    "test_barks": 0,
                    "accuracy": np.nan,
                    "mean_best_similarity": np.nan,
                    "std_best_similarity": np.nan,
                    "unseen_ids": ",".join(sorted(unseen_dogs)),
                })
                continue

            valid = np.array([y in set(profile_labels) for y in y_probe_all])
            X_probe = X_probe_all[valid]
            y_probe = y_probe_all[valid]

            sims = cosine_similarity(X_probe, profile_mat)
            best_idx = np.argmax(sims, axis=1)
            best_scores = sims[np.arange(len(X_probe)), best_idx]
            pred = np.array([profile_labels[i] for i in best_idx])

            acc = accuracy_score(y_probe, pred)

            rows.append({
                "run": run + 1,
                "seed": seed,
                "enroll_k": k,
                "unseen_dogs": len(profile_labels),
                "test_barks": len(y_probe),
                "accuracy": acc,
                "mean_best_similarity": float(np.mean(best_scores)),
                "std_best_similarity": float(np.std(best_scores)),
                "unseen_ids": ",".join(sorted(unseen_dogs)),
            })

    return rows


def main():
    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    all_rows = []

    for name, (train_path, test_path) in BACKBONES.items():
        if not train_path.exists() or not test_path.exists():
            print(f"Skipping {name}: missing files")
            continue

        print("\n==============================")
        print("Backbone:", name)

        X_train = normalize(np.load(train_path).astype("float32"))
        X_test = normalize(np.load(test_path).astype("float32"))

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)

        rows = eval_backbone(X_train, X_test, y_train, y_test)

        for r in rows:
            r["backbone"] = name

        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_ALL, index=False)

    df = pd.DataFrame(all_rows)

    summary = (
        df.groupby(["backbone", "enroll_k"])
        [["accuracy", "test_barks", "mean_best_similarity"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nExisting backbone strict summary:")
    print(summary.to_string(index=False))

    print("\nSaved:", OUT_ALL)
    print("Saved:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
