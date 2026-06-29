import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("results/full_eval_suite")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_PATH = OUT_DIR / "confusion_predictions_all.csv"
PER_DOG_PATH = OUT_DIR / "per_dog_accuracy_all.csv"

OUT_BREED_ACC = OUT_DIR / "breed_accuracy_summary.csv"
OUT_BREED_CONF = OUT_DIR / "breed_confusion_matrix.csv"
OUT_BREED_CONF_NORM = OUT_DIR / "breed_confusion_matrix_normalized.csv"
OUT_ERROR_TYPE = OUT_DIR / "same_vs_cross_breed_errors.csv"
OUT_TOP_PAIRS = OUT_DIR / "top_confused_pairs_with_breed.csv"
OUT_DOG_ACC = OUT_DIR / "per_dog_accuracy_with_breed.csv"


def normalize_dog_id(x):
    if x in ["dog_7a", "dog_7b"]:
        return "dog_7"
    return str(x)


def extract_breed_from_path(path):
    name = Path(str(path)).name.lower()

    # Example:
    # 62390_shibainu_F_dog_111.wav
    # 34559_husky_F_dog_26.wav
    # 1497_husky_M_dog_5.wav
    # 49504_shibainu_M_dog_87.wav

    if "shibainu" in name or "shiba" in name:
        return "Shiba Inu"
    if "husky" in name:
        return "Husky"
    if "chihuahua" in name:
        return "Chihuahua"
    if "germanshepherd" in name or "german_shepherd" in name:
        return "German Shepherd"
    if "pitbull" in name or "pit_bull" in name:
        return "Pitbull"

    return "Unknown"


def build_dog_to_breed():
    train_files = np.load(DATA_DIR / "train_files.npy", allow_pickle=True)
    test_files = np.load(DATA_DIR / "test_files.npy", allow_pickle=True)

    y_train = np.load(DATA_DIR / "train_labels.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "test_labels.npy", allow_pickle=True)

    rows = []

    for dog, f in zip(y_train, train_files):
        rows.append({
            "dog_id": normalize_dog_id(dog),
            "breed": extract_breed_from_path(f),
            "file": str(f),
        })

    for dog, f in zip(y_test, test_files):
        rows.append({
            "dog_id": normalize_dog_id(dog),
            "breed": extract_breed_from_path(f),
            "file": str(f),
        })

    df = pd.DataFrame(rows)

    # Majority breed per dog
    dog_breed = (
        df.groupby(["dog_id", "breed"])
        .size()
        .reset_index(name="count")
        .sort_values(["dog_id", "count"], ascending=[True, False])
        .drop_duplicates("dog_id")
        [["dog_id", "breed"]]
    )

    return dict(zip(dog_breed["dog_id"], dog_breed["breed"]))


def main():
    dog_to_breed = build_dog_to_breed()

    conf = pd.read_csv(CONF_PATH)
    per_dog = pd.read_csv(PER_DOG_PATH)

    conf["true_dog"] = conf["true_dog"].map(normalize_dog_id)
    conf["pred_dog"] = conf["pred_dog"].map(normalize_dog_id)

    conf["true_breed"] = conf["true_dog"].map(dog_to_breed).fillna("Unknown")
    conf["pred_breed"] = conf["pred_dog"].map(dog_to_breed).fillna("Unknown")

    conf["correct"] = conf["true_dog"] == conf["pred_dog"]
    conf["same_breed_error"] = (~conf["correct"]) & (conf["true_breed"] == conf["pred_breed"])
    conf["cross_breed_error"] = (~conf["correct"]) & (conf["true_breed"] != conf["pred_breed"])

    # Breed accuracy summary by loss/enrollment/breed
    breed_acc = (
        conf.groupby(["loss", "enroll_k", "true_breed"])
        .agg(
            num_predictions=("correct", "size"),
            rank1_accuracy=("correct", "mean"),
            mean_score=("score", "mean"),
            num_dogs=("true_dog", "nunique"),
        )
        .reset_index()
        .sort_values(["loss", "enroll_k", "rank1_accuracy"], ascending=[True, True, False])
    )
    breed_acc.to_csv(OUT_BREED_ACC, index=False)

    # Same-breed vs cross-breed error rate
    error_rows = []

    for (loss, k), g in conf.groupby(["loss", "enroll_k"]):
        total = len(g)
        correct = g["correct"].sum()
        errors = total - correct

        same_breed_errors = g["same_breed_error"].sum()
        cross_breed_errors = g["cross_breed_error"].sum()

        error_rows.append({
            "loss": loss,
            "enroll_k": k,
            "total_predictions": total,
            "correct": int(correct),
            "errors": int(errors),
            "rank1_accuracy": correct / total if total else np.nan,
            "same_breed_errors": int(same_breed_errors),
            "cross_breed_errors": int(cross_breed_errors),
            "same_breed_error_fraction_of_errors": same_breed_errors / errors if errors else np.nan,
            "cross_breed_error_fraction_of_errors": cross_breed_errors / errors if errors else np.nan,
        })

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(OUT_ERROR_TYPE, index=False)

    # Breed confusion matrix, aggregated across all runs
    breed_conf = (
        conf.groupby(["loss", "enroll_k", "true_breed", "pred_breed"])
        .size()
        .reset_index(name="count")
    )
    breed_conf.to_csv(OUT_BREED_CONF, index=False)

    # Normalized breed confusion matrix
    breed_conf_norm = breed_conf.copy()
    breed_conf_norm["row_total"] = breed_conf_norm.groupby(
        ["loss", "enroll_k", "true_breed"]
    )["count"].transform("sum")
    breed_conf_norm["fraction"] = breed_conf_norm["count"] / breed_conf_norm["row_total"]
    breed_conf_norm.to_csv(OUT_BREED_CONF_NORM, index=False)

    # Top confused dog pairs with breed information
    mistakes = conf[conf["true_dog"] != conf["pred_dog"]].copy()

    top_pairs = (
        mistakes.groupby(["loss", "enroll_k", "true_dog", "pred_dog", "true_breed", "pred_breed"])
        .agg(
            count=("score", "size"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["loss", "enroll_k", "count"], ascending=[True, True, False])
    )
    top_pairs.to_csv(OUT_TOP_PAIRS, index=False)

    # Per-dog accuracy with breed
    per_dog["dog_id"] = per_dog["dog_id"].map(normalize_dog_id)
    per_dog["breed"] = per_dog["dog_id"].map(dog_to_breed).fillna("Unknown")

    per_dog_summary = (
        per_dog.groupby(["loss", "enroll_k", "dog_id", "breed"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_num_probes=("num_probes", "mean"),
            mean_best_score=("mean_best_score", "mean"),
        )
        .reset_index()
        .sort_values(["loss", "enroll_k", "mean_accuracy"], ascending=[True, True, True])
    )
    per_dog_summary.to_csv(OUT_DOG_ACC, index=False)

    print("\nSaved:")
    print(OUT_BREED_ACC)
    print(OUT_BREED_CONF)
    print(OUT_BREED_CONF_NORM)
    print(OUT_ERROR_TYPE)
    print(OUT_TOP_PAIRS)
    print(OUT_DOG_ACC)

    print("\nBreed accuracy summary:")
    print(breed_acc.to_string(index=False))

    print("\nSame vs cross breed errors:")
    print(error_df.to_string(index=False))

    print("\nTop confused pairs:")
    print(top_pairs.head(30).to_string(index=False))

    print("\nHardest dogs:")
    print(per_dog_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
