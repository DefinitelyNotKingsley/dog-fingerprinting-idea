from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path("results/full_eval_suite")
PAPER_DIR = Path("results/paper_tables")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_PATH = Path("results/existing_backbones_strict_summary.csv")
LOSS_ACC_PATH = Path("results/final_100run_fusion3_best_losses_summary.csv")
OPENSET_PATH = OUT_DIR / "master_metrics_summary.csv"
CONF_PATH = OUT_DIR / "confusion_predictions_all.csv"
BREED_ACC_PATH = OUT_DIR / "breed_accuracy_summary.csv"
SAME_CROSS_PATH = OUT_DIR / "same_vs_cross_breed_errors.csv"


def clean_summary_csv(path):
    df = pd.read_csv(path)

    # Drop weird header row created by pandas multi-index export
    if len(df) > 0:
        first = df.iloc[0].astype(str).str.lower().tolist()
        if "mean" in first or "std" in first:
            df = df.iloc[1:].copy()

    # Convert numeric columns where possible
    for c in df.columns:
        
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass


    return df


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100 * float(x):.1f}"


def num(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def make_table1_overall():
    rows = []

    # Backbone cosine/direct rows
    if BACKBONE_PATH.exists():
        b = clean_summary_csv(BACKBONE_PATH)

        # columns:
        # backbone, enroll_k, accuracy, accuracy.1, ...
        for backbone in sorted(b["backbone"].dropna().unique()):
            bb = b[b["backbone"] == backbone].copy()

            def get_acc(k):
                r = bb[bb["enroll_k"].astype(float) == float(k)]
                if len(r) == 0:
                    return np.nan
                return float(r.iloc[0]["accuracy"])

            if "leaky" in backbone:
                notes = "Upper bound; trained with identity leakage"
                valid = "No"
            else:
                notes = "Direct cosine evaluation"
                valid = "Yes"

            feature_name = {
                "arcface_fusion3_leaky_upper_bound": "ArcFace Fusion3 embedding",
                "speechbrain_ecapa": "SpeechBrain ECAPA",
                "dog_finetuned_ecapa": "Dog-finetuned ECAPA",
                "dogecapa": "DogECAPA",
                "dogecapa_asp": "DogECAPA-ASP",
                "dogecapa_tiny": "DogECAPA-Tiny",
                "fusion3_plus_speechbrain_ecapa": "Fusion3 + SpeechBrain ECAPA",
            }.get(backbone, backbone)

            rows.append({
                "Experiment": backbone,
                "Features": feature_name,
                "Loss / Scoring": "Cosine",
                "Strict Valid?": valid,
                "Rank-1 1 Bark (%)": pct(get_acc(1)),
                "Rank-1 3 Bark (%)": pct(get_acc(3)),
                "Rank-1 5 Bark (%)": pct(get_acc(5)),
                "EER 5 Bark (%)": "NA",
                "minDCF 5 Bark": "NA",
                "DIR@FAR=1% 5 Bark (%)": "NA",
                "Notes": notes,
            })

    # Clean Fusion3 loss rows, rank-1 from 100-run closed identification
    loss_rank = {}
    if LOSS_ACC_PATH.exists():
        ldf = clean_summary_csv(LOSS_ACC_PATH)

        for loss in sorted(ldf["loss"].dropna().unique()):
            sub = ldf[ldf["loss"] == loss]
            loss_rank[loss] = {}
            for k in [1, 3, 5]:
                r = sub[sub["enroll_k"].astype(float) == float(k)]
                loss_rank[loss][k] = float(r.iloc[0]["accuracy"]) if len(r) else np.nan

    # Full open-set metrics for k=5
    open_metrics = {}
    if OPENSET_PATH.exists():
        odf = clean_summary_csv(OPENSET_PATH)

        for loss in sorted(odf["loss"].dropna().unique()):
            sub = odf[(odf["loss"] == loss) & (odf["enroll_k"].astype(float) == 5.0)]
            if len(sub):
                r = sub.iloc[0]
                open_metrics[loss] = {
                    "eer": float(r["EER_percent"]),
                    "mindcf": float(r["minDCF"]),
                    "dir01": float(r["DIR@FAR=0.01"]),
                }

    pretty_loss = {
        "triplet": "Triplet",
        "supcon": "Supervised Contrastive",
        "proxy_anchor": "Proxy Anchor",
    }

    for loss in ["triplet", "supcon", "proxy_anchor"]:
        rows.append({
            "Experiment": f"Fusion3 + {pretty_loss.get(loss, loss)}",
            "Features": "Dog2Vec + MFCC + ECAPA",
            "Loss / Scoring": pretty_loss.get(loss, loss),
            "Strict Valid?": "Yes",
            "Rank-1 1 Bark (%)": pct(loss_rank.get(loss, {}).get(1, np.nan)),
            "Rank-1 3 Bark (%)": pct(loss_rank.get(loss, {}).get(3, np.nan)),
            "Rank-1 5 Bark (%)": pct(loss_rank.get(loss, {}).get(5, np.nan)),
            "EER 5 Bark (%)": num(open_metrics.get(loss, {}).get("eer", np.nan), 2),
            "minDCF 5 Bark": num(open_metrics.get(loss, {}).get("mindcf", np.nan), 3),
            "DIR@FAR=1% 5 Bark (%)": pct(open_metrics.get(loss, {}).get("dir01", np.nan)),
            "Notes": "Strict unseen-dog metric learning",
        })

    table = pd.DataFrame(rows)
    table.to_csv(PAPER_DIR / "paper_table1_overall_results.csv", index=False)
    return table


def make_table2_stable_confusions(top_n=30):
    conf = pd.read_csv(CONF_PATH)

    conf = conf[conf["true_dog"] != conf["pred_dog"]].copy()

    # Add breed info if not already present
    if "true_breed" not in conf.columns or "pred_breed" not in conf.columns:
        # Use previously created top-pair file if available for dog->breed mapping
        breed_map = {}
        top_path = OUT_DIR / "top_confused_pairs_with_breed.csv"
        if top_path.exists():
            top = pd.read_csv(top_path)
            for _, r in top.iterrows():
                breed_map[str(r["true_dog"])] = r["true_breed"]
                breed_map[str(r["pred_dog"])] = r["pred_breed"]

        conf["true_breed"] = conf["true_dog"].map(breed_map).fillna("Unknown")
        conf["pred_breed"] = conf["pred_dog"].map(breed_map).fillna("Unknown")

    # Count errors per run/pair
    per_run_pair = (
        conf.groupby(["loss", "enroll_k", "run", "true_dog", "pred_dog", "true_breed", "pred_breed"])
        .agg(
            errors_in_run=("score", "size"),
            mean_score_in_run=("score", "mean"),
        )
        .reset_index()
    )

    stable = (
        per_run_pair.groupby(["loss", "enroll_k", "true_dog", "pred_dog", "true_breed", "pred_breed"])
        .agg(
            runs_appeared=("run", "nunique"),
            avg_errors_per_appearing_run=("errors_in_run", "mean"),
            std_errors_per_appearing_run=("errors_in_run", "std"),
            max_errors_in_single_run=("errors_in_run", "max"),
            total_errors=("errors_in_run", "sum"),
            avg_score=("mean_score_in_run", "mean"),
        )
        .reset_index()
    )

    total_runs = conf["run"].nunique()
    stable["total_runs"] = total_runs
    stable["run_appearance_rate"] = stable["runs_appeared"] / total_runs
    stable["same_breed"] = stable["true_breed"] == stable["pred_breed"]
    stable["breed_relation"] = np.where(stable["same_breed"], "Same breed", "Different breed")

    stable = stable.sort_values(
        ["loss", "enroll_k", "runs_appeared", "avg_errors_per_appearing_run", "total_errors"],
        ascending=[True, True, False, False, False],
    )

    stable.to_csv(PAPER_DIR / "paper_table2_stable_confusions_all.csv", index=False)

    # Presentation-focused table: usually show Proxy Anchor k=5
    proxy5 = stable[(stable["loss"] == "proxy_anchor") & (stable["enroll_k"].astype(float) == 5.0)].copy()
    proxy5 = proxy5.head(top_n)

    proxy5_out = proxy5[[
        "true_dog",
        "pred_dog",
        "true_breed",
        "pred_breed",
        "breed_relation",
        "runs_appeared",
        "total_runs",
        "run_appearance_rate",
        "avg_errors_per_appearing_run",
        "max_errors_in_single_run",
        "total_errors",
        "avg_score",
    ]].copy()

    proxy5_out.rename(columns={
        "true_dog": "True Dog",
        "pred_dog": "Predicted Dog",
        "true_breed": "True Breed",
        "pred_breed": "Predicted Breed",
        "breed_relation": "Same/Different Breed",
        "runs_appeared": "Runs Appeared",
        "total_runs": "Total Runs",
        "run_appearance_rate": "Run Appearance Rate",
        "avg_errors_per_appearing_run": "Avg Errors / Appearing Run",
        "max_errors_in_single_run": "Max Errors in One Run",
        "total_errors": "Total Errors",
        "avg_score": "Avg Similarity Score",
    }, inplace=True)

    proxy5_out.to_csv(PAPER_DIR / "paper_table2_stable_confusions_proxy_anchor_k5.csv", index=False)
    return stable, proxy5_out


def make_table3_breed():
    if not BREED_ACC_PATH.exists():
        return None

    df = pd.read_csv(BREED_ACC_PATH)
    out = df[(df["loss"] == "proxy_anchor") & (df["enroll_k"].astype(float) == 5.0)].copy()
    out = out.sort_values("rank1_accuracy", ascending=False)
    out["Rank-1 Accuracy (%)"] = out["rank1_accuracy"].apply(lambda x: f"{100*x:.1f}")
    out = out[["true_breed", "num_dogs", "num_predictions", "Rank-1 Accuracy (%)", "mean_score"]]
    out.rename(columns={
        "true_breed": "Breed",
        "num_dogs": "# Dogs",
        "num_predictions": "# Predictions",
        "mean_score": "Mean Similarity Score",
    }, inplace=True)
    out.to_csv(PAPER_DIR / "paper_table3_breed_accuracy_proxy_anchor_k5.csv", index=False)
    return out


def make_table4_same_cross():
    if not SAME_CROSS_PATH.exists():
        return None

    df = pd.read_csv(SAME_CROSS_PATH)
    out = df[(df["loss"] == "proxy_anchor") & (df["enroll_k"].astype(float) == 5.0)].copy()
    out["Same Breed Error (%)"] = out["same_breed_error_fraction_of_errors"].apply(lambda x: f"{100*x:.1f}")
    out["Cross Breed Error (%)"] = out["cross_breed_error_fraction_of_errors"].apply(lambda x: f"{100*x:.1f}")
    out = out[[
        "loss",
        "enroll_k",
        "errors",
        "same_breed_errors",
        "cross_breed_errors",
        "Same Breed Error (%)",
        "Cross Breed Error (%)",
    ]]
    out.rename(columns={
        "loss": "Loss",
        "enroll_k": "Enrollment",
        "errors": "Total Errors",
        "same_breed_errors": "Same Breed Errors",
        "cross_breed_errors": "Cross Breed Errors",
    }, inplace=True)
    out.to_csv(PAPER_DIR / "paper_table4_same_vs_cross_breed_proxy_anchor_k5.csv", index=False)
    return out


def main():
    t1 = make_table1_overall()
    stable_all, t2 = make_table2_stable_confusions(top_n=30)
    t3 = make_table3_breed()
    t4 = make_table4_same_cross()

    print("\nSaved paper tables in:", PAPER_DIR)

    print("\nTABLE 1: Overall Results")
    print(t1.to_string(index=False))

    print("\nTABLE 2: Stable Confusions, Proxy Anchor k=5")
    print(t2.to_string(index=False))

    if t3 is not None:
        print("\nTABLE 3: Breed Accuracy, Proxy Anchor k=5")
        print(t3.to_string(index=False))

    if t4 is not None:
        print("\nTABLE 4: Same vs Cross Breed Errors, Proxy Anchor k=5")
        print(t4.to_string(index=False))

    print("\nFiles:")
    for p in sorted(PAPER_DIR.glob("*.csv")):
        print(p)


if __name__ == "__main__":
    main()
