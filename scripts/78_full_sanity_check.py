import pandas as pd
from pathlib import Path

FILES = {
    "145 known / 10 unknown": "results/triplet_fusion3_100splits_all.csv",
    "100 known / 55 unknown": "results/triplet_fusion3_55unknown_all.csv",
    "55 known / 100 unknown": "results/triplet_fusion3_100unknown_all.csv",
}

for name, path in FILES.items():
    print("\n" + "="*70)
    print(name)
    print("="*70)

    df = pd.read_csv(path)

    print("file:", path)
    print("rows:", len(df))
    print("runs:", df["run"].nunique())
    print("enroll_k:", sorted(df["enroll_k"].unique()))
    print("thresholds:", sorted(df["threshold"].unique()))
    print("expected rows:", 100 * 3 * len(df["threshold"].unique()))

    print("\nRows per enrollment:")
    print(df.groupby("enroll_k").size())

    # best threshold per enrollment
    best = (
        df.groupby(["enroll_k", "threshold"])["overall_accuracy"]
        .mean()
        .reset_index()
        .sort_values(["enroll_k", "overall_accuracy"], ascending=[True, False])
        .groupby("enroll_k")
        .head(1)
    )

    print("\nBest threshold by overall accuracy:")
    print(best.to_string(index=False))

    # threshold-dependent metrics at best threshold
    print("\nMetrics at best threshold:")
    rows = []
    for _, r in best.iterrows():
        k = r["enroll_k"]
        th = r["threshold"]
        sub = df[(df["enroll_k"] == k) & (df["threshold"] == th)]

        rows.append({
            "enroll_k": k,
            "threshold": th,
            "overall_acc": f"{100*sub['overall_accuracy'].mean():.2f} ± {100*sub['overall_accuracy'].std():.2f}",
            "known_acc": f"{100*sub['known_id_accuracy'].mean():.2f} ± {100*sub['known_id_accuracy'].std():.2f}",
            "unknown_recall": f"{100*sub['unknown_recall'].mean():.2f} ± {100*sub['unknown_recall'].std():.2f}",
            "false_unknown": f"{100*sub['false_unknown_rate'].mean():.2f} ± {100*sub['false_unknown_rate'].std():.2f}",
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # threshold-independent metrics
    base = df.drop_duplicates(["run", "enroll_k"])

    global_summary = (
        base.groupby("enroll_k")
        [["EER_percent", "minDCF", "DIR@FAR=0.001", "DIR@FAR=0.01", "DIR@FAR=0.1"]]
        .agg(["mean", "std"])
    )

    print("\nGlobal metrics:")
    pd.set_option("display.max_columns", None)
    print(global_summary)

    # sanity checks
    print("\nSanity checks:")

    # DIR monotonic
    dir_ok = (
        (base["DIR@FAR=0.001"] <= base["DIR@FAR=0.01"]) &
        (base["DIR@FAR=0.01"] <= base["DIR@FAR=0.1"])
    ).all()
    print("DIR monotonic 0.001 <= 0.01 <= 0.1:", dir_ok)

    # enrollment EER trend
    eer_means = base.groupby("enroll_k")["EER_percent"].mean()
    eer_ok = eer_means.loc[5] <= eer_means.loc[3] <= eer_means.loc[1]
    print("EER improves with enrollment 5 <= 3 <= 1:", eer_ok)

    # enrollment DIR@0.1 trend
    dir_means = base.groupby("enroll_k")["DIR@FAR=0.1"].mean()
    dir_ok2 = dir_means.loc[5] >= dir_means.loc[3] >= dir_means.loc[1]
    print("DIR@0.1 improves with enrollment 5 >= 3 >= 1:", dir_ok2)

    # threshold behavior for 5-bark
    k5 = df[df["enroll_k"] == 5]
    threshold_curve = k5.groupby("threshold")[["known_id_accuracy", "unknown_recall", "false_unknown_rate"]].mean()

    print("\n5-bark threshold behavior:")
    print(threshold_curve)

    out = Path(path).with_name(Path(path).stem + "_sanity_summary.csv")
    global_summary.to_csv(out)
    print("\nSaved:", out)

print("\nDONE")
