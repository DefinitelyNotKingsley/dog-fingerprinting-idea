from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
IN_PATH = RESULTS_DIR / "voxblink_centerloss_all_results.csv"
OUT_PATH = RESULTS_DIR / "voxblink_centerloss_final_table.csv"

df = pd.read_csv(IN_PATH)

metric_cols = [
    "EER_percent",
    "minDCF",
    "DIR@FAR=0.001",
    "DIR@FAR=0.01",
    "DIR@FAR=0.1",
]

summary = (
    df.groupby("enroll_k")[metric_cols]
    .agg(["mean", "std"])
    .reset_index()
)

summary.columns = [
    "enroll_k",
    "EER_mean_percent",
    "EER_std_percent",
    "minDCF_mean",
    "minDCF_std",
    "DIR@FAR=0.001_mean",
    "DIR@FAR=0.001_std",
    "DIR@FAR=0.01_mean",
    "DIR@FAR=0.01_std",
    "DIR@FAR=0.1_mean",
    "DIR@FAR=0.1_std",
]

for col in summary.columns:
    if col != "enroll_k":
        summary[col] = summary[col].round(4)

summary["Enrollment"] = summary["enroll_k"].astype(str) + " bark(s)"

final = summary[
    [
        "Enrollment",
        "EER_mean_percent",
        "EER_std_percent",
        "minDCF_mean",
        "minDCF_std",
        "DIR@FAR=0.001_mean",
        "DIR@FAR=0.01_mean",
        "DIR@FAR=0.1_mean",
    ]
]

final.to_csv(OUT_PATH, index=False)

print("\nVoxBlink-style + Center Loss final table:")
print(final.to_string(index=False))
print(f"\nSaved: {OUT_PATH}")
