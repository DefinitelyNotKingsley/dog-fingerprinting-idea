from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
OUT = RESULTS_DIR / "final_model_comparison_table.csv"

models = [
    ("Fusion3 + CenterLoss", "voxblink_centerloss_summary.csv"),
    ("Fusion3 + Triplet Loss", "triplet_fusion3_summary.csv"),
    ("DogECAPA-LDE", "dogecapa_lde_summary.csv"),
    ("DogECAPA-Tiny Softmax", "dogecapa_tiny_softmax_summary.csv"),
    ("DogECAPA-Tiny Stable AAM", "dogecapa_tiny_aam_stable_summary.csv"),
]

rows = []

for model_name, fname in models:
    path = RESULTS_DIR / fname
    if not path.exists():
        continue

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index = df.index.astype(str)

    row = df.loc["5"]

    def val(a, b):
        x = row[(a, b)]
        if hasattr(x, "iloc"):
            x = x.iloc[0]
        return float(x)

    rows.append({
        "Model": model_name,
        "Enrollment": "5 barks",
        "EER_mean_percent": val("EER_percent", "mean"),
        "EER_std_percent": val("EER_percent", "std"),
        "minDCF_mean": val("minDCF", "mean"),
        "minDCF_std": val("minDCF", "std"),
        "DIR@FAR=0.001_mean": val("DIR@FAR=0.001", "mean"),
        "DIR@FAR=0.01_mean": val("DIR@FAR=0.01", "mean"),
        "DIR@FAR=0.1_mean": val("DIR@FAR=0.1", "mean"),
    })

out = pd.DataFrame(rows)
out["EER_mean_percent"] = pd.to_numeric(out["EER_mean_percent"])
out = out.sort_values("EER_mean_percent")
out.to_csv(OUT, index=False)

print(out.to_string(index=False))
print("\nSaved:", OUT)
