from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

files = {
    "ArcFace": RESULTS_DIR / "voxblink_style_final_table.csv",
    "ArcFace + Center Loss": RESULTS_DIR / "voxblink_centerloss_final_table.csv",
    "Attention Fusion + Center Loss": RESULTS_DIR / "voxblink_attention_centerloss_final_table.csv",
}

rows = []

for model_name, path in files.items():
    if not path.exists():
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)

    for _, r in df.iterrows():
        rows.append({
            "Model": model_name,
            "Enrollment": r["Enrollment"],
            "EER (%) ↓": r["EER_mean_percent"],
            "minDCF ↓": r["minDCF_mean"],
            "DIR@FAR=0.001 ↑": r["DIR@FAR=0.001_mean"],
            "DIR@FAR=0.01 ↑": r["DIR@FAR=0.01_mean"],
            "DIR@FAR=0.1 ↑": r["DIR@FAR=0.1_mean"],
        })

out = pd.DataFrame(rows)
out = out.sort_values(["Enrollment", "EER (%) ↓"])

out_path = RESULTS_DIR / "voxblink_model_comparison_table.csv"
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
print(f"\nSaved: {out_path}")
