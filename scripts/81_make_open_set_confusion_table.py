import pandas as pd

df = pd.read_csv("results/triplet_fusion3_100splits_all.csv")

# Best threshold for each enrollment size
best = (
    df.groupby(["enroll_k", "threshold"])["overall_accuracy"]
      .mean()
      .reset_index()
      .sort_values(["enroll_k", "overall_accuracy"], ascending=[True, False])
      .groupby("enroll_k")
      .head(1)
)

rows = []

for _, r in best.iterrows():
    k = int(r["enroll_k"])
    th = r["threshold"]

    sub = df[(df.enroll_k == k) & (df.threshold == th)]

    known_correct = sub["known_id_accuracy"]
    false_unknown = sub["false_unknown_rate"]

    wrong_known = 1 - known_correct - false_unknown

    unknown_correct = sub["unknown_recall"]
    false_accept = 1 - unknown_correct

    rows.append({
        "Enrollment": f"{k} barks",
        "Threshold": th,

        "Known → Correct (%)":
            f"{known_correct.mean()*100:.2f} ± {known_correct.std()*100:.2f}",

        "Known → Wrong ID (%)":
            f"{wrong_known.mean()*100:.2f} ± {wrong_known.std()*100:.2f}",

        "Known → Unknown (%)":
            f"{false_unknown.mean()*100:.2f} ± {false_unknown.std()*100:.2f}",

        "Unknown → Unknown (%)":
            f"{unknown_correct.mean()*100:.2f} ± {unknown_correct.std()*100:.2f}",

        "Unknown → Known (%)":
            f"{false_accept.mean()*100:.2f} ± {false_accept.std()*100:.2f}",
    })

out = pd.DataFrame(rows)

print(out.to_string(index=False))

out.to_csv(
    "results/open_set_confusion_summary_100splits.csv",
    index=False,
)

print("\nSaved:")
print("results/open_set_confusion_summary_100splits.csv")