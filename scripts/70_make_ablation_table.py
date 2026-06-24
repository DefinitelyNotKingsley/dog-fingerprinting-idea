from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
OUT = RESULTS_DIR / "ablation_table.csv"

rows = [
    {
        "Experiment": "Fusion3 + CenterLoss",
        "Features": "Dog2Vec + MFCC + ECAPA",
        "Training": "ArcFace + Center Loss",
        "EER_5bark": 27.87,
        "DIR@FAR=0.1": 56.85,
        "Takeaway": "Strong baseline; classification-style metric learning works."
    },
    {
        "Experiment": "Fusion3 + Triplet Loss",
        "Features": "Dog2Vec + MFCC + ECAPA",
        "Training": "Triplet Loss",
        "EER_5bark": 19.49,
        "DIR@FAR=0.1": 64.06,
        "Takeaway": "Best model; direct fingerprint training improves open-set ID."
    },
    {
        "Experiment": "DogECAPA-LDE",
        "Features": "Log-Mel only",
        "Training": "ArcFace",
        "EER_5bark": 50.00,
        "DIR@FAR=0.1": 0.00,
        "Takeaway": "Collapsed embeddings; standalone ECAPA did not transfer."
    },
    {
        "Experiment": "DogECAPA-Tiny Softmax",
        "Features": "Log-Mel only",
        "Training": "Softmax",
        "EER_5bark": 49.01,
        "DIR@FAR=0.1": 1.98,
        "Takeaway": "Avoided collapse but weak open-set fingerprints."
    },
    {
        "Experiment": "DogECAPA-Tiny Stable AAM",
        "Features": "Log-Mel only",
        "Training": "SpeechBrain-style AAM",
        "EER_5bark": 49.60,
        "DIR@FAR=0.1": 2.29,
        "Takeaway": "Stable but weak; AAM alone did not solve dog fingerprinting."
    },
    {
        "Experiment": "WavLM smoke test",
        "Features": "WavLM",
        "Training": "None",
        "EER_5bark": None,
        "DIR@FAR=0.1": None,
        "Takeaway": "Weak same-vs-different separation; gap = 0.021."
    },
    {
        "Experiment": "HuBERT smoke test",
        "Features": "HuBERT",
        "Training": "None",
        "EER_5bark": None,
        "DIR@FAR=0.1": None,
        "Takeaway": "Weak same-vs-different separation; gap = 0.015."
    },
]

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(df.to_string(index=False))
print("\nSaved:", OUT)
