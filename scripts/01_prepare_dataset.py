import pandas as pd
from pathlib import Path

CSV_PATH = Path("/Users/kingsley/DogSpeak_Dataset/dogspeaks_72dogs_100to150_balanced_split.csv")
AUDIO_ROOT = Path("/Users/kingsley/DogSpeak_Dataset/dogspeak_released")

df = pd.read_csv(CSV_PATH)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== DOG COUNTS =====")
if "dog_id" in df.columns:
    print(df["dog_id"].value_counts().head(10))
else:
    print("No dog_id column found")

print("\n===== SPLIT COUNTS =====")
if "split" in df.columns:
    print(df["split"].value_counts())
else:
    print("No split column found")

print("\n===== POSSIBLE FILE/PATH COLUMNS =====")
for col in df.columns:
    col_lower = col.lower()

    if (
        "path" in col_lower
        or "file" in col_lower
        or "wav" in col_lower
    ):
        print(col)