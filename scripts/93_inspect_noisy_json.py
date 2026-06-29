import json
from pathlib import Path
from collections import Counter
import pandas as pd

DOWNLOADS = Path.home() / "Downloads"

# If needed, hard-code:
# JSON_PATH = Path("/Users/kingsley/Downloads/channel_sentences.json")
JSON_PATH = None

MIN_DUR = 0.25
MAX_DUR = 5.0
MIN_CLIPS_PER_DOG = 50

OUT_DIR = Path("results/noisy_json_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_json():
    if JSON_PATH is not None:
        return JSON_PATH

    candidates = sorted(
        DOWNLOADS.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError("No .json file found in ~/Downloads")

    return candidates[0]


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_video_id(sentence_id):
    sid = str(sentence_id)

    # Format 1:
    # UC8rMT7uXltc3ZkShVwM19hA+mqSLOVoXh20+00000+00000
    if "+" in sid and not sid.startswith("["):
        return sid.split("+")[0]

    # Format 2:
    # [Best Boy Benson]_[2023-05-30_Vx7uANnNqKY]+00000_00009
    # Extract YouTube ID inside final underscore block if present
    if "]_[" in sid:
        try:
            inside = sid.split("]_[", 1)[1].split("]", 1)[0]
            # inside looks like: 2023-05-30_Vx7uANnNqKY
            parts = inside.split("_")
            if len(parts) >= 2:
                return parts[-1]
        except Exception:
            pass

    # Fallback
    return sid.split("+")[0]


def main():
    json_path = find_json()
    print("Using JSON:", json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    total_clips = 0
    duration_pass = 0
    file_exists_pass = 0

    dog_counts_raw = Counter()
    dog_counts_duration = Counter()
    dog_counts_usable = Counter()
    video_counts_usable = Counter()

    reject_reasons = Counter()
    rows = []

    for dog_id, clips in data.items():
        if clips is None:
            reject_reasons["dog_has_null_clip_list"] += 1
            continue

        dog_counts_raw[dog_id] += len(clips)

        for clip in clips:
            total_clips += 1

            if not isinstance(clip, dict):
                reject_reasons["clip_not_dict"] += 1
                continue

            sentence_id = clip.get("sentence_id")
            audio_path_raw = clip.get("audio_path")
            dur = safe_float(clip.get("duration"))
            start_ts = safe_float(clip.get("start_ts"))
            end_ts = safe_float(clip.get("end_ts"))

            if dur is None:
                reject_reasons["missing_or_invalid_duration"] += 1
                continue

            if dur < MIN_DUR:
                reject_reasons["too_short_<0.25s"] += 1
                continue

            if dur > MAX_DUR:
                reject_reasons["too_long_>5s"] += 1
                continue

            duration_pass += 1
            dog_counts_duration[dog_id] += 1

            if not audio_path_raw:
                reject_reasons["missing_audio_path"] += 1
                continue

            audio_path = Path(audio_path_raw)

            # Skip local file-existence check because audio paths are server paths
            file_exists_pass += 1
            dog_counts_usable[dog_id] += 1

            video_id = extract_video_id(sentence_id)
            video_counts_usable[video_id] += 1

            rows.append({
                "dog_id": dog_id,
                "sentence_id": sentence_id,
                "video_id": video_id,
                "audio_path": str(audio_path),
                "duration": dur,
                "start_ts": start_ts,
                "end_ts": end_ts,
            })

    dogs_ge50_duration = [d for d, c in dog_counts_duration.items() if c >= MIN_CLIPS_PER_DOG]
    dogs_ge50_usable = [d for d, c in dog_counts_usable.items() if c >= MIN_CLIPS_PER_DOG]
    clips_after_dog_filter = sum(c for d, c in dog_counts_usable.items() if c >= MIN_CLIPS_PER_DOG)

    usable_df = pd.DataFrame(rows)
    if len(usable_df):
        usable_ge50_df = usable_df[usable_df["dog_id"].isin(dogs_ge50_usable)].copy()
    else:
        usable_ge50_df = usable_df

    dog_df = pd.DataFrame([
        {
            "dog_id": d,
            "raw_clips": dog_counts_raw[d],
            "duration_pass": dog_counts_duration[d],
            "usable_clips": dog_counts_usable[d],
            "kept_ge50": dog_counts_usable[d] >= MIN_CLIPS_PER_DOG,
        }
        for d in sorted(dog_counts_raw)
    ])

    video_df = pd.DataFrame([
        {"video_id": v, "usable_clips": c}
        for v, c in video_counts_usable.most_common()
    ])

    reject_df = pd.DataFrame([
        {"reason": k, "count": v}
        for k, v in reject_reasons.most_common()
    ])

    dog_df.to_csv(OUT_DIR / "dog_clip_counts.csv", index=False)
    video_df.to_csv(OUT_DIR / "video_clip_counts.csv", index=False)
    reject_df.to_csv(OUT_DIR / "reject_reasons.csv", index=False)
    usable_df.to_csv(OUT_DIR / "usable_clips_all_duration_only.csv", index=False)
    usable_ge50_df.to_csv(OUT_DIR / "usable_clips_dogs_ge50_duration_only.csv", index=False)

    print("\n==============================")
    print("RAW DATA")
    print("Dogs:", len(data))
    print("Total clips:", total_clips)

    print("\n==============================")
    print("FILTER SUMMARY")
    print(f"Duration pass ({MIN_DUR}s <= duration <= {MAX_DUR}s):", duration_pass)
    print("Existing audio files after duration filter:", file_exists_pass)
    print("Dogs with >=50 clips after duration filter:", len(dogs_ge50_duration))
    print("Dogs with >=50 usable clips:", len(dogs_ge50_usable))
    print("Usable clips after dog >=50 filter:", clips_after_dog_filter)
    print("Unique videos after filters:", len(video_counts_usable))

    print("\n==============================")
    print("REJECT REASONS")
    for k, v in reject_reasons.most_common():
        print(f"{k}: {v}")

    print("\n==============================")
    print("TOP 20 DOGS BY USABLE CLIPS")
    for dog, count in dog_counts_usable.most_common(20):
        print(dog, count)

    print("\n==============================")
    print("TOP 20 VIDEOS BY USABLE CLIPS")
    for video, count in video_counts_usable.most_common(20):
        print(video, count)

    print("\nSaved:")
    print(OUT_DIR / "dog_clip_counts.csv")
    print(OUT_DIR / "video_clip_counts.csv")
    print(OUT_DIR / "reject_reasons.csv")
    print(OUT_DIR / "usable_clips_all.csv")
    print(OUT_DIR / "usable_clips_dogs_ge50.csv")


if __name__ == "__main__":
    main()
