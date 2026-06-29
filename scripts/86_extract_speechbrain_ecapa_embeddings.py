from pathlib import Path
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DOGSPEAK_ROOT = Path.home() / "DogSpeak_Dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAVE_DIR = "pretrained_models/spkrec-ecapa-voxceleb"

OUT_TRAIN = RESULTS_DIR / "speechbrain_ecapa_train_embeddings.npy"
OUT_TEST = RESULTS_DIR / "speechbrain_ecapa_test_embeddings.npy"


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0)


def extract_split(files, model, out_path):
    embs = []

    for i, rel in enumerate(files):
        path = DOGSPEAK_ROOT / rel

        if not path.exists():
            raise FileNotFoundError(path)

        wav = load_audio(path).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_batch(wav.unsqueeze(0))
            emb = emb.squeeze().detach().cpu().numpy()

        embs.append(emb)

        if (i + 1) % 500 == 0:
            arr = np.vstack(embs).astype("float32")
            np.save(out_path, arr)
            print(f"Saved checkpoint {out_path}: {arr.shape}")

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)} done")

    arr = np.vstack(embs).astype("float32")
    np.save(out_path, arr)
    print("Saved final:", out_path, arr.shape)


def main():
    print("Device:", DEVICE)

    train_files = np.load(DATA_DIR / "train_files.npy", allow_pickle=True)
    test_files = np.load(DATA_DIR / "test_files.npy", allow_pickle=True)

    print("Train files:", train_files.shape)
    print("Test files:", test_files.shape)

    model = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=SAVE_DIR,
        run_opts={"device": DEVICE},
    )

    print("Extracting train...")
    extract_split(train_files, model, OUT_TRAIN)

    print("Extracting test...")
    extract_split(test_files, model, OUT_TEST)


if __name__ == "__main__":
    main()
