from pathlib import Path
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DOGSPEAK_ROOT = Path("/Users/kingsley/DogSpeak_Dataset")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_PRINT = 500


def load_mono(path):
    signal, sr = torchaudio.load(path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal


def extract_split(classifier, split_name, file_path):
    files = np.load(file_path, allow_pickle=True)
    embs = []

    print(f"\nExtracting {split_name}: {len(files)} files")

    for i, rel_path in enumerate(files):
        wav_path = DOGSPEAK_ROOT / rel_path

        try:
            signal = load_mono(wav_path).to(DEVICE)
            with torch.no_grad():
                emb = classifier.encode_batch(signal)
            emb = emb.squeeze().detach().cpu().numpy()
        except Exception as e:
            print(f"FAILED {i}: {wav_path} | {e}")
            emb = np.zeros(192, dtype="float32")

        embs.append(emb.astype("float32"))

        if (i + 1) % BATCH_PRINT == 0:
            print(f"  {split_name}: {i + 1}/{len(files)}")

    embs = np.vstack(embs)
    out = RESULTS_DIR / f"speechbrain_ecapa_{split_name}_embeddings.npy"
    np.save(out, embs)

    print("Saved:", out, embs.shape)


def main():
    print("Device:", DEVICE)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
    )
    classifier.eval()

    extract_split(classifier, "train", DATA_DIR / "train_files.npy")
    extract_split(classifier, "test", DATA_DIR / "test_files.npy")


if __name__ == "__main__":
    main()
