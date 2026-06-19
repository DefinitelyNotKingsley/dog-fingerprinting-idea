from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path("/Users/kingsley/DogSpeak_Dataset")
BASE = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/dog2vec")
OUT = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/ecapa")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/Users/kingsley/DogSpeak_Dataset/benchmark_72dogs/pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)

def extract_one(rel_path):
    path = ROOT / rel_path
    signal, sr = torchaudio.load(str(path))

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    signal = signal.to(DEVICE)

    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()

    return emb

def extract_split(split):
    files = np.load(BASE / f"{split}_files.npy", allow_pickle=True)
    labels = np.load(BASE / f"{split}_labels.npy", allow_pickle=True)
    breeds = np.load(BASE / f"{split}_breeds.npy", allow_pickle=True)

    embs = []
    for f in tqdm(files, desc=f"Extracting {split}"):
        embs.append(extract_one(f))

    embs = np.vstack(embs)

    np.save(OUT / f"{split}_embeddings.npy", embs)
    np.save(OUT / f"{split}_files.npy", files)
    np.save(OUT / f"{split}_labels.npy", labels)
    np.save(OUT / f"{split}_breeds.npy", breeds)

    print(split, embs.shape)

def main():
    print("Device:", DEVICE)
    extract_split("train")
    extract_split("test")
    print("Saved ECAPA embeddings to:", OUT)

if __name__ == "__main__":
    main()
