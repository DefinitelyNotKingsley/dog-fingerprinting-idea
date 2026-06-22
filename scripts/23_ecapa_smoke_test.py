from pathlib import Path
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DOGSPEAK_ROOT = Path("/Users/kingsley/DogSpeak_Dataset")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

files = np.load("data/train_files.npy", allow_pickle=True)
wav_path = DOGSPEAK_ROOT / files[0]

print("Device:", DEVICE)
print("Wav:", wav_path)
print("Exists:", wav_path.exists())

signal, sr = torchaudio.load(wav_path)

if signal.shape[0] > 1:
    signal = signal.mean(dim=0, keepdim=True)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)

emb = classifier.encode_batch(signal.to(DEVICE))
emb = emb.squeeze().detach().cpu().numpy()

print("Embedding shape:", emb.shape)
print("Embedding first 5:", emb[:5])
