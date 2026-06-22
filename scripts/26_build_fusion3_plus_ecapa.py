from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

X_train = np.load(DATA_DIR / "fusion3_train.npy").astype("float32")
X_test = np.load(DATA_DIR / "fusion3_test.npy").astype("float32")

ecapa_train = np.load(RESULTS_DIR / "speechbrain_ecapa_train_embeddings.npy").astype("float32")
ecapa_test = np.load(RESULTS_DIR / "speechbrain_ecapa_test_embeddings.npy").astype("float32")

print("fusion3 train:", X_train.shape)
print("ecapa train:", ecapa_train.shape)
print("fusion3 test:", X_test.shape)
print("ecapa test:", ecapa_test.shape)

X_train_new = np.concatenate([X_train, ecapa_train], axis=1)
X_test_new = np.concatenate([X_test, ecapa_test], axis=1)

np.save(DATA_DIR / "fusion3_plus_speechbrain_ecapa_train.npy", X_train_new)
np.save(DATA_DIR / "fusion3_plus_speechbrain_ecapa_test.npy", X_test_new)

print("saved train:", X_train_new.shape)
print("saved test:", X_test_new.shape)
