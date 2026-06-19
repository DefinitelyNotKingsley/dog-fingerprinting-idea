from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
OUT = Path("data/holdout10")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_UNKNOWN_DOGS = 10

X_train = np.load(DATA_DIR / "fusion3_train.npy")
X_test = np.load(DATA_DIR / "fusion3_test.npy")
y_train = np.load(DATA_DIR / "fusion3_train_labels.npy", allow_pickle=True)
y_test = np.load(DATA_DIR / "fusion3_test_labels.npy", allow_pickle=True)

rng = np.random.default_rng(SEED)
all_dogs = np.array(sorted(set(y_train)))
unknown_dogs = set(rng.choice(all_dogs, size=NUM_UNKNOWN_DOGS, replace=False))
known_dogs = set(all_dogs) - unknown_dogs

train_known_mask = np.array([y in known_dogs for y in y_train])
test_known_mask = np.array([y in known_dogs for y in y_test])
test_unknown_mask = np.array([y in unknown_dogs for y in y_test])

np.save(OUT / "known_train_X.npy", X_train[train_known_mask])
np.save(OUT / "known_train_y.npy", y_train[train_known_mask])

np.save(OUT / "known_test_X.npy", X_test[test_known_mask])
np.save(OUT / "known_test_y.npy", y_test[test_known_mask])

np.save(OUT / "unknown_test_X.npy", X_test[test_unknown_mask])
np.save(OUT / "unknown_test_y.npy", y_test[test_unknown_mask])

np.save(OUT / "unknown_dogs.npy", np.array(sorted(unknown_dogs)))
np.save(OUT / "known_dogs.npy", np.array(sorted(known_dogs)))

print("Known dogs:", len(known_dogs))
print("Unknown dogs:", len(unknown_dogs))
print("Unknown dog IDs:", sorted(unknown_dogs))
print("Known train:", X_train[train_known_mask].shape)
print("Known test:", X_test[test_known_mask].shape)
print("Unknown test:", X_test[test_unknown_mask].shape)
