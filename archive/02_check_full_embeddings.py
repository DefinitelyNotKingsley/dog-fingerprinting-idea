from pathlib import Path
import numpy as np

EMB_DIR = Path(
    "/Users/kingsley/DogSpeak_Dataset/benchmark_72dogs/embeddings_balanced/dog2vec"
)

X_train = np.load(EMB_DIR / "train_embeddings.npy")
X_test = np.load(EMB_DIR / "test_embeddings.npy")

y_train = np.load(
    EMB_DIR / "train_labels.npy",
    allow_pickle=True
)

y_test = np.load(
    EMB_DIR / "test_labels.npy",
    allow_pickle=True
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("Train dogs:", len(set(y_train)))
print("Test dogs:", len(set(y_test)))

print("Embedding dimension:", X_train.shape[1])

print("\nFirst 5 labels:")
print(y_train[:5])