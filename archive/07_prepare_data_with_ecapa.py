from pathlib import Path
import numpy as np

D2V_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/dog2vec")
MFCC_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/mfcc")
ECAPA_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/ecapa")
OUT_DIR = Path("data")

OUT_DIR.mkdir(exist_ok=True)

def fix_labels(labels):
    return np.array(["dog_7" if x in ["dog_7a", "dog_7b"] else x for x in labels])

def load(base, split):
    X = np.load(base / f"{split}_embeddings.npy")
    y = np.load(base / f"{split}_labels.npy", allow_pickle=True)
    files = np.load(base / f"{split}_files.npy", allow_pickle=True)
    return X, y, files

def main():
    d2v_train, y_train, train_files = load(D2V_DIR, "train")
    d2v_test, y_test, test_files = load(D2V_DIR, "test")

    mfcc_train, y_train_mfcc, train_files_mfcc = load(MFCC_DIR, "train")
    mfcc_test, y_test_mfcc, test_files_mfcc = load(MFCC_DIR, "test")

    ecapa_train, y_train_ecapa, train_files_ecapa = load(ECAPA_DIR, "train")
    ecapa_test, y_test_ecapa, test_files_ecapa = load(ECAPA_DIR, "test")

    assert np.array_equal(y_train, y_train_mfcc)
    assert np.array_equal(y_test, y_test_mfcc)
    assert np.array_equal(y_train, y_train_ecapa)
    assert np.array_equal(y_test, y_test_ecapa)

    assert np.array_equal(train_files, train_files_mfcc)
    assert np.array_equal(test_files, test_files_mfcc)
    assert np.array_equal(train_files, train_files_ecapa)
    assert np.array_equal(test_files, test_files_ecapa)

    y_train = fix_labels(y_train)
    y_test = fix_labels(y_test)

    X_train = np.concatenate([d2v_train, mfcc_train, ecapa_train], axis=1)
    X_test = np.concatenate([d2v_test, mfcc_test, ecapa_test], axis=1)

    np.save(OUT_DIR / "fusion3_train.npy", X_train)
    np.save(OUT_DIR / "fusion3_test.npy", X_test)
    np.save(OUT_DIR / "fusion3_train_labels.npy", y_train)
    np.save(OUT_DIR / "fusion3_test_labels.npy", y_test)

    print("Saved 3-feature fusion data")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Train dogs:", len(set(y_train)))
    print("Test dogs:", len(set(y_test)))

if __name__ == "__main__":
    main()
