from pathlib import Path
import numpy as np

DOG2VEC_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/dog2vec")
MFCC_DIR = Path("/Users/kingsley/DogSpeak_Dataset/fingerprint_experiments/full_embeddings/mfcc")
OUT_DIR = Path("data")

OUT_DIR.mkdir(exist_ok=True)


def fix_labels(labels):
    return np.array(["dog_7" if label in ["dog_7a", "dog_7b"] else label for label in labels])


def load_set(base_dir):
    X_train = np.load(base_dir / "train_embeddings.npy")
    X_test = np.load(base_dir / "test_embeddings.npy")
    y_train = np.load(base_dir / "train_labels.npy", allow_pickle=True)
    y_test = np.load(base_dir / "test_labels.npy", allow_pickle=True)
    train_files = np.load(base_dir / "train_files.npy", allow_pickle=True)
    test_files = np.load(base_dir / "test_files.npy", allow_pickle=True)
    return X_train, X_test, y_train, y_test, train_files, test_files


def main():
    d2v_train, d2v_test, y_train_d2v, y_test_d2v, train_files_d2v, test_files_d2v = load_set(DOG2VEC_DIR)
    mfcc_train, mfcc_test, y_train_mfcc, y_test_mfcc, train_files_mfcc, test_files_mfcc = load_set(MFCC_DIR)

    assert np.array_equal(y_train_d2v, y_train_mfcc), "Train labels do not match"
    assert np.array_equal(y_test_d2v, y_test_mfcc), "Test labels do not match"
    assert np.array_equal(train_files_d2v, train_files_mfcc), "Train files do not match"
    assert np.array_equal(test_files_d2v, test_files_mfcc), "Test files do not match"

    y_train = fix_labels(y_train_d2v)
    y_test = fix_labels(y_test_d2v)

    X_train = np.concatenate([d2v_train, mfcc_train], axis=1)
    X_test = np.concatenate([d2v_test, mfcc_test], axis=1)

    np.save(OUT_DIR / "fusion_train.npy", X_train)
    np.save(OUT_DIR / "fusion_test.npy", X_test)
    np.save(OUT_DIR / "train_labels.npy", y_train)
    np.save(OUT_DIR / "test_labels.npy", y_test)
    np.save(OUT_DIR / "train_files.npy", train_files_d2v)
    np.save(OUT_DIR / "test_files.npy", test_files_d2v)

    print("Saved prepared fusion data")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Train dogs:", len(set(y_train)))
    print("Test dogs:", len(set(y_test)))
    print("dog_7 train count:", list(y_train).count("dog_7"))
    print("dog_7 test count:", list(y_test).count("dog_7"))


if __name__ == "__main__":
    main()
