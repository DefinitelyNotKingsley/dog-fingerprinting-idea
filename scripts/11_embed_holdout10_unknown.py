from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FusionMLP(nn.Module):
    def __init__(self, input_dim=1040, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


def main():
    X_unknown = np.load(DATA_DIR / "holdout10/unknown_test_X.npy").astype("float32")

    mean = np.load(RESULTS_DIR / "feature_mean.npy")
    std = np.load(RESULTS_DIR / "feature_std.npy")
    X_unknown = (X_unknown - mean) / std

    model = FusionMLP(input_dim=1040, embed_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "holdout10_fusion3_mlp_arcface.pt", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        emb = model(torch.tensor(X_unknown).to(DEVICE)).cpu().numpy()

    np.save(RESULTS_DIR / "holdout10_unknown_test_embeddings.npy", emb)

    print("Saved unknown embeddings:", emb.shape)


if __name__ == "__main__":
    main()
