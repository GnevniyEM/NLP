from pathlib import Path

import numpy as np
import pandas as pd


def read_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    return train, test


def read_embeddings() -> tuple[np.ndarray, np.ndarray]:
    X_train = np.load("data/processed/train_emb.npy")
    X_test = np.load("data/processed/test_emb.npy")
    return X_train, X_test


def save_submission(test_ids, pred, out_path: str = "submission_emb_lr.csv") -> Path:
    out = Path(out_path)
    pd.DataFrame({"id": test_ids, "target": pred}).to_csv(out, index=False)
    return out
