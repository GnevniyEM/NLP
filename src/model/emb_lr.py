import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.io import read_embeddings, read_raw, save_submission


def run_emb_lr_gridsearch(run_dir: Path, log) -> None:
    train, test = read_raw()
    X_train, X_test = read_embeddings()
    y = train["target"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    param_grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0]}

    cfg = {
        "timestamp_berlin": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds"),
        "data": {
            "train_path": "data/raw/train.csv",
            "test_path": "data/raw/test.csv",
            "train_emb_path": "data/processed/train_emb.npy",
            "test_emb_path": "data/processed/test_emb.npy",
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
        },
        "model": {"name": "LogisticRegression", "solver": "lbfgs", "max_iter": 2000},
        "grid": param_grid,
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42},
        "metrics": ["f1", "accuracy"],
        "refit": "f1",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("GridSearch start. param_grid=%s", param_grid)

    gs = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring={"f1": "f1", "accuracy": "accuracy"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    gs.fit(X_train, y)

    best_i = gs.best_index_
    best_f1 = float(gs.cv_results_["mean_test_f1"][best_i])
    best_acc = float(gs.cv_results_["mean_test_accuracy"][best_i])

    log.info("GridSearch done.")
    log.info("Best params: %s", gs.best_params_)
    log.info("CV mean F1=%.6f, mean Accuracy=%.6f", best_f1, best_acc)

    # вывод в терминал (как просили)
    print("Best params:", gs.best_params_)
    print(f"Best CV F1 (mean): {best_f1:.6f}")
    print(f"Best CV Accuracy (mean): {best_acc:.6f}")
    print("Run dir:", run_dir.as_posix())

    # артефакты
    pd.DataFrame(gs.cv_results_).to_csv(run_dir / "cv_results.csv", index=False)

    summary = {
        "best_params": gs.best_params_,
        "cv_mean_f1": best_f1,
        "cv_mean_accuracy": best_acc,
        "n_splits": cv.get_n_splits(),
        "train_shape": [int(train.shape[0]), int(train.shape[1])],
        "test_shape": [int(test.shape[0]), int(test.shape[1])],
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    test_pred = gs.predict(X_test).astype(int)
    sub_path_root = save_submission(test["id"].to_numpy(), test_pred, out_path="submission_emb_lr.csv")
    sub_path_copy = save_submission(test["id"].to_numpy(), test_pred, out_path=str(run_dir / "submission_emb_lr.csv"))

    log.info("Saved submission: %s", sub_path_root.as_posix())
    log.info("Saved submission copy: %s", sub_path_copy.as_posix())
