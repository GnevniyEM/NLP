import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import logging
import json
from datetime import datetime

# --- logging ---
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = log_dir / f"run_{run_id}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("cwd=%s", Path.cwd())
logger.info("run_id=%s", run_id)

# --- data ---
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")
logger.info("train shape=%s, test shape=%s", train.shape, test.shape)

def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", " __url__ ", regex=True)
    s = s.str.replace(r"@\w+", " __user__ ", regex=True)
    s = s.str.replace(r"#(\w+)", r" \1 ", regex=True)
    s = s.str.replace(r"\d+", " __num__ ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

X = clean_text(train["text"])
y = train["target"].astype(int)
X_test = clean_text(test["text"])

tfidf = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
])

pipe = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(max_iter=2000, solver="liblinear")),
])

param_grid = {
    "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "clf__penalty": ["l1", "l2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring={"f1": "f1", "accuracy": "accuracy"},
    refit="f1",
    cv=cv,
    n_jobs=-1,
)

logger.info("GridSearch start. param_grid=%s", param_grid)
gs.fit(X, y)
logger.info("GridSearch done.")

best_i = gs.best_index_
best_f1 = gs.cv_results_["mean_test_f1"][best_i]
best_acc = gs.cv_results_["mean_test_accuracy"][best_i]

logger.info("Best params: %s", gs.best_params_)
logger.info("CV mean F1=%.6f, mean Accuracy=%.6f", best_f1, best_acc)

# save full cv results
cv_path = log_dir / f"run_{run_id}_cv_results.csv"
pd.DataFrame(gs.cv_results_).to_csv(cv_path, index=False)
logger.info("Saved cv_results: %s", cv_path)

# save summary json
summary = {
    "run_id": run_id,
    "best_params": gs.best_params_,
    "cv_mean_f1": float(best_f1),
    "cv_mean_accuracy": float(best_acc),
    "n_splits": cv.get_n_splits(),
    "train_shape": [int(train.shape[0]), int(train.shape[1])],
    "test_shape": [int(test.shape[0]), int(test.shape[1])],
}
summary_path = log_dir / f"run_{run_id}_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info("Saved summary: %s", summary_path)

pred = gs.predict(X_test)

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub_path = Path("submission.csv")
sub.to_csv(sub_path, index=False)
logger.info("Saved submission: %s", sub_path)

print("Best params:", gs.best_params_)
print("CV F1 (mean):", best_f1)
print("CV Accuracy (mean):", best_acc)
print("Saved:", sub_path)
print("Log file:", log_file)
