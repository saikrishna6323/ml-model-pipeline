"""Trains a lightweight classifier to predict CI/CD pipeline failure
from historical run features. Real, runnable on public GitHub Actions
run data (via GitHub API) or the included synthetic sample dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEATURE_COLS = [
    "rolling_failure_rate", "dependency_files_changed", "step_duration_seconds",
    "num_recent_errors", "day_of_week", "hour_of_day",
]

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train(data_path: str, model_out: str = "model.joblib"):
    df = load_data(data_path)
    X = df[FEATURE_COLS]
    y = df["failed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    joblib.dump(clf, model_out)
    return clf

if __name__ == "__main__":
    train("data/sample_runs.csv")
