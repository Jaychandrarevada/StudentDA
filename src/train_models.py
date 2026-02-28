"""Train and compare student-risk classifiers.

Usage:
python src/train_models.py --data data/student_data.csv --target at_risk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "attendance_pct",
    "internal_score",
    "assignment_avg",
    "quiz_avg",
    "lms_logins_per_week",
    "lms_hours_per_week",
    "content_views",
    "forum_posts",
    "previous_gpa",
    "department",
    "hostel_resident",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression, Random Forest and XGBoost models for student risk prediction."
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column (0/1 or boolean label)")
    parser.add_argument("--id-columns", nargs="*", default=[], help="Optional ID columns to drop")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store metrics and model")
    return parser.parse_args()


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols), ("cat", categorical_transformer, categorical_cols)]
    )


def build_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=450,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
            scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"},
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(by="cv_roc_auc_mean", ascending=False)


def final_train_and_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    model_name: str,
    model: Any,
) -> tuple[Pipeline, dict[str, float | str]]:
    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | str] = {
        "model": model_name,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred),
    }
    return pipeline, metrics


def train_and_save(
    data_path: str,
    target: str = "at_risk",
    id_columns: list[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> dict[str, Any]:
    id_columns = id_columns or []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    dropped = [col for col in id_columns if col in df.columns]
    if dropped:
        df = df.drop(columns=dropped)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    preprocessor = build_preprocessor(X_train)
    models = build_models(seed)
    leaderboard = evaluate_models(X_train, y_train, preprocessor, models, seed)

    best_model_name = str(leaderboard.iloc[0]["model"])
    pipeline, test_metrics = final_train_and_test(
        X_train, X_test, y_train, y_test, preprocessor, best_model_name, models[best_model_name]
    )

    leaderboard_path = out_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    test_metrics_path = out_dir / "test_metrics.json"
    test_metrics_path.write_text(json.dumps(test_metrics, indent=2))

    model_path = out_dir / "best_model.joblib"
    joblib.dump(pipeline, model_path)

    summary = {
        "leaderboard_path": str(leaderboard_path),
        "test_metrics_path": str(test_metrics_path),
        "model_path": str(model_path),
        "best_model": best_model_name,
        "test_metrics": test_metrics,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = train_and_save(
        data_path=args.data,
        target=args.target,
        id_columns=args.id_columns,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
