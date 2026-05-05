"""
Tuned insurance claims training pipeline.

Improvements implemented:
1) Leakage-safe preprocessing in pipeline.
2) Class imbalance handling with class weights.
3) Optional XGBoost model comparison (if installed).
4) Threshold tuning to improve precision while preserving recall.
5) Cross-validation and artifact export for dashboard use.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

WORKING_DIR = r"c:\Users\bhoom\Downloads\ml1"
INPUT_FILE = "motor_data_combined.csv"

BEST_MODEL_FILE = "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.json"
TRAIN_REPORT_FILE = "tuned_model_report.json"
THRESHOLD_SWEEP_FILE = "threshold_sweep.csv"
MODEL_COMPARE_FILE = "model_comparison.csv"
CONFUSION_FILE = "confusion_matrix_tuned.csv"

TARGET_MIN_RECALL = 0.60
TRAIN_SAMPLE_MAX = 100000
USE_XGBOOST = False


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    df = df.drop_duplicates().copy()
    df["CLAIM"] = (df["CLAIM_PAID"].fillna(0) > 0).astype(int)

    df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], errors="coerce")
    df["INSR_END"] = pd.to_datetime(df["INSR_END"], errors="coerce")
    df["PROD_YEAR"] = pd.to_numeric(df["PROD_YEAR"], errors="coerce")

    # More robust than raw EFFECTIVE_YR field in this dataset.
    df["EFFECTIVE_YEAR_DERIVED"] = df["INSR_BEGIN"].dt.year
    df["POLICY_DURATION_DAYS"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
    df["VEHICLE_AGE_YEARS"] = df["EFFECTIVE_YEAR_DERIVED"] - df["PROD_YEAR"]

    return df


def build_preprocessor(feature_cols, categorical_cols):
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def threshold_search(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.60):
    rows = []
    for t in np.arange(0.10, 0.96, 0.02):
        pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": round(float(t), 2), "precision": p, "recall": r, "f1": f1})

    sweep = pd.DataFrame(rows)

    constrained = sweep[sweep["recall"] >= min_recall].copy()
    if len(constrained) > 0:
        # Prioritize precision while preserving desired recall.
        best = constrained.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        # Fallback: best F1 if recall constraint cannot be met.
        best = sweep.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0]

    return sweep, float(best["threshold"])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def main():
    print("=" * 72)
    print("INSURANCE CLAIMS - TUNED TRAINING")
    print("=" * 72)

    os.chdir(WORKING_DIR)

    df = load_data()
    print(f"Rows after dedup: {len(df):,}")
    print(f"Claim rate: {df['CLAIM'].mean():.4f}")

    feature_cols = [
        "SEX",
        "INSR_TYPE",
        "INSURED_VALUE",
        "PREMIUM",
        "SEATS_NUM",
        "CARRYING_CAPACITY",
        "TYPE_VEHICLE",
        "CCM_TON",
        "MAKE",
        "USAGE",
        "VEHICLE_AGE_YEARS",
        "POLICY_DURATION_DAYS",
    ]
    categorical_cols = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "MAKE", "USAGE"]

    X = df[feature_cols].copy()
    y = df["CLAIM"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if len(X_train) > TRAIN_SAMPLE_MAX:
        sample_idx = y_train.groupby(y_train).sample(
            frac=TRAIN_SAMPLE_MAX / len(X_train),
            random_state=42,
        ).index
        X_train_fit = X_train.loc[sample_idx]
        y_train_fit = y_train.loc[sample_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    print(f"Train rows (full): {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Train rows (fit sample): {len(X_train_fit):,}")

    preprocessor = build_preprocessor(feature_cols, categorical_cols)

    models = {
        "logreg_balanced": LogisticRegression(
            max_iter=150,
            class_weight="balanced",
            n_jobs=None,
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=40,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    # Optional advanced model.
    if USE_XGBOOST:
        try:
            from xgboost import XGBClassifier

            pos = max(int((y_train_fit == 1).sum()), 1)
            neg = max(int((y_train_fit == 0).sum()), 1)
            scale_pos_weight = neg / pos

            models["xgboost"] = XGBClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.10,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )
            print("XGBoost detected: included in model comparison.")
        except Exception:
            print("XGBoost not installed: skipping XGBoost comparison.")
    else:
        print("XGBoost skipped (USE_XGBOOST=False).")

    results = []
    trained = {}

    for name, estimator in models.items():
        print(f"\nTraining: {name}")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train_fit, y_train_fit)

        prob = pipe.predict_proba(X_test)[:, 1]
        base_pred = (prob >= 0.5).astype(int)

        row = {
            "model": name,
            "accuracy_0_5": accuracy_score(y_test, base_pred),
            "precision_0_5": precision_score(y_test, base_pred, zero_division=0),
            "recall_0_5": recall_score(y_test, base_pred, zero_division=0),
            "f1_0_5": f1_score(y_test, base_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
        }
        results.append(row)
        trained[name] = (pipe, prob)

    compare_df = pd.DataFrame(results).sort_values(["pr_auc", "f1_0_5"], ascending=[False, False])
    compare_df.to_csv(MODEL_COMPARE_FILE, index=False)

    best_name = compare_df.iloc[0]["model"]
    best_pipe, best_prob = trained[best_name]
    print(f"\nBest base model by PR-AUC: {best_name}")

    sweep_df, best_threshold = threshold_search(y_test.values, best_prob, min_recall=TARGET_MIN_RECALL)
    sweep_df.to_csv(THRESHOLD_SWEEP_FILE, index=False)
    print(f"Selected threshold: {best_threshold:.2f}")

    metrics = evaluate(y_test.values, best_prob, best_threshold)

    # CV check on a capped sample for speed.
    cv_size = min(50000, len(X_train_fit))
    if cv_size < len(X_train_fit):
        idx = y_train_fit.groupby(y_train_fit).sample(frac=cv_size / len(X_train_fit), random_state=42).index
        X_cv = X_train_fit.loc[idx]
        y_cv = y_train_fit.loc[idx]
    else:
        X_cv = X_train_fit
        y_cv = y_train_fit

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipe, X_cv, y_cv, cv=cv, scoring="average_precision", n_jobs=-1)

    report = {
        "best_model": best_name,
        "target_min_recall": TARGET_MIN_RECALL,
        "selected_threshold": best_threshold,
        "holdout": metrics,
        "cv_pr_auc_mean": float(np.mean(cv_scores)),
        "cv_pr_auc_std": float(np.std(cv_scores)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "claim_rate_train": float(y_train.mean()),
        "claim_rate_test": float(y_test.mean()),
    }

    cm = np.array(metrics["confusion_matrix"])
    pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(CONFUSION_FILE)

    with open(TRAIN_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=2)

    joblib.dump(best_pipe, BEST_MODEL_FILE)

    print("\nSaved artifacts:")
    print(f"- {BEST_MODEL_FILE}")
    print(f"- {THRESHOLD_FILE}")
    print(f"- {TRAIN_REPORT_FILE}")
    print(f"- {THRESHOLD_SWEEP_FILE}")
    print(f"- {MODEL_COMPARE_FILE}")
    print(f"- {CONFUSION_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
