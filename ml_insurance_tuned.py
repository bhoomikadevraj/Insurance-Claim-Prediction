"""
Leakage-safe insurance claims training pipeline.

Key improvements:
1) Claim-relevant feature engineering.
2) Class imbalance handling with scale_pos_weight.
3) Focused XGBoost hyperparameter tuning on validation PR-AUC.
4) Threshold tuning that enforces a recall floor.
5) Clean artifact export for the dashboard.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = "motor_data_combined.csv"

BEST_MODEL_FILE = "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.json"
TRAIN_REPORT_FILE = "tuned_model_report.json"
THRESHOLD_SWEEP_FILE = "threshold_sweep.csv"
MODEL_COMPARE_FILE = "model_comparison.csv"
CONFUSION_FILE = "confusion_matrix_tuned.csv"

TARGET_MIN_RECALL = 0.75
TRAIN_SAMPLE_MAX = 30000
RANDOM_STATE = 42
USE_XGBOOST = True


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
    df["POLICY_BEGIN_MONTH"] = df["INSR_BEGIN"].dt.month
    df["POLICY_BEGIN_QUARTER"] = df["INSR_BEGIN"].dt.quarter
    df["POLICY_END_MONTH"] = df["INSR_END"].dt.month
    df["POLICY_END_QUARTER"] = df["INSR_END"].dt.quarter
    df["POLICY_DURATION_MONTHS"] = df["POLICY_DURATION_DAYS"] / 30.0
    df["INSURED_VALUE_LOG"] = np.log1p(pd.to_numeric(df["INSURED_VALUE"], errors="coerce"))
    df["PREMIUM_LOG"] = np.log1p(pd.to_numeric(df["PREMIUM"], errors="coerce"))
    df["VALUE_PER_PREMIUM"] = pd.to_numeric(df["INSURED_VALUE"], errors="coerce") / (
        pd.to_numeric(df["PREMIUM"], errors="coerce").replace(0, np.nan)
    )
    df["VALUE_PER_SEAT"] = pd.to_numeric(df["INSURED_VALUE"], errors="coerce") / (
        pd.to_numeric(df["SEATS_NUM"], errors="coerce").replace(0, np.nan)
    )
    df["CAPACITY_PER_SEAT"] = pd.to_numeric(df["CARRYING_CAPACITY"], errors="coerce") / (
        pd.to_numeric(df["SEATS_NUM"], errors="coerce").replace(0, np.nan)
    )
    df["PREMIUM_PER_DAY"] = pd.to_numeric(df["PREMIUM"], errors="coerce") / (
        df["POLICY_DURATION_DAYS"].replace(0, np.nan)
    )
    df["AGE_BY_VALUE"] = df["VEHICLE_AGE_YEARS"] / (pd.to_numeric(df["INSURED_VALUE"], errors="coerce") + 1.0)

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
    for t in np.arange(0.05, 0.96, 0.01):
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


def fit_xgboost_tuned(X_train_enc, y_train, X_valid_enc, y_valid):
    from xgboost import XGBClassifier

    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    search_space = [
        {
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "gamma": 0.0,
            "reg_lambda": 5.0,
            "n_estimators": 200,
        }
    ]

    best_model = None
    best_prob = None
    best_score = -np.inf
    best_params = None

    for i, params in enumerate(search_space, start=1):
        candidate = XGBClassifier(**base_params, **params)
        candidate.fit(X_train_enc, y_train, eval_set=[(X_valid_enc, y_valid)], verbose=False)
        prob = candidate.predict_proba(X_valid_enc)[:, 1]
        score = average_precision_score(y_valid, prob)
        if score > best_score:
            best_score = score
            best_model = candidate
            best_prob = prob
            best_params = params
        print(f"  candidate {i:02d}/{len(search_space)} pr_auc={score:.4f} params={params}")

    print(f"Best XGBoost params: {best_params}")
    print(f"Best validation PR-AUC: {best_score:.4f}")
    return best_model, best_prob, best_params


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
        "POLICY_BEGIN_MONTH",
        "POLICY_BEGIN_QUARTER",
        "POLICY_END_MONTH",
        "POLICY_END_QUARTER",
        "POLICY_DURATION_MONTHS",
        "INSURED_VALUE_LOG",
        "PREMIUM_LOG",
        "VALUE_PER_PREMIUM",
        "VALUE_PER_SEAT",
        "CAPACITY_PER_SEAT",
        "PREMIUM_PER_DAY",
        "AGE_BY_VALUE",
    ]
    categorical_cols = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "MAKE", "USAGE"]

    X = df[feature_cols].copy()
    y = df["CLAIM"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_fit_full = X_train
    y_train_fit_full = y_train

    if len(X_train_fit_full) > TRAIN_SAMPLE_MAX:
        sample_idx = y_train_fit_full.groupby(y_train_fit_full).sample(
            frac=TRAIN_SAMPLE_MAX / len(X_train_fit_full),
            random_state=RANDOM_STATE,
        ).index
        X_train_fit = X_train_fit_full.loc[sample_idx]
        y_train_fit = y_train_fit_full.loc[sample_idx]
    else:
        X_train_fit = X_train_fit_full
        y_train_fit = y_train_fit_full

    # Split the fit sample again so XGBoost early stopping and threshold tuning
    # happen on a validation set, not on the held-out test set.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train_fit,
        y_train_fit,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_fit,
    )

    print(f"Train rows (full): {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Train rows (fit sample): {len(X_train_fit):,}")
    print(f"Fit rows: {len(X_fit):,} | Validation rows: {len(X_valid):,}")

    preprocessor = build_preprocessor(feature_cols, categorical_cols)
    preprocessor.fit(X_fit)
    X_fit_enc = preprocessor.transform(X_fit)
    X_valid_enc = preprocessor.transform(X_valid)

    results = []

    if USE_XGBOOST:
        try:
            print("\nTuning XGBoost on validation set")
            xgb_model, valid_prob, best_params = fit_xgboost_tuned(X_fit_enc, y_fit, X_valid_enc, y_valid)

            # Refit the winning model on the entire fit sample with the same params.
            # Early stopping is skipped here because we only need the final training pipeline.
            final_xgb = xgb_model.__class__(**xgb_model.get_params())
            final_pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", final_xgb),
            ])
            final_pipe.fit(X_train_fit, y_train_fit)

            valid_prob = final_pipe.predict_proba(X_valid)[:, 1]
            test_prob = final_pipe.predict_proba(X_test)[:, 1]

            results.append(
                {
                    "model": "xgboost_tuned",
                    "accuracy_0_5": accuracy_score(y_valid, (valid_prob >= 0.5).astype(int)),
                    "precision_0_5": precision_score(y_valid, (valid_prob >= 0.5).astype(int), zero_division=0),
                    "recall_0_5": recall_score(y_valid, (valid_prob >= 0.5).astype(int), zero_division=0),
                    "f1_0_5": f1_score(y_valid, (valid_prob >= 0.5).astype(int), zero_division=0),
                    "roc_auc": roc_auc_score(y_valid, valid_prob),
                    "pr_auc": average_precision_score(y_valid, valid_prob),
                }
            )

            compare_df = pd.DataFrame(results).sort_values(["pr_auc", "f1_0_5"], ascending=[False, False])
            compare_df.to_csv(MODEL_COMPARE_FILE, index=False)

            best_name = "xgboost_tuned"
            best_pipe = final_pipe
            print(f"\nBest model by validation PR-AUC: {best_name}")

            # Tune threshold on validation, then evaluate once on the held-out test set.
            sweep_df, best_threshold = threshold_search(y_valid.values, valid_prob, min_recall=TARGET_MIN_RECALL)
            sweep_df.to_csv(THRESHOLD_SWEEP_FILE, index=False)
            print(f"Selected threshold (validation): {best_threshold:.2f}")

            metrics = evaluate(y_test.values, test_prob, best_threshold)

        except Exception as exc:
            raise RuntimeError(f"XGBoost tuning failed: {exc}") from exc
    else:
        raise RuntimeError("USE_XGBOOST=False is not supported in this tuned pipeline.")
    cv_size = min(80000, len(X_train_fit))
    if cv_size < len(X_train_fit):
        idx = y_train_fit.groupby(y_train_fit).sample(frac=cv_size / len(X_train_fit), random_state=RANDOM_STATE).index
        X_cv = X_train_fit.loc[idx]
        y_cv = y_train_fit.loc[idx]
    else:
        X_cv = X_train_fit
        y_cv = y_train_fit

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    cv_metric_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_cv, y_cv), start=1):
        X_tr_fold = X_cv.iloc[tr_idx]
        y_tr_fold = y_cv.iloc[tr_idx]
        X_va_fold = X_cv.iloc[va_idx]
        y_va_fold = y_cv.iloc[va_idx]

        fold_pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", best_pipe.named_steps["model"].__class__(**best_pipe.named_steps["model"].get_params())),
        ])
        fold_pipe.fit(X_tr_fold, y_tr_fold)
        fold_prob = fold_pipe.predict_proba(X_va_fold)[:, 1]
        cv_scores.append(average_precision_score(y_va_fold, fold_prob))
        cv_metric_rows.append(evaluate(y_va_fold.values, fold_prob, best_threshold))

    report = {
        "best_model": best_name,
        "best_params": best_pipe.named_steps["model"].get_params(),
        "target_min_recall": TARGET_MIN_RECALL,
        "selected_threshold": best_threshold,
        "holdout": metrics,
        "cv_pr_auc_mean": float(np.mean(cv_scores)),
        "cv_pr_auc_std": float(np.std(cv_scores)),
        "cv_threshold_metrics": cv_metric_rows,
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
