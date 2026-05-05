"""Evaluate a high-confidence (top-X%) prediction strategy without retraining.

Run from the `ml/` folder:
    python high_confidence_evaluate.py

Outputs `high_confidence_report.json` with comparison metrics.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.json"
REF_CSV = "motor_data_combined.csv"
REPORT_FILE = "high_confidence_report.json"


def get_probs(pipe, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipe.named_steps["preprocess"]
    booster = pipe.named_steps["model"]
    transformed = preprocessor.transform(X)
    expected_width = getattr(booster, "n_features_in_", transformed.shape[1])

    if transformed.shape[1] < expected_width:
        missing = expected_width - transformed.shape[1]
        if sparse.issparse(transformed):
            transformed = sparse.hstack(
                [transformed, sparse.csr_matrix((transformed.shape[0], missing), dtype=transformed.dtype)],
                format="csr",
            )
        else:
            transformed = np.hstack([transformed, np.zeros((transformed.shape[0], missing), dtype=transformed.dtype)])
    elif transformed.shape[1] > expected_width:
        transformed = transformed[:, :expected_width]

    return booster.predict_proba(transformed)[:, 1]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["INSR_BEGIN"] = pd.to_datetime(data["INSR_BEGIN"], errors="coerce")
    data["INSR_END"] = pd.to_datetime(data["INSR_END"], errors="coerce")
    data["PROD_YEAR"] = pd.to_numeric(data["PROD_YEAR"], errors="coerce")

    data["EFFECTIVE_YEAR_DERIVED"] = data["INSR_BEGIN"].dt.year
    data["POLICY_DURATION_DAYS"] = (data["INSR_END"] - data["INSR_BEGIN"]).dt.days
    data["VEHICLE_AGE_YEARS"] = data["EFFECTIVE_YEAR_DERIVED"] - data["PROD_YEAR"]
    data["POLICY_BEGIN_MONTH"] = data["INSR_BEGIN"].dt.month
    data["POLICY_BEGIN_QUARTER"] = data["INSR_BEGIN"].dt.quarter
    data["POLICY_END_MONTH"] = data["INSR_END"].dt.month
    data["POLICY_END_QUARTER"] = data["INSR_END"].dt.quarter
    data["POLICY_DURATION_MONTHS"] = data["POLICY_DURATION_DAYS"] / 30.0
    data["INSURED_VALUE_LOG"] = np.log1p(pd.to_numeric(data["INSURED_VALUE"], errors="coerce"))
    data["PREMIUM_LOG"] = np.log1p(pd.to_numeric(data["PREMIUM"], errors="coerce"))
    data["VALUE_PER_PREMIUM"] = pd.to_numeric(data["INSURED_VALUE"], errors="coerce") / (
        pd.to_numeric(data["PREMIUM"], errors="coerce").replace(0, np.nan)
    )
    data["VALUE_PER_SEAT"] = pd.to_numeric(data["INSURED_VALUE"], errors="coerce") / (
        pd.to_numeric(data["SEATS_NUM"], errors="coerce").replace(0, np.nan)
    )
    data["CAPACITY_PER_SEAT"] = pd.to_numeric(data["CARRYING_CAPACITY"], errors="coerce") / (
        pd.to_numeric(data["SEATS_NUM"], errors="coerce").replace(0, np.nan)
    )
    data["PREMIUM_PER_DAY"] = pd.to_numeric(data["PREMIUM"], errors="coerce") / (
        data["POLICY_DURATION_DAYS"].replace(0, np.nan)
    )
    data["AGE_BY_VALUE"] = data["VEHICLE_AGE_YEARS"] / (pd.to_numeric(data["INSURED_VALUE"], errors="coerce") + 1.0)

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
    return data[feature_cols]


def evaluate_binary(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main():
    os.chdir(WORKING_DIR)

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
    pipe = joblib.load(MODEL_FILE)

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        threshold = float(json.load(f).get("threshold", 0.5))

    usecols = [
        "SEX",
        "INSR_BEGIN",
        "INSR_END",
        "INSR_TYPE",
        "INSURED_VALUE",
        "PREMIUM",
        "SEATS_NUM",
        "CARRYING_CAPACITY",
        "TYPE_VEHICLE",
        "CCM_TON",
        "MAKE",
        "USAGE",
        "PROD_YEAR",
        "CLAIM_PAID",
    ]

    print("Loading reference CSV (selected cols)...")
    df = pd.read_csv(REF_CSV, usecols=usecols)
    df = df.drop_duplicates().copy()
    df["CLAIM"] = (df["CLAIM_PAID"].fillna(0) > 0).astype(int)

    X = build_features(df)
    y = df["CLAIM"].values

    print("Computing probabilities (may take a moment)...")
    probs = get_probs(pipe, X)

    # Baseline: decision threshold
    baseline_pred = (probs >= threshold).astype(int)
    baseline_metrics = evaluate_binary(y, baseline_pred)
    baseline_metrics.update({"threshold": threshold, "n_positive": int(baseline_pred.sum()), "n_total": len(baseline_pred)})

    report = {"baseline": baseline_metrics, "top_x_results": []}

    for pct in [0.10, 0.15, 0.20]:
        k = max(1, int(len(probs) * pct))
        order = np.argsort(probs)[::-1]
        top_idx = order[:k]
        preds_top = np.zeros(len(probs), dtype=int)
        preds_top[top_idx] = 1
        metrics = evaluate_binary(y, preds_top)
        metrics.update({"top_pct": pct, "k": int(k), "coverage": float(k / len(probs))})
        report["top_x_results"].append(metrics)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved report:", REPORT_FILE)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
