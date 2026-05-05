"""
Leakage-safe insurance claim modeling pipeline.

Key fixes vs prior version:
1. Target uses CLAIM_PAID with missing treated as 0 (no claim), not row drop.
2. Train/test split happens before preprocessing fit.
3. Preprocessing is fit only on training data via sklearn Pipeline.
4. Cross-validation added for stability checks.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

MODEL_FILE = "insurance_claims_pipeline_audited.pkl"
METRICS_FILE = "model_performance_audited.csv"
CONFUSION_FILE = "confusion_matrix_audited.csv"
CV_FILE = "cross_validation_audited.csv"
PROFILE_FILE = "data_profile_audited.json"


print("=" * 72)
print("INSURANCE CLAIMS - AUDITED PIPELINE (LEAKAGE SAFE)")
print("=" * 72)

os.chdir(WORKING_DIR)
df = pd.read_csv(INPUT_FILE)
print(f"Loaded rows: {len(df):,}")

# Remove exact duplicates caused by prior file concatenation.
before_dedup = len(df)
df = df.drop_duplicates().copy()
print(f"Removed duplicates: {before_dedup - len(df):,}")
print(f"Rows after dedup: {len(df):,}")

# Correct target: missing CLAIM_PAID -> 0 means no paid claim recorded.
df["CLAIM"] = (df["CLAIM_PAID"].fillna(0) > 0).astype(int)

raw_claim_rate = float(df["CLAIM"].mean())
print(f"Claim rate: {raw_claim_rate:.4f} ({raw_claim_rate * 100:.2f}%)")
print("Class counts:")
print(df["CLAIM"].value_counts())

# Date and numeric preparation (safe, no target leakage).
df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], errors="coerce")
df["INSR_END"] = pd.to_datetime(df["INSR_END"], errors="coerce")
df["PROD_YEAR"] = pd.to_numeric(df["PROD_YEAR"], errors="coerce")

# EFFECTIVE_YR in this dataset is often 2-digit year/string; derive from INSR_BEGIN instead.
df["EFFECTIVE_YEAR_DERIVED"] = df["INSR_BEGIN"].dt.year

# Engineered features.
df["POLICY_DURATION_DAYS"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
df["VEHICLE_AGE_YEARS"] = df["EFFECTIVE_YEAR_DERIVED"] - df["PROD_YEAR"]

# Keep only model features (exclude direct leakage columns).
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

X = df[feature_cols].copy()
y = df["CLAIM"].copy()

categorical_cols = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "MAKE", "USAGE"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

# Split first, then fit transformers only on train.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain/test split:")
print(f"Train rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Train claim rate: {y_train.mean():.4f}")
print(f"Test claim rate: {y_test.mean():.4f}")

# Preprocessing + model in one pipeline.
numeric_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ]
)

model = RandomForestClassifier(
    n_estimators=80,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Holdout evaluation.
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
    "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
    "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
    "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
    "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
    "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
    "test_roc_auc": roc_auc_score(y_test, y_test_prob),
    "test_pr_auc": average_precision_score(y_test, y_test_prob),
}

print("\nHoldout metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion matrix [TN FP; FN TP]:")
print(cm)

print("\nClassification report (test):")
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

# Cross-validation on capped sample for speed/stability.
print("Running 5-fold stratified CV on sampled training data...")
cv_sample_size = min(50000, len(X_train))
if cv_sample_size < len(X_train):
    sample_idx = y_train.groupby(y_train).sample(
        frac=cv_sample_size / len(X_train),
        random_state=42,
    ).index
    X_cv = X_train.loc[sample_idx]
    y_cv = y_train.loc[sample_idx]
else:
    X_cv = X_train
    y_cv = y_train

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipeline,
    X_cv,
    y_cv,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)
print(f"CV F1 mean: {cv_scores.mean():.6f}")
print(f"CV F1 std:  {cv_scores.std():.6f}")

# Save artifacts.
joblib.dump(pipeline, MODEL_FILE)

pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)
pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Pred_0", "Pred_1"],
).to_csv(CONFUSION_FILE)

pd.DataFrame(
    {
        "fold": list(range(1, len(cv_scores) + 1)),
        "f1": cv_scores,
    }
).to_csv(CV_FILE, index=False)

profile = {
    "rows_after_dedup": int(len(df)),
    "claim_rate": raw_claim_rate,
    "class_counts": {str(k): int(v) for k, v in df["CLAIM"].value_counts().to_dict().items()},
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_claim_rate": float(y_train.mean()),
    "test_claim_rate": float(y_test.mean()),
}

with open(PROFILE_FILE, "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2)

print("\nSaved files:")
print(f"- {MODEL_FILE}")
print(f"- {METRICS_FILE}")
print(f"- {CONFUSION_FILE}")
print(f"- {CV_FILE}")
print(f"- {PROFILE_FILE}")

print("\nDone.")
