"""
Predict with the audited leakage-safe pipeline.
"""

import os
import joblib
import pandas as pd

WORKING_DIR = r"c:\Users\bhoom\Downloads\ml1"
MODEL_FILE = "insurance_claims_pipeline_audited.pkl"
INPUT_FILE = "motor_data_combined.csv"
OUTPUT_FILE = "batch_predictions_audited.csv"

os.chdir(WORKING_DIR)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")

pipeline = joblib.load(MODEL_FILE)

df = pd.read_csv(INPUT_FILE).drop_duplicates().copy()

# Build features in the same semantic form expected by training.
df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], errors="coerce")
df["INSR_END"] = pd.to_datetime(df["INSR_END"], errors="coerce")
df["PROD_YEAR"] = pd.to_numeric(df["PROD_YEAR"], errors="coerce")

df["EFFECTIVE_YEAR_DERIVED"] = df["INSR_BEGIN"].dt.year
df["POLICY_DURATION_DAYS"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
df["VEHICLE_AGE_YEARS"] = df["EFFECTIVE_YEAR_DERIVED"] - df["PROD_YEAR"]

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
proba = pipeline.predict_proba(X)[:, 1]
pred = (proba >= 0.5).astype(int)

out = pd.DataFrame(
    {
        "prediction": pred,
        "probability_claim": proba,
        "risk_level": pd.cut(
            proba,
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        ),
    }
)
out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(out["prediction"].value_counts())
