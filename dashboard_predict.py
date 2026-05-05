"""
Streamlit dashboard for insurance claim prediction.

Run:
streamlit run dashboard_predict.py
"""

import json
import os
import calendar
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

WORKING_DIR = r"c:\Users\bhoom\Downloads\ml1"
MODEL_FILE = "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.json"
REPORT_FILE = "tuned_model_report.json"
REFERENCE_CSV = "motor_data_combined.csv"
POLICY_MIN_DATE = date(2000, 1, 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["INSR_BEGIN"] = pd.to_datetime(data["INSR_BEGIN"], errors="coerce")
    data["INSR_END"] = pd.to_datetime(data["INSR_END"], errors="coerce")
    data["PROD_YEAR"] = pd.to_numeric(data["PROD_YEAR"], errors="coerce")

    data["EFFECTIVE_YEAR_DERIVED"] = data["INSR_BEGIN"].dt.year
    data["POLICY_DURATION_DAYS"] = (data["INSR_END"] - data["INSR_BEGIN"]).dt.days
    data["VEHICLE_AGE_YEARS"] = data["EFFECTIVE_YEAR_DERIVED"] - data["PROD_YEAR"]

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
    return data[feature_cols]


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def claim_paid_to_label(value) -> int:
    paid = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return int(paid > 0)


def date_selectors(container, label: str, default_date: date, min_date: date, max_date: date) -> date:
    years = list(range(min_date.year, max_date.year + 1))
    default_year = min(max(default_date.year, min_date.year), max_date.year)
    year = container.selectbox(f"{label} YEAR", years, index=years.index(default_year))

    min_month = min_date.month if year == min_date.year else 1
    max_month = max_date.month if year == max_date.year else 12
    months = list(range(min_month, max_month + 1))
    default_month = min(max(default_date.month, min_month), max_month)
    month = container.selectbox(f"{label} MONTH", months, index=months.index(default_month))

    max_day_for_month = calendar.monthrange(year, month)[1]
    min_day = min_date.day if year == min_date.year and month == min_date.month else 1
    max_day = max_date.day if year == max_date.year and month == max_date.month else max_day_for_month
    days = list(range(min_day, max_day + 1))
    default_day = min(max(default_date.day, min_day), max_day)
    day = container.selectbox(f"{label} DAY", days, index=days.index(default_day))
    return date(year, month, day)


@st.cache_data(show_spinner=False)
def load_reference_options() -> dict:
    defaults = {
        "sex": [0, 1, 2],
        "insr_type": [1202, 1201, 1204],
        "type_vehicle": ["Automobile", "Pick-up", "Truck", "Motor-cycle"],
        "make": ["TOYOTA", "ISUZU", "NISSAN", "MITSUBISHI"],
        "usage": ["Private", "Own Goods", "General Cartage", "Fare Paying Passengers"],
        "prod_year_min": 1950,
        "prod_year_max": 2018,
        "policy_max_date": date(2019, 12, 31),
        "policy_default_begin": date(2013, 1, 1),
        "policy_default_end": date(2014, 1, 1),
    }

    if not os.path.exists(REFERENCE_CSV):
        return defaults

    cols = ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "MAKE", "USAGE", "PROD_YEAR", "INSR_BEGIN", "INSR_END"]
    ref = pd.read_csv(REFERENCE_CSV, usecols=cols)

    def ordered_values(col: str, limit: int | None = None) -> list:
        values = (
            ref[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )
        result = values.index.tolist()
        return result[:limit] if limit else result

    prod_year = pd.to_numeric(ref["PROD_YEAR"], errors="coerce").dropna()
    begin_dates = pd.to_datetime(ref["INSR_BEGIN"], errors="coerce", format="%d-%b-%y").dropna()
    end_dates = pd.to_datetime(ref["INSR_END"], errors="coerce", format="%d-%b-%y").dropna()
    all_policy_dates = pd.concat([begin_dates, end_dates], ignore_index=True)

    if not prod_year.empty:
        defaults["prod_year_min"] = int(prod_year.min())
        defaults["prod_year_max"] = int(prod_year.max())
    if not all_policy_dates.empty:
        defaults["policy_max_date"] = all_policy_dates.max().date()
    if not begin_dates.empty:
        default_begin = begin_dates.min().date()
        defaults["policy_default_begin"] = max(default_begin, POLICY_MIN_DATE)
        defaults["policy_default_end"] = min(
            date(defaults["policy_default_begin"].year + 1, defaults["policy_default_begin"].month, defaults["policy_default_begin"].day),
            defaults["policy_max_date"],
        )

    defaults["sex"] = sorted(pd.to_numeric(ref["SEX"], errors="coerce").dropna().astype(int).unique().tolist())
    defaults["insr_type"] = sorted(pd.to_numeric(ref["INSR_TYPE"], errors="coerce").dropna().astype(int).unique().tolist())
    defaults["type_vehicle"] = ordered_values("TYPE_VEHICLE")
    defaults["make"] = ordered_values("MAKE")
    defaults["usage"] = ordered_values("USAGE")
    return defaults


def main():
    os.chdir(WORKING_DIR)

    st.set_page_config(page_title="Insurance Claim Risk Dashboard", page_icon="📊", layout="wide")

    st.title("Insurance Claim Risk Dashboard")
    st.caption("Predict claim risk with tuned model and business threshold")

    if not os.path.exists(MODEL_FILE):
        st.error("Model not found. Run ml_insurance_tuned.py first.")
        st.stop()

    model = joblib.load(MODEL_FILE)
    options = load_reference_options()

    threshold = 0.5
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Decision Threshold", f"{threshold:.2f}")

    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            rep = json.load(f)
        holdout = rep.get("holdout", {})
        col2.metric("Best Model", rep.get("best_model", "n/a"))
        col3.metric("Test Accuracy", f"{holdout.get('accuracy', 0):.1%}")
        col4.metric("Claim Precision", f"{holdout.get('precision', 0):.1%}")
        col5.metric("Claim Recall", f"{holdout.get('recall', 0):.1%}")
        col6.metric("PR-AUC", f"{holdout.get('pr_auc', 0):.3f}")
        st.caption(
            "Precision means how often CLAIM predictions were correct on the test set. "
            "Recall means how many real claims the model found."
        )

    tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV Prediction"])

    with tab1:
        st.subheader("Single Policy")
        c1, c2, c3 = st.columns(3)

        sex_default = options["sex"].index(1) if 1 in options["sex"] else 0
        sex = c1.selectbox("SEX", options["sex"], index=sex_default)
        insr_type = c2.selectbox("INSR_TYPE", options["insr_type"])
        insured_value = c3.number_input("INSURED_VALUE", min_value=0.0, value=50000.0)

        premium = c1.number_input("PREMIUM", min_value=0.0, value=1000.0)
        seats_num = c2.number_input("SEATS_NUM", min_value=1.0, value=4.0)
        carrying_capacity = c3.number_input("CARRYING_CAPACITY", min_value=0.0, value=1000.0)

        type_vehicle = c1.selectbox("TYPE_VEHICLE", options["type_vehicle"])
        ccm_ton = c2.number_input("CCM_TON", min_value=0.0, value=1500.0)
        make_index = options["make"].index("TOYOTA") if "TOYOTA" in options["make"] else 0
        make = c3.selectbox("MAKE", options["make"], index=make_index)

        usage_index = options["usage"].index("Private") if "Private" in options["usage"] else 0
        usage = c1.selectbox("USAGE", options["usage"], index=usage_index)
        prod_year = c1.number_input(
            "PROD_YEAR",
            min_value=float(options["prod_year_min"]),
            max_value=float(options["prod_year_max"]),
            value=float(options["prod_year_max"]),
        )
        claim_paid = c2.number_input("CLAIM_PAID (known actual, optional)", min_value=0.0, value=0.0)

        st.caption("Insurance dates")
        d1, d2 = st.columns(2)
        insr_begin = date_selectors(
            d1,
            "INSR_BEGIN",
            options["policy_default_begin"],
            POLICY_MIN_DATE,
            options["policy_max_date"],
        )
        insr_end = date_selectors(
            d2,
            "INSR_END",
            options["policy_default_end"],
            POLICY_MIN_DATE,
            options["policy_max_date"],
        )

        if st.button("Predict Risk", type="primary"):
            if insr_end <= insr_begin:
                st.error("INSR_END must be after INSR_BEGIN. Check the Excel row date before predicting.")
                st.stop()

            single = pd.DataFrame(
                [
                    {
                        "SEX": sex,
                        "INSR_TYPE": insr_type,
                        "INSURED_VALUE": insured_value,
                        "PREMIUM": premium,
                        "SEATS_NUM": seats_num,
                        "CARRYING_CAPACITY": carrying_capacity,
                        "TYPE_VEHICLE": type_vehicle,
                        "CCM_TON": ccm_ton,
                        "MAKE": make,
                        "USAGE": usage,
                        "INSR_BEGIN": str(insr_begin),
                        "INSR_END": str(insr_end),
                        "PROD_YEAR": prod_year,
                        "CLAIM_PAID": claim_paid,
                    }
                ]
            )

            X = build_features(single)
            prob = float(model.predict_proba(X)[:, 1][0])
            pred = int(prob >= threshold)

            st.success("Prediction complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Claim Probability", f"{prob:.2%}")
            m2.metric("Predicted Class", "CLAIM" if pred == 1 else "NO CLAIM")
            m3.metric("Risk Level", risk_label(prob))
            actual = claim_paid_to_label(claim_paid)
            m4.metric("Actual From CLAIM_PAID", "CLAIM" if actual == 1 else "NO CLAIM")
            if pred == actual:
                st.info("Prediction matches the CLAIM_PAID label entered for this row.")
            if claim_paid > 0 and pred == 0:
                st.warning(
                    "This row has a positive CLAIM_PAID amount, so the actual class is CLAIM. "
                    "The model predicted NO CLAIM because the probability is below the current decision threshold."
                )
            if claim_paid == 0 and pred == 1:
                st.warning(
                    "This row has CLAIM_PAID as zero or blank, so the actual class is NO CLAIM. "
                    "The model predicted CLAIM because the estimated probability is above the current decision threshold."
                )

    with tab2:
        st.subheader("Batch Prediction via CSV")
        st.write("Required columns: SEX, INSR_TYPE, INSURED_VALUE, PREMIUM, SEATS_NUM, CARRYING_CAPACITY, TYPE_VEHICLE, CCM_TON, MAKE, USAGE, INSR_BEGIN, INSR_END, PROD_YEAR")

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            X = build_features(raw)
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)

            out = raw.copy()
            out["probability_claim"] = probs
            out["prediction"] = preds
            out["risk_level"] = pd.Series(probs).apply(risk_label)
            if "CLAIM_PAID" in out.columns:
                out["actual_from_claim_paid"] = out["CLAIM_PAID"].apply(claim_paid_to_label)
                out["prediction_match"] = out["prediction"] == out["actual_from_claim_paid"]

            st.dataframe(out.head(20), use_container_width=True)

            st.download_button(
                "Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="dashboard_predictions.csv",
                mime="text/csv",
            )

            st.write("Prediction summary")
            summary = out["prediction"].value_counts().rename(index={0: "NO_CLAIM", 1: "CLAIM"})
            st.bar_chart(summary)


if __name__ == "__main__":
    main()
