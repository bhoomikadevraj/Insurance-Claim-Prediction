"""
Streamlit dashboard for insurance claim prediction.

Run:
streamlit run dashboard_predict.py
"""

import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "insurance_claims_best_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.json"
REPORT_FILE = "tuned_model_report.json"
REFERENCE_CSV = "motor_data_combined.csv"
POLICY_MIN_DATE = date(2000, 1, 1)
SAMPLE_ROW = {
    "SEX": 0,
    "INSR_BEGIN": date(2013, 8, 8),
    "INSR_END": date(2014, 8, 7),
    "INSR_TYPE": 1202,
    "INSURED_VALUE": 519755.22,
    "PREMIUM": 7209.14,
    "SEATS_NUM": 4.0,
    "CARRYING_CAPACITY": 6.0,
    "TYPE_VEHICLE": "Pick-up",
    "CCM_TON": 3153.0,
    "MAKE": "NISSAN",
    "USAGE": "Own Goods",
    "PROD_YEAR": 2007.0,
    "CLAIM_PAID": 0.0,
}


def inject_premium_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 122, 92, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(73, 82, 245, 0.16), transparent 24%),
                linear-gradient(180deg, #0d1117 0%, #0f1420 100%);
        }
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        #MainMenu {
            display: none;
            visibility: hidden;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 0.55rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        div[data-testid="stMetric"] {
            background: rgba(16, 22, 35, 0.86);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetricLabel"] {
            color: rgba(226, 232, 240, 0.72);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 700;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 15, 25, 0.95), rgba(13, 19, 30, 0.98));
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        .premium-panel {
            background: rgba(16, 22, 35, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 1rem 1.15rem 0.95rem 1.15rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(10px);
            margin-bottom: 0.55rem;
        }
        .hero-title {
            font-size: 1.95rem;
            line-height: 1.05;
            margin-bottom: 0.35rem;
            color: #f8fafc;
        }
        .hero-subtitle {
            color: rgba(226, 232, 240, 0.74);
            font-size: 0.95rem;
            margin-bottom: 0.1rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.04);
            color: #e2e8f0;
            font-size: 0.82rem;
        }
        .status-pill--accent {
            background: linear-gradient(135deg, rgba(255, 122, 92, 0.22), rgba(73, 82, 245, 0.2));
            border-color: rgba(255, 255, 255, 0.14);
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 20px;
            padding: 0.95rem 1rem 0.8rem 1rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
        }
        .hero-shell {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }
        .hero-kicker {
            color: rgba(226, 232, 240, 0.7);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 0.35rem;
            margin-top: 0.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            color: rgba(226, 232, 240, 0.75);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def predict_claim_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Predict claim probability while tolerating a small feature-width mismatch.

    Some saved XGBoost artifacts were trained with a wider one-hot matrix than the
    current preprocessor emits. If that happens, pad missing columns with zeros so
    the dashboard can still score rows without retraining.
    """

    preprocessor = model.named_steps["preprocess"]
    booster = model.named_steps["model"]
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


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def claim_paid_to_label(value) -> int:
    paid = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return int(paid > 0)


def clamp_date(value: date, min_date: date, max_date: date) -> date:
    if value < min_date:
        return min_date
    if value > max_date:
        return max_date
    return value


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
    inject_premium_css()

    st.markdown(
        """
        <div class="premium-panel">
            <div class="hero-shell">
                <div class="hero-kicker">Insurance scoring workspace</div>
                <div class="hero-title">Claim Risk Dashboard</div>
                <div class="hero-subtitle">Premium scoring interface for the latest tuned model. Use threshold mode for standard decisions or top-X% for high-confidence approvals.</div>
            </div>
            <div class="pill-row">
                <span class="status-pill status-pill--accent">Latest tuned model</span>
                <span class="status-pill">High-confidence mode</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    loading = st.empty()
    loading.info("Loading model and reference data...")

    if not os.path.exists(MODEL_FILE):
        st.error("Model not found. Run ml_insurance_tuned.py first.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def load_model():
        return joblib.load(MODEL_FILE)

    model = load_model()
    options = load_reference_options()
    loading.empty()

    def compute_reference_cutoff(model, top_pct: float, sample_size: int = 50000):
        """Compute cutoff probability from reference CSV for the given top percentage.

        top_pct: 0.10 means top 10%% (highest probabilities)
        """
        if not os.path.exists(REFERENCE_CSV):
            return None
        usecols = ["SEX", "INSR_BEGIN", "INSR_END", "INSR_TYPE", "INSURED_VALUE", "PREMIUM", "SEATS_NUM", "CARRYING_CAPACITY", "TYPE_VEHICLE", "CCM_TON", "MAKE", "USAGE", "PROD_YEAR", "CLAIM_PAID"]
        ref = pd.read_csv(REFERENCE_CSV, usecols=usecols)
        if ref.empty:
            return None
        sample = ref.sample(n=min(len(ref), sample_size), random_state=42)
        Xref = build_features(sample)
        probs_ref = predict_claim_probability(model, Xref)
        # cutoff that keeps top `top_pct` fraction
        pct = 100.0 - (top_pct * 100.0)
        cutoff = float(np.percentile(probs_ref, pct))
        return cutoff

    threshold = 0.5
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.5))

    # Decision mode: use fixed threshold or top-X% high-confidence
    decision_mode = st.selectbox("Decision Mode", ["Threshold", "Top X% (High Confidence)"], key="decision_mode")
    top_pct = 0.10
    if decision_mode.startswith("Top X"):
        top_pct = st.slider("Top X% (highest-confidence predictions)", min_value=1, max_value=50, value=10, key="top_pct_slider") / 100.0

    top_row = st.columns([1.15, 1.15, 1.15, 1.15])
    top_row[0].metric("Decision Threshold", f"{threshold:.2f}")
    top_row[1].metric("Operating Mode", "Threshold" if decision_mode == "Threshold" else f"Top {int(top_pct * 100)}%")
    top_row[2].metric("Input Style", "Manual or sample row")
    top_row[3].metric("Score Flow", "Latest pipeline")

    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            rep = json.load(f)
        holdout = rep.get("holdout", {})
        with st.expander("Model snapshot", expanded=False):
            snapshot_cols = st.columns(4)
            snapshot_cols[0].metric("Best Model", rep.get("best_model", "n/a"))
            snapshot_cols[1].metric("Test Accuracy", f"{holdout.get('accuracy', 0):.1%}")
            snapshot_cols[2].metric("Claim Precision", f"{holdout.get('precision', 0):.1%}")
            snapshot_cols[3].metric("Claim Recall", f"{holdout.get('recall', 0):.1%}")
            st.caption(f"PR-AUC: {holdout.get('pr_auc', 0):.3f} | Precision = correct claim calls | Recall = claims found")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV Prediction"])

    with tab1:
        st.subheader("Single Policy")
        st.caption("Choose a mode, or load the provided sample row with one click.")

        def apply_sample_row():
            for key, value in SAMPLE_ROW.items():
                st.session_state[key] = value

        top_bar = st.columns([1, 1, 2])
        if top_bar[0].button("Load sample row", use_container_width=True):
            apply_sample_row()
            st.rerun()
        top_bar[1].write("")

        c1, c2, c3 = st.columns(3)

        sex_default = options["sex"].index(int(SAMPLE_ROW["SEX"])) if int(SAMPLE_ROW["SEX"]) in options["sex"] else 0
        sex = c1.selectbox("SEX", options["sex"], index=sex_default, key="SEX")
        insr_type = c2.selectbox("INSR_TYPE", options["insr_type"], key="INSR_TYPE")
        insured_value = c3.number_input("INSURED_VALUE", min_value=0.0, value=float(SAMPLE_ROW["INSURED_VALUE"]), key="INSURED_VALUE")

        premium = c1.number_input("PREMIUM", min_value=0.0, value=float(SAMPLE_ROW["PREMIUM"]), key="PREMIUM")
        seats_num = c2.number_input("SEATS_NUM", min_value=1.0, value=float(SAMPLE_ROW["SEATS_NUM"]), key="SEATS_NUM")
        carrying_capacity = c3.number_input("CARRYING_CAPACITY", min_value=0.0, value=float(SAMPLE_ROW["CARRYING_CAPACITY"]), key="CARRYING_CAPACITY")

        type_vehicle = c1.selectbox("TYPE_VEHICLE", options["type_vehicle"], index=options["type_vehicle"].index(SAMPLE_ROW["TYPE_VEHICLE"]) if SAMPLE_ROW["TYPE_VEHICLE"] in options["type_vehicle"] else 0, key="TYPE_VEHICLE")
        ccm_ton = c2.number_input("CCM_TON", min_value=0.0, value=float(SAMPLE_ROW["CCM_TON"]), key="CCM_TON")
        make = c3.selectbox("MAKE", options["make"], index=options["make"].index(SAMPLE_ROW["MAKE"]) if SAMPLE_ROW["MAKE"] in options["make"] else 0, key="MAKE")

        usage = c1.selectbox("USAGE", options["usage"], index=options["usage"].index(SAMPLE_ROW["USAGE"]) if SAMPLE_ROW["USAGE"] in options["usage"] else 0, key="USAGE")
        prod_year = c1.number_input(
            "PROD_YEAR",
            min_value=float(options["prod_year_min"]),
            max_value=float(options["prod_year_max"]),
            value=float(SAMPLE_ROW["PROD_YEAR"]),
            key="PROD_YEAR",
        )
        claim_paid = c2.number_input("CLAIM_PAID (known actual, optional)", min_value=0.0, value=float(SAMPLE_ROW["CLAIM_PAID"]), key="CLAIM_PAID")

        st.caption("Insurance dates")
        d1, d2 = st.columns(2)
        insr_begin = d1.date_input(
            "INSR_BEGIN",
            value=clamp_date(SAMPLE_ROW["INSR_BEGIN"], POLICY_MIN_DATE, options["policy_max_date"]),
            min_value=POLICY_MIN_DATE,
            max_value=options["policy_max_date"],
            key="INSR_BEGIN",
        )
        insr_end = d2.date_input(
            "INSR_END",
            value=clamp_date(SAMPLE_ROW["INSR_END"], POLICY_MIN_DATE, options["policy_max_date"]),
            min_value=POLICY_MIN_DATE,
            max_value=options["policy_max_date"],
            key="INSR_END",
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
            prob = float(predict_claim_probability(model, X)[0])
            if decision_mode == "Threshold":
                pred = int(prob >= threshold)
                used_cutoff = threshold
                mode_info = f"Threshold={threshold:.3f}"
            else:
                ref_cutoff = compute_reference_cutoff(model, top_pct)
                if ref_cutoff is None:
                    st.error("Reference data not available to compute Top-X% cutoff.")
                    st.stop()
                pred = int(prob >= ref_cutoff)
                used_cutoff = ref_cutoff
                mode_info = f"Top {int(top_pct*100)}% cutoff={ref_cutoff:.3f}"

            st.success("Prediction complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Claim Probability", f"{prob:.2%}")
            m2.metric("Predicted Class", "CLAIM" if pred == 1 else "NO CLAIM")
            m3.metric("Risk Level", risk_label(prob))
            actual = claim_paid_to_label(claim_paid)
            m4.metric("Actual From CLAIM_PAID", "CLAIM" if actual == 1 else "NO CLAIM")
            st.caption(mode_info)
            st.code(
                f"SEX={sex}, INSR_BEGIN={insr_begin}, INSR_END={insr_end}, INSR_TYPE={insr_type}, "
                f"INSURED_VALUE={insured_value}, PREMIUM={premium}, SEATS_NUM={seats_num}, "
                f"CARRYING_CAPACITY={carrying_capacity}, TYPE_VEHICLE={type_vehicle}, CCM_TON={ccm_ton}, "
                f"MAKE={make}, USAGE={usage}, PROD_YEAR={prod_year}",
                language="text",
            )
            if pred == actual:
                st.info("Prediction matches the CLAIM_PAID label entered for this row.")
            if claim_paid > 0 and pred == 0:
                st.warning(
                    "This row has a positive CLAIM_PAID amount, so the actual class is CLAIM. "
                    "The model predicted NO CLAIM because the probability is below the current decision rule."
                )
            if claim_paid == 0 and pred == 1:
                st.warning(
                    "This row has CLAIM_PAID as zero or blank, so the actual class is NO CLAIM. "
                    "The model predicted CLAIM because the estimated probability is above the current decision rule."
                )

    with tab2:
        st.subheader("Batch Prediction via CSV")
        st.write("Required columns: SEX, INSR_TYPE, INSURED_VALUE, PREMIUM, SEATS_NUM, CARRYING_CAPACITY, TYPE_VEHICLE, CCM_TON, MAKE, USAGE, INSR_BEGIN, INSR_END, PROD_YEAR")

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            X = build_features(raw)
            probs = predict_claim_probability(model, X)
            if decision_mode == "Threshold":
                preds = (probs >= threshold).astype(int)
                rule_label = f"Threshold >= {threshold:.3f}"
            else:
                # select top-k within the uploaded file
                k = max(1, int(len(probs) * top_pct))
                order = np.argsort(probs)[::-1]
                top_idx = order[:k]
                preds = np.zeros(len(probs), dtype=int)
                preds[top_idx] = 1
                rule_label = f"Top {int(top_pct * 100)}% highest-confidence rows"

            out = raw.copy()
            out["probability_claim"] = probs
            out["prediction"] = preds
            out["risk_level"] = pd.Series(probs).apply(risk_label)
            if "CLAIM_PAID" in out.columns:
                out["actual_from_claim_paid"] = out["CLAIM_PAID"].apply(claim_paid_to_label)
                out["prediction_match"] = out["prediction"] == out["actual_from_claim_paid"]

            st.info(rule_label)
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
