# Insurance Claim Risk Prediction Model

A production-ready machine learning system for predicting insurance claim risk using XGBoost. This project includes a tuned classification model, Streamlit dashboard, and comprehensive evaluation tools for both single-policy scoring and batch predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Requirements](#requirements)

## Overview

This project builds a **tuned XGBoost classifier** that predicts whether a motor insurance policy is likely to result in a claim. The model addresses the challenge of highly imbalanced classification (7.5% positive class rate) through:

- Custom XGBoost tuning with `scale_pos_weight` for class imbalance handling
- Validation-based optimization using PR-AUC (Precision-Recall Area Under Curve)
- Intelligent threshold selection balancing recall and precision
- High-confidence filtering for conservative approval workflows

## Features

✅ **Dual Decision Strategies**
- Threshold-based prediction for broad claim detection (65%+ recall)
- Top-X% high-confidence filtering for precision-focused workflows (25%+ precision at 10% coverage)

✅ **Production-Ready Pipeline**
- Leakage-safe preprocessing with consistent train/inference alignment
- Feature engineering for policy duration, vehicle age, and value ratios
- Serialized pipeline for direct scoring without retraining

✅ **Interactive Streamlit Dashboard**
- Single policy entry with instant risk scoring
- Batch CSV scoring for large-scale evaluation
- Real-time strategy switching (threshold vs. high-confidence mode)
- Visual probability distributions and decision metrics

✅ **Comprehensive Evaluation**
- Confusion matrices and classification metrics
- Threshold sweep analysis
- Top-X% coverage curves
- Feature importance rankings

## Installation

### Clone the Repository

```bash
git clone https://github.com/bhoomikadevraj/ml.git
cd ml
```

### Set Up Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Dashboard Locally

```bash
streamlit run dashboard_predict.py
```

The dashboard will open at `http://localhost:8501` with:
1. **Manual Input Tab** - Enter policy details and get instant risk score
2. **Batch Scoring Tab** - Upload CSV files for bulk prediction
3. **Strategy Toggle** - Switch between threshold and high-confidence modes

### Sample Policy Input

| Field | Example |
|-------|---------|
| Vehicle Age (years) | 5 |
| Policy Duration (days) | 365 |
| Insured Value | 15000 |
| Premium | 800 |
| Seats | 5 |
| Usage | Commercial |

### Batch CSV Format

Your CSV file should contain the same columns as the training data:
- `SEX`, `INSR_TYPE`, `USAGE`, `TYPE_VEHICLE`, `MAKE`
- `INSR_BEGIN`, `INSR_END` (date columns)
- `INSURED_VALUE`, `PREMIUM`, `SEATS_NUM`, `CARRYING_CAPACITY`, `CCM_TON`, `PROD_YEAR`

## Model Details

### Architecture

- **Algorithm**: XGBoost Classifier
- **Framework**: scikit-learn Pipeline
- **Training Data**: 641,470 motor insurance policies
- **Test Data**: 160,368 policies (holdout evaluation)
- **Positive Class Rate**: 7.50% (highly imbalanced)

### Feature Engineering

**Derived Features:**
- `VEHICLE_AGE_YEARS` - Vehicle wear and risk proxy
- `POLICY_DURATION_DAYS` - Coverage length
- `POLICY_DURATION_MONTHS` - Seasonal patterns
- `POLICY_BEGIN_MONTH`, `POLICY_BEGIN_QUARTER` - Temporal features
- `INSURED_VALUE_LOG`, `PREMIUM_LOG` - Normalized large-value fields
- `VALUE_PER_PREMIUM` - Policy value intensity
- `PREMIUM_PER_DAY` - Premium cost per coverage day
- `VALUE_PER_SEAT` - Value distribution per seat
- `CAPACITY_PER_SEAT` - Capacity per passenger
- `AGE_BY_VALUE` - Vehicle age relative to insured value

**Preprocessing:**
- Duplicate row removal
- Numeric imputation: median
- Categorical imputation: most frequent value
- One-hot encoding for categorical features

### Hyperparameters

```python
{
  'n_estimators': 100,
  'max_depth': 5,
  'learning_rate': 0.1,
  'scale_pos_weight': 12.3,  # Inverse of class ratio
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'random_state': 42
}
```

## Results

### Threshold-Based Operating Point (threshold = 0.6154)

| Metric | Value |
|--------|-------|
| Accuracy | 72.54% |
| Precision | 16.73% |
| Recall | 65.73% |
| F1 Score | 26.67% |
| ROC AUC | 0.778 |
| PR AUC | 0.194 |

**Confusion Matrix:**
```
              Predicted_No    Predicted_Yes
Actual_No     109,012         39,335
Actual_Yes    4,119           7,902
```

### High-Confidence Top-X% Strategy

| Coverage | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Top 10% | 25.98% | 28.02% | 26.96% |
| Top 15% | 24.00% | 38.82% | 29.66% |
| Top 20% | 22.66% | 48.88% | 30.97% |

### Interpretation

- **Threshold Mode**: Best for broad claim screening where recall > 65% is essential
- **High-Confidence Mode**: Best for conservative approval workflows where precision matters more
- **Ranking Power**: ROC AUC of 0.778 indicates strong ability to rank claims by risk

## Project Structure

```
ml/
├── dashboard_predict.py              # Main Streamlit app
├── ml_insurance_tuned.py             # Model training & evaluation script
├── requirements.txt                   # Python dependencies
├── insurance_claims_best_pipeline.pkl # Trained model artifact (LFS)
├── best_threshold.json                # Optimal decision threshold
├── tuned_model_report.json            # Model performance metrics
├── high_confidence_report.json        # Top-X% evaluation results
├── motor_data_combined.csv            # Training dataset
├── report_assets/                     # Generated visualizations
│   ├── confusion_matrix_tuned.png
│   ├── threshold_sweep_curve.png
│   ├── top_x_curve.png
│   └── feature_importance_tuned.png
└── README.md                          # This file
```

## Deployment

### Streamlit Community Cloud

1. **Push to GitHub** (already done to `https://github.com/bhoomikadevraj/ml`)

2. **Go to Streamlit Community Cloud** at https://share.streamlit.io

3. **Create New App:**
   - Repository: `bhoomikadevraj/ml`
   - Branch: `main`
   - Main file: `dashboard_predict.py`

4. **Deploy** - The app will automatically use the saved model artifacts

### Environment Variables (Optional)

For production deployments, consider adding:
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_CLIENT_LOGGING_LEVEL=info
```

### Required Artifacts

Ensure these files are committed to the repository:
- `dashboard_predict.py` - Streamlit app
- `requirements.txt` - Dependencies
- `insurance_claims_best_pipeline.pkl` - Model (stored via Git LFS)
- `best_threshold.json` - Decision threshold
- `tuned_model_report.json` - Performance metrics
- `high_confidence_report.json` - Top-X% results

## Requirements

### Python Version
- Python 3.8+

### Core Dependencies
```
streamlit >= 1.28
scikit-learn >= 1.3
xgboost >= 2.0
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

See `requirements.txt` for complete list with pinned versions.

### Hardware
- **Minimum**: 2GB RAM, multi-core processor
- **Recommended**: 4GB+ RAM for batch scoring large datasets

## Performance Notes

- **Single Policy Scoring**: < 100ms latency
- **Batch CSV Scoring**: ~1-2ms per row
- **Model Size**: ~128MB (includes LFS artifacts)

## Known Limitations

- Dataset remains strongly imbalanced (7.5% positive class)
- Positive class precision constrained by class imbalance
- Limited feature engineering - future improvements possible through domain-specific features
- No calibration currently applied to probability scores

## Future Improvements

- Calibration methods (Platt scaling, isotonic regression)
- Second-stage review rules for borderline cases
- Additional feature engineering from domain experts
- A/B testing framework for threshold optimization
- Real-time model performance monitoring

## License

This project is provided as-is for educational and evaluation purposes.

## Contact

For questions or collaboration inquiries, please reach out through GitHub issues or pull requests.
