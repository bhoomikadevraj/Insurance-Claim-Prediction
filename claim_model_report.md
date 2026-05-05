# Table of Contents
1. Introduction
2. Objectives of the Project
3. Dataset Description
4. Data Preprocessing
5. Model Implementation
6. Results and Analysis
7. Conclusion
8. Appendix
9. Deployment Notes

# 1. Introduction

This project builds an insurance claim risk model that predicts whether a policy is likely to result in a claim. The latest version uses a tuned XGBoost pipeline trained on engineered policy, vehicle, and timing features. The goal is not just accuracy, but practical decision support for a highly imbalanced dataset where claim precision and recall matter more than a single headline score.

In a problem like this, a model can look acceptable on accuracy while still making too many false claim predictions. That is why the report focuses on precision-recall behavior, threshold choice, and high-confidence selection instead of accuracy alone. The intent is to support a real workflow where a user enters policy details and gets a risk estimate that can be acted on immediately.

The dashboard has been updated to use the latest saved model artifact and supports two decision styles:

- **Threshold mode** for standard binary prediction.
- **Top-X% high-confidence mode** for situations where only the strongest predictions should be accepted.

# 2. Objectives of the Project

The main objectives were:

- Predict claim occurrence from policy records with a leakage-safe machine learning pipeline.
- Improve model quality on an imbalanced dataset without relying on retraining for every operating point.
- Support a business-friendly decision workflow in the Streamlit dashboard.
- Compare a fixed threshold strategy against a high-confidence top-X% strategy.
- Produce clear charts and summary artifacts that can be used in reporting and presentation.

# 3. Dataset Description

The source data comes from `motor_data_combined.csv` and contains policy and vehicle attributes such as:

- `SEX`
- `INSR_BEGIN`, `INSR_END`
- `INSR_TYPE`
- `INSURED_VALUE`
- `PREMIUM`
- `SEATS_NUM`
- `CARRYING_CAPACITY`
- `TYPE_VEHICLE`
- `CCM_TON`
- `MAKE`
- `USAGE`
- `PROD_YEAR`
- `CLAIM_PAID`

The label is derived as:

```text
CLAIM = 1 if CLAIM_PAID > 0 else 0
```

Key dataset facts from the latest tuned run:

| Item | Value |
| --- | ---: |
| Training rows | 641,470 |
| Test rows | 160,368 |
| Train claim rate | 7.50% |
| Test claim rate | 7.50% |

This is a strongly imbalanced classification problem, so precision-recall behavior is more important than raw accuracy alone.

The dataset combines structured policy information with simple vehicle descriptors. Some fields are directly numeric, while others are categorical and need one-hot encoding before modeling. The date columns are especially useful because they let us derive policy duration and vehicle age, both of which can capture risk patterns that are not obvious from the raw fields alone.

The target class is sparse, which means a model can achieve good-looking accuracy by predicting the majority class too often. For that reason, the project tracks confusion-matrix counts, recall, and PR-AUC alongside accuracy.

# 4. Data Preprocessing

The preprocessing pipeline is designed to be leakage-safe and consistent between training and inference.

Steps applied:

1. Duplicate rows are removed.
2. `CLAIM` is created from `CLAIM_PAID`.
3. Dates are parsed from `INSR_BEGIN` and `INSR_END`.
4. Numeric conversions are applied to policy and vehicle fields.
5. Feature engineering creates several derived variables:
   - `VEHICLE_AGE_YEARS`
   - `POLICY_DURATION_DAYS`
   - `POLICY_BEGIN_MONTH`
   - `POLICY_BEGIN_QUARTER`
   - `POLICY_END_MONTH`
   - `POLICY_END_QUARTER`
   - `POLICY_DURATION_MONTHS`
   - `INSURED_VALUE_LOG`
   - `PREMIUM_LOG`
   - `VALUE_PER_PREMIUM`
   - `VALUE_PER_SEAT`
   - `CAPACITY_PER_SEAT`
   - `PREMIUM_PER_DAY`
   - `AGE_BY_VALUE`
6. Numeric fields are imputed with the median.
7. Categorical fields are imputed with the most frequent value and one-hot encoded.

This structure keeps the dashboard aligned with the training pipeline and reduces the risk of train/inference mismatches.

Why these features help:

- `VEHICLE_AGE_YEARS` gives a simple proxy for vehicle wear and risk profile.
- `POLICY_DURATION_DAYS` and the month/quarter features help capture seasonality and policy period effects.
- The log transforms make large-value fields easier for the model to use.
- Ratio features such as `VALUE_PER_PREMIUM` and `PREMIUM_PER_DAY` help the model compare policy value with cost intensity.

The dashboard uses the same feature-building logic as the training pipeline, which is important because even small mismatches can cause scoring errors or inconsistent probabilities.

# 5. Model Implementation

The final model is an **XGBoost classifier** trained inside a scikit-learn pipeline. The pipeline combines the shared preprocessing step with the fitted model so the dashboard can score input rows directly.

Implementation highlights:

- Tuned XGBoost with class imbalance handling using `scale_pos_weight`.
- Validation-based optimization using PR-AUC.
- Threshold selection based on a recall-aware sweep.
- Saved artifacts for production scoring:
  - `insurance_claims_best_pipeline.pkl`
  - `best_threshold.json`
  - `tuned_model_report.json`

The dashboard also includes a high-confidence strategy that does not retrain the model. Instead, it ranks predictions by probability and keeps only the top fraction of highest-confidence rows.

Operationally, that means the app can be used in two ways:

- If you want coverage, use the threshold rule and score every row.
- If you want stronger precision, use top-X% and only keep the most confident rows.

This makes the same model useful in both broad screening and conservative approval settings.

Generated report charts are stored in `report_assets/`.

![Confusion Matrix](report_assets/confusion_matrix_tuned.png)

![Threshold Sweep](report_assets/threshold_sweep_curve.png)

![Top-X% High-Confidence Strategy](report_assets/top_x_curve.png)

![Feature Importance](report_assets/feature_importance_tuned.png)

# 6. Results and Analysis

## Threshold-Based Operating Point

The current selected threshold is **0.6154**.

Holdout performance at this threshold:

| Metric | Value |
| --- | ---: |
| Accuracy | 72.54% |
| Precision | 16.73% |
| Recall | 65.73% |
| F1 | 26.67% |
| ROC AUC | 0.778 |
| PR AUC | 0.194 |

The confusion matrix for this operating point is:

|  | Pred_0 | Pred_1 |
| --- | ---: | ---: |
| Actual_0 | 109,012 | 39,335 |
| Actual_1 | 4,119 | 7,902 |

This threshold gives moderate recall, but precision remains low because the dataset is highly imbalanced and many policy rows look similar in feature space.

In practical terms, this operating point is good when missing a real claim is more costly than reviewing a false alarm. It is less ideal when the business wants only a small set of highly reliable claim flags.

## High-Confidence Top-X% Strategy

Instead of using a fixed cutoff, the dashboard can accept only the highest-confidence predictions.

Comparison on the latest reference evaluation:

| Strategy | Coverage | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Top 10% | 10.0% | 25.98% | 28.02% | 26.96% |
| Top 15% | 15.0% | 24.00% | 38.82% | 29.66% |
| Top 20% | 20.0% | 22.66% | 48.88% | 30.97% |

Interpretation:

- Top 10% gives the highest precision, which is useful when the business wants only the safest approvals.
- As coverage increases, recall improves, but precision naturally declines.
- This is a good tradeoff when the operational goal is to reduce false positives rather than classify every policy.

The top-X% method is especially useful when a downstream team can only review a limited number of cases. Instead of forcing every row through the same decision boundary, it focuses on the strongest signals and leaves the rest unflagged.

## Model Comparison

The tuned XGBoost model is the current best performer in the saved comparison artifact.

| Model | Accuracy @ 0.5 | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost_tuned | 63.88% | 15.56% | 86.22% | 26.37% | 0.810 | 0.241 |

This indicates strong ranking power, even if the raw positive class precision remains constrained by the imbalance of the problem.

That result is important because it shows the model can still rank risky policies reasonably well even when the final binary decision depends heavily on threshold or top-X selection. In other words, the score is more informative than the default class label by itself.

## Dashboard Use

The Streamlit app now reflects the latest model artifacts and supports the following workflow:

1. Enter a single policy manually.
2. Load the sample row with one click for a quick test.
3. Switch between threshold mode and top-X% high-confidence mode.
4. Review the score, predicted class, and the current decision rule.
5. Upload a CSV for batch scoring if needed.

This is useful because it lets a user move from experimentation to a repeatable decision interface without rebuilding the model.

# 7. Conclusion

The latest model and dashboard are now aligned around a practical decision workflow.

Main takeaways:

- The tuned XGBoost pipeline is the latest production-ready model.
- A fixed threshold around 0.6154 gives balanced claim detection with recall above 65%.
- The high-confidence top-X% approach improves precision when the application can accept fewer predictions.
- The dashboard now supports both operating styles, and the report includes charts generated from the latest model outputs.

Overall, the project is now better suited for real use because it combines a reusable model artifact, a clear dashboard, and two decision strategies depending on how conservative the business wants to be.

The main limitation is that the data remains imbalanced, so precision is still not high enough to treat every claim alert as equally reliable. The high-confidence strategy helps solve that by only accepting the strongest predictions. If the project is extended later, the next best improvements would be better feature engineering, calibration, or a second-stage review rule for borderline cases.

# 8. Appendix

## A. Saved Artifacts

The project keeps the main model and evaluation outputs as files in the workspace so the dashboard and report stay synchronized:

| Artifact | Purpose |
| --- | --- |
| `insurance_claims_best_pipeline.pkl` | Saved preprocessing + XGBoost pipeline |
| `best_threshold.json` | Selected threshold for standard prediction mode |
| `tuned_model_report.json` | Final holdout metrics and model summary |
| `high_confidence_report.json` | Top-X% evaluation results |
| `threshold_sweep.csv` | Validation threshold sweep |
| `confusion_matrix_tuned.csv` | Confusion matrix for the selected threshold |
| `report_assets/*.png` | Charts used in this report |

## B. Practical Interpretation

The model is best viewed as a ranking system first and a binary classifier second. The probability score helps decide which rows are more likely to be claims, and the threshold or top-X rule converts that score into a business action.

- A lower threshold increases recall but can flood the user with false positives.
- A higher threshold or top-X rule increases precision but leaves more cases unflagged.
- The dashboard therefore supports both screening and conservative review.

## C. Known Limitation

The dataset is still strongly imbalanced, and the positive class is hard to separate cleanly. That is why the best precision is still modest even though the ranking metrics are respectable. This is normal for claim-style problems with limited positive examples.

# 9. Deployment Notes

The app is ready for public Streamlit deployment with the current dashboard file.

What is needed:

1. Push the folder to GitHub.
2. Keep `dashboard_predict.py` as the main Streamlit entry point.
3. Make sure `requirements.txt` is present.
4. Connect the repository to Streamlit Community Cloud.

Recommended deploy file:

- `dashboard_predict.py`

The public app should load the same latest model artifact and the same threshold file that are already used locally. That means the deployed version will behave the same way as the current dashboard.

If the repository is updated later, redeploying Streamlit Cloud should automatically pick up the changes after a new push.
