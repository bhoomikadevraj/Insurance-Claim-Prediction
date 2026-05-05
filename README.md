# Insurance Claim Risk Dashboard

This project contains a tuned insurance claim prediction model and a Streamlit dashboard for scoring single policies or CSV batches.

## Run locally

```bash
cd c:\Users\heman\Downloads\ml\ml
streamlit run dashboard_predict.py
```

## Public deployment

This app is ready for Streamlit Community Cloud.

Required files:
- `dashboard_predict.py`
- `requirements.txt`
- `insurance_claims_best_pipeline.pkl`
- `best_threshold.json`
- `tuned_model_report.json`
- `high_confidence_report.json`
- `report_assets/`

Suggested deployment steps:
1. Push this folder to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Set the main file to `dashboard_predict.py`.
4. Deploy.

## Report

The model report is in `claim_model_report.md` and includes the latest charts and evaluation summary.
