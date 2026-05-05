"""
PREDICT INSURANCE CLAIMS - USE SAVED MODEL
===========================================
This script loads the saved trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("INSURANCE CLAIMS PREDICTION - USING SAVED MODEL")
print("="*70)

working_dir = r'c:\Users\bhoom\Downloads\ml1'
os.chdir(working_dir)

# ============================================================
# LOAD SAVED MODEL AND FEATURES
# ============================================================
print("\n[LOADING] Saved Model and Configuration...")
print("-" * 70)

# Load the trained model
model_filename = 'insurance_claims_model.pkl'
if not os.path.exists(model_filename):
    print(f"❌ ERROR: Model file '{model_filename}' not found!")
    print(f"Please run 'ml_insurance_pipeline_with_save.py' first to train and save the model.")
    exit(1)

model = joblib.load(model_filename)
print(f"✓ Model loaded: {model_filename}")

# Load feature columns
feature_cols_filename = 'feature_columns.pkl'
feature_columns = joblib.load(feature_cols_filename)
print(f"✓ Feature columns loaded: {feature_cols_filename}")
print(f"  Total features expected: {len(feature_columns)}")

# ============================================================
# EXAMPLE 1: MAKE PREDICTIONS ON TEST DATA
# ============================================================
print("\n[EXAMPLE 1] Predictions on Random Test Samples...")
print("-" * 70)

# Load the combined dataset
df = pd.read_csv('motor_data_combined.csv')

# Data preparation (same as training)
df['CLAIM'] = (df['CLAIM_PAID'] > 0).astype(int)
critical_cols = ['CLAIM_PAID', 'EFFECTIVE_YR', 'PROD_YEAR', 'PREMIUM', 'INSURED_VALUE', 'INSR_BEGIN', 'INSR_END', 'SEX', 'INSR_TYPE', 'MAKE', 'TYPE_VEHICLE', 'USAGE']
df = df.dropna(subset=critical_cols)
df = df.drop_duplicates()

df['EFFECTIVE_YR'] = pd.to_numeric(df['EFFECTIVE_YR'], errors='coerce')
df['INSR_BEGIN'] = pd.to_datetime(df['INSR_BEGIN'], errors='coerce')
df['INSR_END'] = pd.to_datetime(df['INSR_END'], errors='coerce')
df['POLICY_DURATION_DAYS'] = (df['INSR_END'] - df['INSR_BEGIN']).dt.days
df['VEHICLE_AGE_YEARS'] = df['EFFECTIVE_YR'] - df['PROD_YEAR']

df['SEATS_NUM'] = df['SEATS_NUM'].fillna(df['SEATS_NUM'].median())
df['CARRYING_CAPACITY'] = df['CARRYING_CAPACITY'].fillna(0)
df['CCM_TON'] = df['CCM_TON'].fillna(df['CCM_TON'].median())

# Select features
features_to_keep = ['SEX', 'INSR_TYPE', 'INSURED_VALUE', 'PREMIUM', 'SEATS_NUM', 
                    'CARRYING_CAPACITY', 'TYPE_VEHICLE', 'CCM_TON', 'MAKE', 'USAGE', 
                    'VEHICLE_AGE_YEARS', 'POLICY_DURATION_DAYS']
X = df[features_to_keep].copy()
X_encoded = pd.get_dummies(X, columns=['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE'], drop_first=False)

# Ensure all columns match
for col in feature_columns:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[feature_columns]

print(f"✓ Data prepared for prediction")
print(f"  Total samples available: {len(X_encoded):,}")
print(f"  Features: {X_encoded.shape[1]}")

# Make predictions on random samples
n_samples = 20
sample_indices = np.random.choice(len(X_encoded), min(n_samples, len(X_encoded)), replace=False)
X_sample = X_encoded.iloc[sample_indices]
y_actual = df['CLAIM'].iloc[sample_indices].values

predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)

print(f"\n📌 Sample Predictions ({len(sample_indices)} random samples):")
print(f"\n{'#':<4} {'Actual':<12} {'Predicted':<12} {'Probability':<15} {'Risk Level':<15}")
print("-" * 60)

for i, (idx, pred, prob, actual) in enumerate(zip(sample_indices, predictions, probabilities, y_actual)):
    actual_str = "Claim" if actual == 1 else "No Claim"
    pred_str = "Claim" if pred == 1 else "No Claim"
    confidence = max(prob)
    risk_level = "🔴 HIGH (>70%)" if prob[1] > 0.7 else "🟡 MEDIUM (30-70%)" if prob[1] > 0.3 else "🟢 LOW (<30%)"
    
    print(f"{i+1:<4} {actual_str:<12} {pred_str:<12} {confidence:.2%}           {risk_level:<15}")

# ============================================================
# EXAMPLE 2: PREDICT ON CUSTOM NEW DATA
# ============================================================
print("\n\n[EXAMPLE 2] Prediction on Custom New Data...")
print("-" * 70)

# Create custom sample data
custom_data = pd.DataFrame({
    'SEX': [1, 1, 0],
    'INSR_TYPE': [1, 2, 1],
    'INSURED_VALUE': [50000, 100000, 75000],
    'PREMIUM': [500, 1200, 800],
    'SEATS_NUM': [2, 4, 2],
    'CARRYING_CAPACITY': [2000, 5000, 3000],
    'TYPE_VEHICLE': ['Tractor', 'Van', 'Tractor'],
    'CCM_TON': [3000, 2000, 2500],
    'MAKE': ['JOHN DEER', 'FORD', 'JOHN DEER'],
    'USAGE': ['Agricultural Own Farm', 'Delivery', 'Agricultural Own Farm'],
    'VEHICLE_AGE_YEARS': [5, 3, 4],
    'POLICY_DURATION_DAYS': [365, 180, 365]
})

print("\nCustom Data:")
print(custom_data)

# Encode custom data
X_custom_encoded = pd.get_dummies(custom_data, columns=['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE'], drop_first=False)

# Ensure all columns match
for col in feature_columns:
    if col not in X_custom_encoded.columns:
        X_custom_encoded[col] = 0
X_custom_encoded = X_custom_encoded[feature_columns]

# Make predictions
custom_predictions = model.predict(X_custom_encoded)
custom_probabilities = model.predict_proba(X_custom_encoded)

print(f"\n📌 Predictions for Custom Data:")
print(f"\n{'Sample':<8} {'Prediction':<12} {'Probability':<15} {'Risk Level':<20}")
print("-" * 60)

for i, (pred, prob) in enumerate(zip(custom_predictions, custom_probabilities)):
    pred_str = "🔴 CLAIM" if pred == 1 else "🟢 NO CLAIM"
    confidence = prob[1]  # Probability of claim
    risk_level = "🔴 VERY HIGH (>90%)" if confidence > 0.9 else "🔴 HIGH (70-90%)" if confidence > 0.7 else "🟡 MEDIUM (30-70%)" if confidence > 0.3 else "🟢 LOW (<30%)"
    
    print(f"{i+1:<8} {pred_str:<12} {confidence:.2%}           {risk_level:<20}")

# ============================================================
# EXAMPLE 3: BATCH PREDICTIONS WITH CSV
# ============================================================
print("\n\n[EXAMPLE 3] Batch Predictions Summary...")
print("-" * 70)

# Get predictions for all data
all_predictions = model.predict(X_encoded)
all_probabilities = model.predict_proba(X_encoded)

print(f"\n✓ Batch predictions completed on {len(X_encoded):,} samples")
print(f"\nPrediction Distribution:")
print(f"  - Predicted Claims: {(all_predictions == 1).sum():,} ({(all_predictions == 1).sum()/len(all_predictions)*100:.2f}%)")
print(f"  - Predicted No Claims: {(all_predictions == 0).sum():,} ({(all_predictions == 0).sum()/len(all_predictions)*100:.2f}%)")

print(f"\nRisk Level Distribution:")
high_risk = (all_probabilities[:, 1] > 0.7).sum()
medium_risk = ((all_probabilities[:, 1] > 0.3) & (all_probabilities[:, 1] <= 0.7)).sum()
low_risk = (all_probabilities[:, 1] <= 0.3).sum()

print(f"  - 🔴 HIGH RISK (>70%): {high_risk:,} ({high_risk/len(all_predictions)*100:.2f}%)")
print(f"  - 🟡 MEDIUM RISK (30-70%): {medium_risk:,} ({medium_risk/len(all_predictions)*100:.2f}%)")
print(f"  - 🟢 LOW RISK (<30%): {low_risk:,} ({low_risk/len(all_predictions)*100:.2f}%)")

# Save batch predictions to CSV
predictions_df = pd.DataFrame({
    'SAMPLE_ID': range(1, len(all_predictions) + 1),
    'PREDICTION': all_predictions,
    'PROBABILITY_NO_CLAIM': all_probabilities[:, 0],
    'PROBABILITY_CLAIM': all_probabilities[:, 1],
    'RISK_LEVEL': pd.cut(all_probabilities[:, 1], 
                         bins=[0, 0.3, 0.7, 1.0], 
                         labels=['LOW', 'MEDIUM', 'HIGH'])
})

predictions_df.to_csv('batch_predictions.csv', index=False)
print(f"\n✓ Batch predictions saved to 'batch_predictions.csv'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)

print(f"""
✓ Model Successfully Loaded and Used!

📊 MODEL INFORMATION:
  - Model Type: Random Forest Classifier
  - Number of Features: {len(feature_columns)}
  - Total Samples Tested: {len(X_encoded):,}

📈 PREDICTION RESULTS:
  - Random Test Samples: 20 predictions made
  - Custom Data Samples: 3 predictions made
  - Batch Predictions: {len(X_encoded):,} predictions made

📁 OUTPUT FILES:
  - batch_predictions.csv - All predictions with probabilities

🔄 NEXT STEPS:
  1. Run 'visualize_performance.py' to see training vs testing performance
  2. Use batch_predictions.csv for further analysis
  3. Deploy model for real-time predictions on new insurance policies
""")

print("="*70)
print("PREDICTION COMPLETED SUCCESSFULLY!")
print("="*70)
