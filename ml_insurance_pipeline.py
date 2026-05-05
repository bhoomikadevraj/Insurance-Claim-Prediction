"""
INSURANCE CLAIMS PREDICTION - MACHINE LEARNING PIPELINE
========================================================
Binary Classification: Will a policy result in a claim? (Yes=1, No=0)
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("INSURANCE CLAIMS PREDICTION - ML PIPELINE")
print("="*70)

# ============================================================
# STEP 1 & 2: LOAD AND EXPLORE DATA
# ============================================================
print("\n[STEP 1-2] Loading Data...")
print("-" * 70)

working_dir = r'c:\Users\bhoom\Downloads\ml1'
os.chdir(working_dir)

# Load combined CSV
df = pd.read_csv('motor_data_combined.csv')
print(f"✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData Info:")
print(f"  - Total rows: {df.shape[0]:,}")
print(f"  - Total columns: {df.shape[1]}")
print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# STEP 3: CREATE TARGET VARIABLE
# ============================================================
print("\n[STEP 3] Creating Target Variable (CLAIM)...")
print("-" * 70)

# Convert CLAIM_PAID to binary: 1 if claim was paid, 0 otherwise
df['CLAIM'] = (df['CLAIM_PAID'] > 0).astype(int)

print(f"✓ Target variable created (before cleaning)")
print(f"  - Claims (1): {(df['CLAIM'] == 1).sum():,} ({(df['CLAIM'] == 1).sum()/len(df)*100:.2f}%)")
print(f"  - No claims (0): {(df['CLAIM'] == 0).sum():,} ({(df['CLAIM'] == 0).sum()/len(df)*100:.2f}%)")

# ============================================================
# STEP 4: DATA CLEANING
# ============================================================
print("\n[STEP 4] Data Cleaning...")
print("-" * 70)

# Check for missing values
print("Missing values before cleaning:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("  ✓ No missing values found!")

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"✓ Duplicates removed: {initial_rows - len(df):,} rows removed")
print(f"  Remaining rows: {len(df):,}")

# Drop rows with missing values in critical columns
critical_cols = ['CLAIM_PAID', 'EFFECTIVE_YR', 'PROD_YEAR', 'PREMIUM', 'INSURED_VALUE', 'INSR_BEGIN', 'INSR_END', 'SEX', 'INSR_TYPE', 'MAKE', 'TYPE_VEHICLE', 'USAGE']
before_drop = len(df)
df = df.dropna(subset=critical_cols)
after_drop = len(df)
print(f"✓ Rows with missing critical values removed: {before_drop - after_drop:,}")

# Fill missing values in non-critical columns
df['SEATS_NUM'] = df['SEATS_NUM'].fillna(df['SEATS_NUM'].median())
df['CARRYING_CAPACITY'] = df['CARRYING_CAPACITY'].fillna(0)  # 0 for missing capacity
df['CCM_TON'] = df['CCM_TON'].fillna(df['CCM_TON'].median())

print(f"  Final dataset rows: {len(df):,}")

# Print CLAIM distribution after cleaning
print(f"\n✓ CLAIM distribution after cleaning:")
print(f"  - Claims (1): {(df['CLAIM'] == 1).sum():,} ({(df['CLAIM'] == 1).sum()/len(df)*100:.2f}%)")
print(f"  - No claims (0): {(df['CLAIM'] == 0).sum():,} ({(df['CLAIM'] == 0).sum()/len(df)*100:.2f}%)")

# Fix data types
print("\nData types corrected:")
print(df.dtypes)

# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================
print("\n[STEP 5] Feature Engineering...")
print("-" * 70)

# Convert EFFECTIVE_YR to numeric
df['EFFECTIVE_YR'] = pd.to_numeric(df['EFFECTIVE_YR'], errors='coerce')

# Convert date columns
df['INSR_BEGIN'] = pd.to_datetime(df['INSR_BEGIN'], errors='coerce')
df['INSR_END'] = pd.to_datetime(df['INSR_END'], errors='coerce')

# Feature 1: Policy Duration (in days)
df['POLICY_DURATION_DAYS'] = (df['INSR_END'] - df['INSR_BEGIN']).dt.days
print(f"✓ Policy Duration created")
print(f"  Mean: {df['POLICY_DURATION_DAYS'].mean():.2f} days")
print(f"  Median: {df['POLICY_DURATION_DAYS'].median():.2f} days")

# Feature 2: Vehicle Age (in years)
df['VEHICLE_AGE_YEARS'] = df['EFFECTIVE_YR'] - df['PROD_YEAR']
print(f"✓ Vehicle Age created")
print(f"  Mean: {df['VEHICLE_AGE_YEARS'].mean():.2f} years")
print(f"  Median: {df['VEHICLE_AGE_YEARS'].median():.2f} years")

# ============================================================
# STEP 6: SELECT FEATURES
# ============================================================
print("\n[STEP 6] Feature Selection...")
print("-" * 70)

# Features to keep (excluding CLAIM_PAID and OBJECT_ID)
features_to_keep = [
    'SEX',
    'INSR_TYPE',
    'INSURED_VALUE',
    'PREMIUM',
    'SEATS_NUM',
    'CARRYING_CAPACITY',
    'TYPE_VEHICLE',
    'CCM_TON',
    'MAKE',
    'USAGE',
    'VEHICLE_AGE_YEARS',
    'POLICY_DURATION_DAYS'
]

# Create feature matrix X and target y
X = df[features_to_keep].copy()
y = df['CLAIM'].copy()

print(f"✓ Features selected: {len(features_to_keep)}")
print(f"  Categorical: {['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']}")
print(f"  Numerical: {['INSURED_VALUE', 'PREMIUM', 'SEATS_NUM', 'CARRYING_CAPACITY', 'CCM_TON', 'VEHICLE_AGE_YEARS', 'POLICY_DURATION_DAYS']}")

# ============================================================
# STEP 7: CATEGORICAL ENCODING (One-Hot Encoding)
# ============================================================
print("\n[STEP 7] Categorical Encoding (One-Hot)...")
print("-" * 70)

# Identify categorical columns
categorical_cols = ['SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
numerical_cols = [col for col in features_to_keep if col not in categorical_cols]

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

print(f"✓ One-hot encoding completed")
print(f"  Features after encoding: {X_encoded.shape[1]}")
print(f"  Shape: {X_encoded.shape}")

# ============================================================
# STEP 8: SPLIT DATA (80% training, 20% testing)
# ============================================================
print("\n[STEP 8] Train-Test Split...")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Data split completed (80-20 stratified split)")
print(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Testing set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  Features: {X_train.shape[1]}")

# Train set distribution
print(f"\n  Training set - Claims distribution:")
print(f"    - No claim (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"    - Claim (1): {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")

# Test set distribution
print(f"\n  Test set - Claims distribution:")
print(f"    - No claim (0): {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"    - Claim (1): {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

# ============================================================
# STEP 9: CHOOSE MODEL (Random Forest)
# ============================================================
print("\n[STEP 9] Model Selection: Random Forest")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("✓ Random Forest model created with parameters:")
print(f"  - Number of trees: 100")
print(f"  - Max depth: 20")
print(f"  - Min samples split: 10")
print(f"  - Min samples leaf: 5")

# ============================================================
# STEP 10: TRAIN THE MODEL
# ============================================================
print("\n[STEP 10] Training the Model...")
print("-" * 70)

print("⏳ Training in progress... (this may take 1-2 minutes)")
model.fit(X_train, y_train)
print("✓ Model training completed!")

# ============================================================
# STEP 11: EVALUATE MODEL
# ============================================================
print("\n[STEP 11] Model Evaluation...")
print("-" * 70)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Get prediction probabilities
y_test_pred_proba = model.predict_proba(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n📊 PERFORMANCE METRICS")
print(f"\nTraining Set:")
print(f"  - Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  - Precision: {train_precision:.4f}")
print(f"  - Recall:    {train_recall:.4f}")

print(f"\nTest Set (Most Important):")
print(f"  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  - Precision: {test_precision:.4f} (avoid false alarms)")
print(f"  - Recall:    {test_recall:.4f} (catch claims - IMPORTANT!)")
print(f"  - F1-Score:  {test_f1:.4f}")

# Confusion Matrix
print(f"\n🔍 CONFUSION MATRIX (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n  Predicted:    No Claim(0)  |  Claim(1)")
print(f"  --------+-----------------+----------")
print(f"  Actual No Claim (0): {cm[0, 0]:>7}      |  {cm[0, 1]:>7}")
print(f"  Actual Claim (1):    {cm[1, 0]:>7}      |  {cm[1, 1]:>7}")

print(f"\nConfusion Matrix Interpretation:")
print(f"  - True Negative (TN):  {cm[0, 0]:,} - Correctly predicted 'No claim'")
print(f"  - False Positive (FP): {cm[0, 1]:,} - Predicted claim but was no claim")
print(f"  - False Negative (FN): {cm[1, 0]:,} - Missed actual claims ⚠️ (WORST ERROR)")
print(f"  - True Positive (TP):  {cm[1, 1]:,} - Correctly predicted claims")

# Classification Report
print(f"\n📋 DETAILED CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Claim', 'Claim']))

# ============================================================
# STEP 12: FEATURE IMPORTANCE
# ============================================================
print("\n[STEP 12] Feature Importance Analysis...")
print("-" * 70)

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"✓ Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:30s} {bar} {row['importance']:.4f}")

# ============================================================
# STEP 13: MAKE PREDICTIONS ON NEW DATA
# ============================================================
print("\n[STEP 13] Predictions on Test Data Examples...")
print("-" * 70)

# Show some examples with probabilities
print(f"\n📌 Sample Predictions (First 10 test samples):")
print(f"\n{'Sample':<8} {'Actual':<10} {'Predicted':<12} {'Confidence':<12} {'Risk Level':<15}")
print("-" * 60)

for i in range(min(10, len(y_test))):
    actual = "Claim" if y_test.iloc[i] == 1 else "No Claim"
    predicted = "Claim" if y_test_pred[i] == 1 else "No Claim"
    confidence = max(y_test_pred_proba[i])
    risk_level = "🔴 HIGH" if y_test_pred_proba[i][1] > 0.7 else "🟡 MEDIUM" if y_test_pred_proba[i][1] > 0.3 else "🟢 LOW"
    
    print(f"{i+1:<8} {actual:<10} {predicted:<12} {confidence:.2%}          {risk_level}")

# ============================================================
# STEP 14: SAVE MODEL AND RESULTS
# ============================================================
print("\n[STEP 14] Saving Results...")
print("-" * 70)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print(f"✓ Feature importance saved: feature_importance.csv")

# Save model performance metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [train_accuracy, train_precision, train_recall, 'N/A'],
    'Testing': [test_accuracy, test_precision, test_recall, test_f1]
})
metrics_df.to_csv('model_performance.csv', index=False)
print(f"✓ Performance metrics saved: model_performance.csv")

# Save confusion matrix
cm_df = pd.DataFrame(cm, 
    index=['Actual: No Claim', 'Actual: Claim'],
    columns=['Predicted: No Claim', 'Predicted: Claim']
)
cm_df.to_csv('confusion_matrix.csv')
print(f"✓ Confusion matrix saved: confusion_matrix.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("PIPELINE SUMMARY")
print("="*70)

print(f"""
✓ COMPLETED STEPS:
  1. ✓ Defined goal: Binary classification (Claim vs No Claim)
  2. ✓ Explored data: {df.shape[0]:,} policies with {df.shape[1]} features
  3. ✓ Created target: CLAIM variable (binary)
  4. ✓ Cleaned data: Removed {initial_rows - len(df):,} duplicates
  5. ✓ Engineered features: Policy Duration, Vehicle Age
  6. ✓ Selected features: {len(features_to_keep)} main features
  7. ✓ Encoded categories: {len(X_encoded.columns)} total features after encoding
  8. ✓ Split data: 80-20 train-test split
  9. ✓ Chose model: Random Forest (100 trees)
  10. ✓ Trained model: Model fit complete
  11. ✓ Evaluated model: Test Accuracy = {test_accuracy*100:.2f}%
  12. ✓ Feature importance: Analyzed top features
  13. ✓ Made predictions: Ready for new data
  14. ✓ Saved results: CSV files generated

📊 KEY METRICS:
  • Test Accuracy:  {test_accuracy*100:.2f}%
  • Precision:      {test_precision:.4f}
  • Recall (Important): {test_recall:.4f}
  • F1-Score:       {test_f1:.4f}

⚠️  IMPORTANT INSIGHT:
  False Negatives (Missed Claims): {cm[1, 0]:,}
  These are the worst errors in insurance - actual claims we missed!
  
  Consider adjusting the model threshold if recall needs improvement.

✓ OUTPUT FILES CREATED:
  • feature_importance.csv
  • model_performance.csv
  • confusion_matrix.csv
""")

print("="*70)
print("ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("="*70)
