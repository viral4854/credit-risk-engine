import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
import joblib

print("⏳ Downloading Real 'German Credit' Dataset...")
dataset = fetch_openml(name='credit-g', version=1, parser='auto')
df = dataset.data
target = dataset.target

# 1. Select MORE Features
# 'checking_status' is arguably the #1 predictor of default
features = [
    'duration', 'credit_amount', 'age', 'job', 
    'checking_status', 'savings_status', 'purpose'
]
X = df[features].copy()
y = target.map({'bad': 1, 'good': 0})

print(f"✅ Data Shape: {X.shape}")

# 2. Advanced Encoding (Explicit List)
# We list exactly which columns we want to turn into numbers
cols_to_encode = ['checking_status', 'savings_status', 'purpose', 'job']
encoders = {}

print("🔧 Encoding categorical columns...")
for col in cols_to_encode:
    # Force the column to be a string first (fixes data type issues)
    X[col] = X[col].astype(str)
    
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    
    # Save the mapping using the "Pylance Safe" method we learned
    encoders[col] = {label: i for i, label in enumerate(le.classes_)}

# Now this key is GUARANTEED to exist
print(f"ℹ️  Checking Status Mapping: {encoders['checking_status']}")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train with 'class_weight="balanced"'
# This tells the model: "Pay 2x more attention to Bad Credit cases"
print("Training Optimized Model...")
model = RandomForestClassifier(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees to catch nuances
    class_weight='balanced', # Crucial for fraud/risk detection
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print(f"🚀 New Model Accuracy: {model.score(X_test, y_test):.2f}")

# Save
joblib.dump(model, "credit_risk_model.pkl")
print("✅ Optimized Model saved.")