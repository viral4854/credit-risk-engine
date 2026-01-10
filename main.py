from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn

# --- DOCUMENTATION MAPPING ---
# This dictionary maps the integer codes back to human-readable strings.
# You can copy the exact mappings from your train_model.py output here.
CODE_BOOK = {
    "job": {
        0: "High Qualification / Management",
        1: "Skilled Employee",
        2: "Unemployed / Unskilled (Non-resident)",
        3: "Unskilled (Resident)"
    },
    "checking_status": {
        0: "0 <= X < 200 DM (Medium Risk)",
        1: "< 0 DM (Overdrawn / High Risk)",
        2: ">= 200 DM (Low Risk)",
        3: "No Checking Account (Safe)"
    },
    "purpose": {
        0: "Business", 1: "Domestic Appliances", 2: "Education",
        3: "Furniture/Equipment", 4: "New Car", 5: "Used Car", 
        6: "Radio/TV", 7: "Repairs", 8: "Retraining"
        # (Note: These indices depend on alphabetical sort of your dataset)
    }
}

# --- INPUT SCHEMA ---
class LoanApplication(BaseModel):
    # NOTICE: I have deleted "example=..." completely.
    duration: float = Field(description="Loan duration in months")
    credit_amount: float = Field(description="Total credit requested in DM")
    age: float = Field(description="Applicant age in years")
    
    job: int = Field(description="0=Mgmt, 1=Skilled, 2=Unemployed, 3=Unskilled")
    checking_status: int = Field(description="0=Medium, 1=Overdrawn, 2=High Balance, 3=No Acct")
    savings_status: int = Field(description="0=Little/None, 1=Moderate, 2=Rich")
    purpose: int = Field(description="Loan purpose code (e.g., 0=Business, 5=Used Car)")

# --- APP STARTUP ---
app = FastAPI(
    title="German Credit Risk API",
    description="A Machine Learning API to predict loan default risk based on the UCI German Credit Dataset.",
    version="1.0"
)

# Load Model
try:
    model = joblib.load("credit_risk_model.pkl")
except:
    model = None

@app.get("/")
def get_mappings():
    """Returns the code book so frontend devs know what numbers to send."""
    return {
        "status": "online",
        "reference_codes": CODE_BOOK
    }

@app.post("/predict")
def predict_risk(application: LoanApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = pd.DataFrame([application.dict()])
    
    # 1. Get Raw Probability (usually 0.1 to 0.7 in this specific dataset)
    raw_probability = model.predict_proba(input_df)[0][1]

    # 2. CALIBRATION LAYER (The "Product" Fix)
    # We map the model's typical range (0.0 -> 0.7) to a user-friendly range (0% -> 100%)
    # Formula: (Value - Min) / (Max - Min)
    calibrated_score = (raw_probability - 0.0) / (0.7 - 0.0)
    
    # Clip it so it doesn't go below 0 or above 1
    calibrated_score = max(0.0, min(1.0, calibrated_score))

    # 3. Use the NEW calibrated score for decisions
    threshold = 0.5 # Now 50% feels like a natural middle ground
    
    return {
        "decision": "DENY" if calibrated_score > threshold else "APPROVE",
        
        # Send the "pretty" score to the frontend
        "risk_probability": round(calibrated_score, 4), 
        
        "risk_class": "High Risk" if calibrated_score > threshold else "Low Risk",
        "applicant_profile": {
            "job_type": CODE_BOOK["job"].get(application.job, "Unknown"),
            "age": application.age
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)