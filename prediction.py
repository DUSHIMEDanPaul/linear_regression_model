import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Interest Rate Predictor API",
    description=(
        "Predicts loan interest rates using a trained Random Forest model. "
        "Built for the ALU Mathematics for Machine Learning summative assignment."
    ),
    version="1.0.0",
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
# Explicitly list allowed origins instead of wildcard for better security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://linear-regression-model-2-42t2.onrender.com",  # replace with your Render URL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH  = Path("best_loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")

def load_artifacts():
    """Load model and scaler, raise clear errors if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError as e:
    print(f"WARNING: {e}. Prediction endpoints will fail until files are present.")
    model, scaler = None, None

# ── Pydantic input model ───────────────────────────────────────────────────────
class LoanInput(BaseModel):
    loan_amnt:                    float = Field(..., ge=500,    le=40000,  description="Loan amount in USD")
    term:                         int   = Field(..., ge=36,     le=60,     description="Loan term in months (36 or 60)")
    installment:                  float = Field(..., ge=10,     le=2000,   description="Monthly installment amount")
    sub_grade:                    int   = Field(..., ge=1,      le=35,     description="Loan sub-grade (1-35)")
    emp_length:                   int   = Field(..., ge=0,      le=10,     description="Employment length in years")
    home_ownership:               int   = Field(..., ge=0,      le=5,      description="Home ownership status (encoded)")
    annual_inc:                   float = Field(..., ge=1000,   le=10000000, description="Annual income in USD")
    verification_status:          int   = Field(..., ge=0,      le=2,      description="Income verification status (0-2)")
    purpose:                      int   = Field(..., ge=0,      le=13,     description="Loan purpose (encoded)")
    dti:                          float = Field(..., ge=0.0,    le=100.0,  description="Debt-to-income ratio")
    delinq_2yrs:                  int   = Field(..., ge=0,      le=30,     description="Delinquencies in last 2 years")
    inq_last_6mths:               int   = Field(..., ge=0,      le=33,     description="Credit inquiries in last 6 months")
    open_acc:                     int   = Field(..., ge=0,      le=90,     description="Number of open credit lines")
    pub_rec:                      int   = Field(..., ge=0,      le=86,     description="Number of derogatory public records")
    revol_bal:                    float = Field(..., ge=0,      le=2904836,description="Total revolving balance")
    revol_util:                   float = Field(..., ge=0.0,    le=892.3,  description="Revolving line utilization rate (%)")
    total_acc:                    int   = Field(..., ge=1,      le=162,    description="Total number of credit lines")
    collections_12_mths_ex_med:   int   = Field(..., ge=0,      le=20,     description="Collections in last 12 months")
    application_type:             int   = Field(..., ge=0,      le=1,      description="Application type (0=Individual, 1=Joint)")
    tot_coll_amt:                 float = Field(..., ge=0,      le=9152545,description="Total collection amounts ever owed")
    tot_cur_bal:                  float = Field(..., ge=0,      le=8000078,description="Total current balance of all accounts")
    open_acc_6m:                  int   = Field(..., ge=0,      le=16,     description="Open trades in last 6 months")
    open_act_il:                  int   = Field(..., ge=0,      le=46,     description="Currently active installment trades")
    open_il_12m:                  int   = Field(..., ge=0,      le=23,     description="Installment accounts opened in past 12 months")
    open_il_24m:                  int   = Field(..., ge=0,      le=34,     description="Installment accounts opened in past 24 months")
    mths_since_rcnt_il:           int   = Field(..., ge=0,      le=400,    description="Months since most recent installment account opened")
    total_bal_il:                 float = Field(..., ge=0,      le=1206666,description="Total current balance of all installment accounts")
    il_util:                      float = Field(..., ge=0.0,    le=600.0,  description="Ratio of total current balance to high credit/credit limit on all installment accounts")
    open_rv_12m:                  int   = Field(..., ge=0,      le=24,     description="Revolving trades opened in past 12 months")
    open_rv_24m:                  int   = Field(..., ge=0,      le=35,     description="Revolving trades opened in past 24 months")
    max_bal_bc:                   float = Field(..., ge=0,      le=603067, description="Maximum current balance owed on all revolving accounts")
    all_util:                     float = Field(..., ge=0.0,    le=500.0,  description="Balance to credit limit on all trades")
    total_rev_hi_lim:             float = Field(..., ge=0,      le=9999999,description="Total revolving high credit/credit limit")
    inq_fi:                       int   = Field(..., ge=0,      le=25,     description="Number of personal finance inquiries")
    total_cu_tl:                  int   = Field(..., ge=0,      le=116,    description="Number of finance trades")
    inq_last_12m:                 int   = Field(..., ge=0,      le=66,     description="Number of credit inquiries in past 12 months")
    acc_open_past_24mths:         int   = Field(..., ge=0,      le=64,     description="Number of trades opened in past 24 months")
    avg_cur_bal:                  float = Field(..., ge=0,      le=958084, description="Average current balance of all accounts")
    bc_open_to_buy:               float = Field(..., ge=0,      le=999999, description="Total open to buy on revolving bankcards")
    bc_util:                      float = Field(..., ge=0.0,    le=1000.0, description="Ratio of total current balance to high credit/credit limit for all bankcard accounts")
    mo_sin_old_il_acct:           int   = Field(..., ge=0,      le=800,    description="Months since oldest bank installment account opened")
    mo_sin_old_rev_tl_op:         int   = Field(..., ge=0,      le=800,    description="Months since oldest revolving account opened")
    mo_sin_rcnt_rev_tl_op:        int   = Field(..., ge=0,      le=400,    description="Months since most recent revolving account opened")
    mo_sin_rcnt_tl:               int   = Field(..., ge=0,      le=400,    description="Months since most recent account opened")
    mort_acc:                     int   = Field(..., ge=0,      le=50,     description="Number of mortgage accounts")
    mths_since_recent_bc:         int   = Field(..., ge=0,      le=800,    description="Months since most recent bankcard account opened")
    mths_since_recent_inq:        int   = Field(..., ge=0,      le=25,     description="Months since most recent inquiry")
    num_accts_ever_120_pd:        int   = Field(..., ge=0,      le=50,     description="Number of accounts ever 120 or more days past due")
    num_actv_bc_tl:               int   = Field(..., ge=0,      le=30,     description="Number of currently active bankcard accounts")
    num_actv_rev_tl:              int   = Field(..., ge=0,      le=40,     description="Number of currently active revolving trades")
    num_bc_sats:                  int   = Field(..., ge=0,      le=50,     description="Number of satisfactory bankcard accounts")
    num_bc_tl:                    int   = Field(..., ge=0,      le=70,     description="Number of bankcard accounts")
    num_il_tl:                    int   = Field(..., ge=0,      le=135,    description="Number of installment accounts")
    num_op_rev_tl:                int   = Field(..., ge=0,      le=55,     description="Number of open revolving accounts")
    num_rev_accts:                int   = Field(..., ge=0,      le=90,     description="Number of revolving accounts")
    num_rev_tl_bal_gt_0:          int   = Field(..., ge=0,      le=55,     description="Number of revolving trades with balance > 0")
    num_sats:                     int   = Field(..., ge=0,      le=90,     description="Number of satisfactory accounts")
    num_tl_90g_dpd_24m:           int   = Field(..., ge=0,      le=40,     description="Number of accounts 90 or more days past due in last 24 months")
    num_tl_op_past_12m:           int   = Field(..., ge=0,      le=30,     description="Number of accounts opened in past 12 months")
    pct_tl_nvr_dlq:               float = Field(..., ge=0.0,    le=100.0,  description="Percent of trades never delinquent")
    percent_bc_gt_75:             float = Field(..., ge=0.0,    le=100.0,  description="Percent of all bankcard accounts > 75% of limit")
    pub_rec_bankruptcies:         int   = Field(..., ge=0,      le=12,     description="Number of public record bankruptcies")
    tot_hi_cred_lim:              float = Field(..., ge=0,      le=9999999,description="Total high credit/credit limit")
    total_bal_ex_mort:            float = Field(..., ge=0,      le=3083944,description="Total credit balance excluding mortgage")
    total_bc_limit:               float = Field(..., ge=0,      le=1105200,description="Total bankcard high credit/credit limit")
    total_il_high_credit_limit:   float = Field(..., ge=0,      le=2116000,description="Total installment high credit/credit limit")

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000, "term": 36, "installment": 450.5,
                "sub_grade": 10, "emp_length": 5, "home_ownership": 2,
                "annual_inc": 65000, "verification_status": 1, "purpose": 3,
                "dti": 18.5, "delinq_2yrs": 0, "inq_last_6mths": 1,
                "open_acc": 10, "pub_rec": 0, "revol_bal": 8000,
                "revol_util": 45.0, "total_acc": 22,
                "collections_12_mths_ex_med": 0, "application_type": 0,
                "tot_coll_amt": 0, "tot_cur_bal": 50000,
                "open_acc_6m": 1, "open_act_il": 3, "open_il_12m": 1,
                "open_il_24m": 2, "mths_since_rcnt_il": 6, "total_bal_il": 12000,
                "il_util": 55.0, "open_rv_12m": 2, "open_rv_24m": 3,
                "max_bal_bc": 3000, "all_util": 48.0, "total_rev_hi_lim": 20000,
                "inq_fi": 1, "total_cu_tl": 2, "inq_last_12m": 3,
                "acc_open_past_24mths": 4, "avg_cur_bal": 5000,
                "bc_open_to_buy": 4000, "bc_util": 40.0,
                "mo_sin_old_il_acct": 60, "mo_sin_old_rev_tl_op": 120,
                "mo_sin_rcnt_rev_tl_op": 3, "mo_sin_rcnt_tl": 3,
                "mort_acc": 1, "mths_since_recent_bc": 5,
                "mths_since_recent_inq": 4, "num_accts_ever_120_pd": 0,
                "num_actv_bc_tl": 3, "num_actv_rev_tl": 4, "num_bc_sats": 4,
                "num_bc_tl": 6, "num_il_tl": 8, "num_op_rev_tl": 7,
                "num_rev_accts": 10, "num_rev_tl_bal_gt_0": 4, "num_sats": 10,
                "num_tl_90g_dpd_24m": 0, "num_tl_op_past_12m": 3,
                "pct_tl_nvr_dlq": 92.0, "percent_bc_gt_75": 25.0,
                "pub_rec_bankruptcies": 0, "tot_hi_cred_lim": 75000,
                "total_bal_ex_mort": 20000, "total_bc_limit": 10000,
                "total_il_high_credit_limit": 25000
            }
        }


# ── Response models ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_interest_rate: float
    unit: str = "%"
    model_version: str = "1.0.0"
    timestamp: str


class RetrainResponse(BaseModel):
    status: str
    message: str
    rows_received: int
    timestamp: str


# ── Helper: build DataFrame in correct feature order ───────────────────────────
FEATURE_ORDER = [
    "loan_amnt", "term", "installment", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med", "application_type",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il", "open_il_12m",
    "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util",
    "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "total_rev_hi_lim",
    "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "pub_rec_bankruptcies", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]


def input_to_dataframe(data: LoanInput) -> pd.DataFrame:
    row = {col: getattr(data, col) for col in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ── Retraining helper (runs in background) ─────────────────────────────────────
def retrain_from_dataframe(df: pd.DataFrame):
    """
    Minimal retraining: fits a new Random Forest on the uploaded data,
    updates best_loan_model.pkl if the new model has lower MAE.
    """
    global model, scaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    if "int_rate" not in df.columns:
        print("Retrain skipped: target column 'int_rate' not found in uploaded data.")
        return

    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["int_rate"]

    if len(X) < 20:
        print("Retrain skipped: not enough rows (need ≥ 20).")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_scaler = StandardScaler()
    X_train_sc = new_scaler.fit_transform(X_train)
    X_test_sc  = new_scaler.transform(X_test)

    new_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    new_model.fit(X_train_sc, y_train)

    new_mae = mean_absolute_error(y_test, new_model.predict(X_test_sc))

    # Compare against current model if possible
    try:
        old_preds = model.predict(new_scaler.transform(X_test))
        old_mae   = mean_absolute_error(y_test, old_preds)
    except Exception:
        old_mae = float("inf")

    if new_mae < old_mae:
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        model, scaler = new_model, new_scaler
        print(f"Model updated. New MAE: {new_mae:.4f} vs Old MAE: {old_mae:.4f}")
    else:
        print(f"Retrain complete but model NOT updated. New MAE: {new_mae:.4f} >= Old MAE: {old_mae:.4f}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Loan Interest Rate Predictor API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: LoanInput):
    """
    **Predict the interest rate** for a loan application.

    - All fields are required and validated for correct data types and realistic ranges.
    - Returns the predicted annual interest rate as a percentage.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded. Ensure best_loan_model.pkl and scaler.pkl are present.",
        )

    try:
        df = input_to_dataframe(data)
        scaled = scaler.transform(df)
        prediction = float(model.predict(scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        predicted_interest_rate=round(prediction, 4),
        unit="%",
        model_version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["Model Update"])
async def retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with new labelled data (must include 'int_rate' column)"),
):
    """
    **Trigger model retraining** with new labelled data.

    - Upload a CSV file containing the feature columns plus an `int_rate` target column.
    - Retraining runs in the **background** so the endpoint returns immediately.
    - The saved model (`best_loan_model.pkl`) is only replaced if the new model achieves a lower MAE.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    background_tasks.add_task(retrain_from_dataframe, df)

    return RetrainResponse(
        status="accepted",
        message="Retraining has been scheduled in the background. The model will be updated if the new one performs better.",
        rows_received=len(df),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
