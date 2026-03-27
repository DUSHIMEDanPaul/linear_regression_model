# Loan Interest Prediction for Entrepreneurial Finance

This repository contains a machine learning-powered loan interest prediction system built as part of a mathematics for machine learning summative project. It includes:

- A FastAPI backend for inference and model retraining
- Serialized model artifacts for immediate use
- A Flutter mobile app client under `summative/flutter_app`

## Mission and Context

My mission is about Job creation and entrepreneurship. In this specific task, I focused on the finance sector, especially determining how loan interests might affect the borrowers in the success of their related business needs, precisely if the latter interests might not burden client's ventures success and on the other side if the lenders(financial institutions) are considering fairly the capacity of client's entrepreneurs in assessing the loans amount they are eligible for.

In practical terms, this project predicts likely loan interest rates based on borrower profile and credit behavior attributes, helping stakeholders reason about affordability, fairness, and business sustainability.

## Dataset

- Name: Lending Club Loan Data
- Source: Kaggle — https://www.kaggle.com/datasets/wordsforthewise/lending-club
- File: `loan.csv` (50,000 rows loaded for RAM stability)
- License: CC0 Public Domain
- Video reference: https://youtu.be/5k1IM05Cw0Y

## Repository Structure

```text
.
├── prediction.py                 # Main FastAPI inference + retraining service
├── best_loan_model.pkl          # Trained model artifact
├── scaler.pkl                   # Fitted scaler artifact
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python runtime pinning
└── summative/
    ├── API/
    │   ├── prediction.py        # API copy/variant for summative packaging
    │   └── requirements.txt
    ├── flutter_app/             # Flutter client app
    └── linear_regression/
        └── multivariate.ipynb.ipynb
```

## Core Backend Features

The backend in `prediction.py` provides:

- Health check endpoints (`/` and `/health`)
- Interest rate prediction endpoint (`/predict`)
- Background retraining endpoint (`/retrain`) from uploaded CSV data
- Validation for all model inputs using Pydantic
- CORS configuration for local and deployed client origins

## Technical Stack

- Python 3.11+
- FastAPI
- Uvicorn
- scikit-learn
- pandas
- numpy
- joblib
- Pydantic

## Local Setup (Backend)

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the API

```powershell
uvicorn prediction:app --host 0.0.0.0 --port 8000 --reload
```

### 4) Open interactive API docs

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

### `GET /`

Returns service metadata and key endpoint references.

### `GET /health`

Returns service readiness including whether model and scaler are loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### `POST /predict`

Predicts annual interest rate (%) from borrower and loan features.

- Content type: `application/json`
- Response includes predicted interest rate, unit, model version, and timestamp.
- Input schema is extensive (dozens of features). Use `/docs` for the full contract and sample payload.

### `POST /retrain`

Schedules retraining in the background using a CSV upload.

- Content type: `multipart/form-data`
- File must be a `.csv`
- Must include `int_rate` as the target column
- Existing model is replaced only if the retrained model improves MAE

## Flutter App (Client)

The Flutter client is located in `summative/flutter_app`.

Typical commands:

```powershell
cd summative/flutter_app
flutter pub get
flutter run
```

If testing on a physical device or emulator, ensure the app points to your reachable backend host/IP instead of `localhost` when required.

## Model and Artifact Notes

- `best_loan_model.pkl` and `scaler.pkl` are expected in the backend working directory.
- If missing, prediction endpoints return a 503-style readiness error.
- Retraining compares new MAE against current behavior before replacing artifacts.

## Reproducibility and Assumptions

- A subset of 50,000 rows from `loan.csv` is used for memory stability.
- Feature engineering and encoding assumptions must match training-time preprocessing for reliable predictions.
- Input ranges are constrained in the API to reduce invalid or extreme requests.

## Ethical and Practical Considerations

This project supports discussion around responsible lending and entrepreneurial opportunity. Predictions should assist decision-making, not replace human judgment. Real-world use should include:

- Fairness and bias checks across borrower groups
- Clear communication of model limitations
- Periodic monitoring and retraining using representative data

## Troubleshooting

- If `model_loaded` is false, confirm `best_loan_model.pkl` exists in the running directory.
- If `scaler_loaded` is false, confirm `scaler.pkl` exists and is compatible with the model.
- If prediction fails, verify payload fields and types against `/docs`.
- If CORS issues occur, update allowed origins in `prediction.py`.

## License and Data Usage

- Dataset license: CC0 Public Domain (per dataset source)
- Repository code license: add a dedicated `LICENSE` file if you want to formalize project code licensing

## Acknowledgment

This work is part of an applied learning effort connecting machine learning to financial inclusion, business sustainability, and entrepreneurship outcomes.
