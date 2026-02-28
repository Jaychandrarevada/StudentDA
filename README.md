# StudentDA - Full-Stack ML Student Risk Monitoring System

This project is now a **deployable full-stack application** where ML is the core:
- Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Stores model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC).
- Serves a web UI + API for model training, metrics viewing, and live prediction.

## Architecture
- **Backend**: FastAPI (`app/main.py`)
- **ML pipeline**: scikit-learn + XGBoost (`src/train_models.py`)
- **Frontend**: HTML/CSS/JS (`app/templates`, `app/static`)
- **Data generator**: synthetic realistic student dataset if you do not have real data (`src/generate_demo_data.py`)

## Why this solves your problem
You said you do not have a dataset but need a working full-stack app with ML-integrated outputs and metrics.
This repo now supports exactly that flow:
1. Generate demo dataset.
2. Train 3 models.
3. Compare models and show evaluation metrics.
4. Run prediction from UI/API.
5. Deploy with Docker or any cloud that supports FastAPI.

## Quick Start (Local)
# StudentDA - Learning Analytics Based Student Risk Prediction

This project provides a practical machine learning pipeline to **identify students at risk of academic underperformance early** using institutional and LMS activity data.

## Models Included
The training pipeline includes the three models you asked for:
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (robust non-linear ensemble)
- **XGBoost** (high-performance gradient boosting)

The system trains all three models, compares them using stratified cross-validation, and automatically saves the best-performing model by ROC-AUC.

## Suggested Input Features
Use a CSV dataset where each row is one student record and one target column indicates risk.

Typical features:
- Attendance percentage
- Internal assessment scores
- Assignment submission and marks
- Quiz/lab performance
- LMS login frequency
- Time spent on LMS
- Content access count
- Forum interactions
- Historical GPA / previous semester result

Target column example:
- `at_risk` (0 = not at risk, 1 = at risk)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_demo_data.py --rows 1200 --output data/student_data.csv
uvicorn app.main:app --reload
```
Open: `http://127.0.0.1:8000`

## API Endpoints
- `POST /api/train` → trains LR/RF/XGBoost, saves best model and metrics
- `GET /api/metrics` → returns leaderboard + test metrics
- `POST /api/predict` → returns risk probability and band for a student feature payload

## Training from CLI
```bash
python src/train_models.py \
  --data data/student_data.csv \
  --target at_risk \
  --id-columns student_id
```

## Artifacts Produced
- `artifacts/leaderboard.csv`
- `artifacts/test_metrics.json`
- `artifacts/best_model.joblib`

## Deployment (Docker)
```bash
docker build -t studentda .
docker run -p 8000:8000 studentda
```

Then visit `http://localhost:8000`.

## Notes for Academic Demo / Viva
- Show that all 3 models are trained and compared (leaderboard).
- Show test metrics (accuracy + ROC-AUC + F1 etc.) from `/api/metrics`.
- Show live inference in UI with a sample student profile.
- Explain that real institutional data can replace synthetic data with no code changes to API contract.


## Windows Run Commands (Recommended)
If `uvicorn` is not recognized in CMD/PowerShell, use module mode or the launcher script:

```bash
python -m pip install -r requirements.txt
python src/generate_demo_data.py --rows 1200 --output data/student_data.csv
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Alternative (no PATH dependency):

```bash
python run.py
```

## Troubleshooting
### 1) `'uvicorn' is not recognized`
This means the `uvicorn` executable is not on PATH. Use:
- `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`, or
- `python run.py`

### 2) `SyntaxError` in `src/train_models.py`
If you see a mismatch error like `closing parenthesis ')' does not match opening parenthesis '{'`, your local file is likely corrupted or partially edited.

Validate quickly:
```bash
python -m py_compile src/train_models.py
```

If it fails:
- restore `src/train_models.py` from the latest repo commit,
- avoid manual edits/copy-paste from formatted documents,
- rerun `python -m py_compile src/train_models.py` and then start API again.
## Outputs
After training, artifacts are saved in `artifacts/`:
- `leaderboard.csv` - CV comparison of Logistic Regression, Random Forest, and XGBoost
- `test_metrics.json` - holdout test metrics for the selected best model
- `best_model.joblib` - full preprocessing + model pipeline for inference

## Accuracy Improvement Tips
To improve prediction accuracy beyond a basic setup:
1. Balance classes (SMOTE / class weights / threshold tuning)
2. Perform feature engineering (engagement trends, lag-based features)
3. Tune hyperparameters (GridSearchCV / Optuna)
4. Use time-aware validation if semester sequence matters
5. Track precision-recall trade-offs to reduce false negatives (missed at-risk students)

---
If you want, I can next add:
- a Streamlit dashboard for faculty/admin roles,
- feature importance explainability (SHAP),
- and an API endpoint for real-time risk scoring.
