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
