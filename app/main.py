from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.train_models import FEATURE_COLUMNS, train_and_save

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "test_metrics.json"
LEADERBOARD_PATH = ARTIFACT_DIR / "leaderboard.csv"

app = FastAPI(title="Student Risk Monitoring System")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class StudentFeatures(BaseModel):
    attendance_pct: float = Field(ge=0, le=100)
    internal_score: float = Field(ge=0, le=100)
    assignment_avg: float = Field(ge=0, le=100)
    quiz_avg: float = Field(ge=0, le=100)
    lms_logins_per_week: float = Field(ge=0)
    lms_hours_per_week: float = Field(ge=0)
    content_views: float = Field(ge=0)
    forum_posts: float = Field(ge=0)
    previous_gpa: float = Field(ge=0, le=10)
    department: str
    hostel_resident: str


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path("app/templates/index.html").read_text()


@app.post("/api/train")
def run_training() -> dict:
    data_path = Path("data/student_data.csv")
    if not data_path.exists():
        raise HTTPException(status_code=400, detail="Dataset missing. Generate data first.")

    result = train_and_save(
        data_path=str(data_path),
        target="at_risk",
        id_columns=["student_id"],
        output_dir=str(ARTIFACT_DIR),
    )
    return result


@app.get("/api/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists() or not LEADERBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Run training first.")

    return {
        "test_metrics": json.loads(METRICS_PATH.read_text()),
        "leaderboard": pd.read_csv(LEADERBOARD_PATH).to_dict(orient="records"),
    }


@app.post("/api/predict")
def predict(features: StudentFeatures) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No model found. Run training first.")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{col: getattr(features, col) for col in FEATURE_COLUMNS}])
    probability = float(model.predict_proba(row)[0, 1])
    label = int(probability >= 0.5)

    return {
        "risk_probability": round(probability, 4),
        "predicted_label": label,
        "risk_band": "HIGH" if probability >= 0.65 else "MEDIUM" if probability >= 0.4 else "LOW",
    }
