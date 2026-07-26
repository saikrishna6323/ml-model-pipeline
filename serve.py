"""Small FastAPI service that loads the trained classifier and exposes
/predict, so it can plug into predictive-cicd-remediation's prediction
module as an alternative to the LLM-backed predictor."""
from __future__ import annotations
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from train import FEATURE_COLS

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

app = FastAPI(title="CI/CD Failure Classifier Service")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    rolling_failure_rate: float
    dependency_files_changed: int
    step_duration_seconds: float
    num_recent_errors: int
    day_of_week: int
    hour_of_day: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    row = pd.DataFrame([req.dict()])[FEATURE_COLS]
    probability = float(model.predict_proba(row)[0, 1])
    return {"failure_probability": probability}
