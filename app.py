"""FastAPI service for heart-disease binary prediction.

Run: .venv/bin/uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "working-data" / "final_model_no_source.joblib"
state: dict = {}
REQUESTS = Counter(
    "heart_api_requests_total",
    "HTTP requests handled by the heart disease API.",
    ("method", "path", "status"),
)
REQUEST_DURATION = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request processing time in seconds.",
    ("method", "path"),
)
logger = logging.getLogger("heart_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(MODEL_PATH)
    state["model"] = artifact["model"]
    state["threshold"] = float(artifact["threshold"])
    yield
    state.clear()


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    path = request.url.path
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_DURATION.labels(request.method, path).observe(elapsed)
    logger.info(
        "request method=%s path=%s status=%s duration_seconds=%.4f",
        request.method,
        path,
        response.status_code,
        elapsed,
    )
    return response


class PatientFeatures(BaseModel):
    """Optional fields may be omitted/null; the model imputes missing values."""

    age: int = Field(ge=1, le=120)
    sex: Literal["Male", "Female"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: float | None = Field(default=None, gt=0, le=300, description="Resting BP (mm Hg)")
    chol: float | None = Field(default=None, gt=0, le=700, description="Cholesterol (mg/dl)")
    fbs: Literal[0, 1] | None = Field(default=None, description="Fasting blood sugar > 120")
    restecg: Literal["normal", "lv hypertrophy", "st-t abnormality"] | None = None
    thalch: float | None = Field(default=None, gt=0, le=250, description="Max heart rate")
    exang: Literal[0, 1] | None = Field(default=None, description="Exercise-induced angina")
    oldpeak: float | None = Field(default=None, ge=-5, le=10)
    slope: Literal["upsloping", "flat", "downsloping"] | None = None
    ca: float | None = Field(default=None, ge=0, le=3, description="Major vessels (0-3)")
    thal: Literal["normal", "fixed defect", "reversable defect"] | None = None


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["no disease", "disease"]
    probability: float
    threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures) -> Prediction:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = {k: (np.nan if v is None else v) for k, v in patient.model_dump().items()}
    proba = float(state["model"].predict_proba(pd.DataFrame([row]))[0, 1])
    pred = int(proba >= state["threshold"])
    return Prediction(
        prediction=pred,
        label="disease" if pred else "no disease",
        probability=round(proba, 4),
        threshold=state["threshold"],
    )
