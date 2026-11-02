
from pathlib import Path
import json
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from motion_api import router as motion_router


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "voice_acoustic_classifier.joblib"
META_PATH = BASE_DIR / "models" / "voice_model_metadata.json"


# Load Voice model and metadata
model = joblib.load(MODEL_PATH)

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_columns"]


# Create ONE FastAPI application
app = FastAPI(
    title="Silent Symptom Diary ML API",
    description="Voice and Motion ML API for research and hackathon use.",
    version="1.0.0"
)


# Register Motion API router
app.include_router(motion_router)


# ---------------- VOICE API ----------------

class VoiceFeatures(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="The 22 precomputed acoustic feature values listed in voice_model_metadata.json"
    )


@app.get("/")
def home():
    return {
        "message": "Silent Symptom Diary ML API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "feature_count": len(FEATURES),
        "raw_audio_supported": False,
        "motion_module": "registered"
    }


@app.post("/predict/voice")
def predict_voice(payload: VoiceFeatures):

    missing = [
        name for name in FEATURES
        if name not in payload.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required feature values",
                "missing_features": missing
            }
        )

    try:
        values = np.asarray(
            [payload.features[name] for name in FEATURES],
            dtype=float
        )

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be numeric."
        )

    if not np.all(np.isfinite(values)):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers."
        )

    # Preserve exact training feature order
    row = values.reshape(1, -1)

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return {
        "prediction": prediction,
        "dataset_label": (
            "parkinsons_label"
            if prediction == 1
            else "healthy_control"
        ),
        "class_probabilities": {
            str(int(label)): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        },
        "input_type": "precomputed_acoustic_features",
        "raw_audio_supported": False,
        "notice": (
            "Experimental research/hackathon output only; "
            "not a medical diagnosis."
        )
    }
