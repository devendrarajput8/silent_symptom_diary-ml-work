import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path

from audio_feature_extractor import extract_audio_features


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "raw_voice_classifier.joblib"
METADATA_PATH = MODEL_DIR / "raw_voice_metadata.json"


# Load model and metadata
import json

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

voice_model = joblib.load(MODEL_PATH)

FEATURES = metadata["feature_columns"]


# Create FastAPI application
app = FastAPI(
    title="Silent Symptom Diary - Voice ML API",
    description="Experimental raw audio voice analysis API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Silent Symptom Diary Voice API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "feature_count": len(FEATURES),
        "raw_audio_supported": True
    }


@app.post("/predict/voice-audio")
async def predict_voice_audio(audio: UploadFile = File(...)):

    # Check file extension
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a WAV audio file."
        )

    temp_path = None

    try:
        # Read uploaded audio
        contents = await audio.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty."
            )

        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file is too large. Maximum size is 15 MB."
            )

        # Save temporary WAV file
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # Validate WAV metadata
        info = sf.info(temp_path)

        if info.duration < 1.5 or info.duration > 10.5:
            raise HTTPException(
                status_code=400,
                detail="Recording duration must be between 1.5 and 10 seconds."
            )

        if info.channels < 1:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio channel configuration."
            )

        # Extract acoustic features
        extracted = extract_audio_features(temp_path)

        # Check required features
        missing = [
            col for col in FEATURES
            if col not in extracted
        ]

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Feature extraction mismatch: {missing}"
            )

        # Prepare model input in exact training order
        input_df = pd.DataFrame(
            [[extracted[col] for col in FEATURES]],
            columns=FEATURES
        )

        input_df = input_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Predict
        prediction = int(voice_model.predict(input_df)[0])

        probabilities = voice_model.predict_proba(input_df)[0]

        probability_result = {
            str(int(label)): float(probability)
            for label, probability in zip(
                voice_model.classes_,
                probabilities
            )
        }

        return {
            "status": "success",
            "prediction": prediction,
            "label": (
                "Parkinson's pattern"
                if prediction == 1
                else "Healthy control pattern"
            ),
            "class_probabilities": probability_result,
            "recording_duration_seconds": round(info.duration, 2),
            "notice": "Experimental research output, not a medical diagnosis."
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Voice prediction failed: {str(error)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)