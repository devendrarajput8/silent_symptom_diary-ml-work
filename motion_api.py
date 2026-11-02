
from pathlib import Path

import joblib
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/predict/motion",
    tags=["Motion Analysis"]
)

# Project models directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# Load all three trained pipelines
MODEL_PATHS = {
    "constancy": MODEL_DIR / "Constancy_of_rest_model.pkl",
    "postural": MODEL_DIR / "Postural_tremor_model.pkl",
    "rest": MODEL_DIR / "Rest_tremor_model.pkl",
}

models = {}

for name, path in MODEL_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Motion model file not found: {path}"
        )

    models[name] = joblib.load(path)


# Input schema
class MotionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 48 extracted motion features"
    )


def run_prediction(model_name: str, input_data: MotionRequest):

    model = models[model_name]

    required_features = list(model.feature_names_in_)

    received_features = input_data.features

    missing = [
        feature for feature in required_features
        if feature not in received_features
    ]

    extra = [
        feature for feature in received_features
        if feature not in required_features
    ]

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature mismatch",
                "missing_features": missing,
                "extra_features": extra
            }
        )

    # Maintain exact training feature order
    row = pd.DataFrame(
        [[received_features[f] for f in required_features]],
        columns=required_features
    )

    try:
        prediction = int(model.predict(row)[0])

        result = {
            "model": model_name,
            "prediction": prediction,
            "notice": (
                "Experimental motion model output. "
                "Not a medical diagnosis."
            )
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]

            result["class_probabilities"] = {
                str(int(label)): float(probability)
                for label, probability in zip(
                    model.classes_,
                    probabilities
                )
            }

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


@router.post("/constancy")
def predict_constancy(data: MotionRequest):
    return run_prediction("constancy", data)


@router.post("/postural")
def predict_postural(data: MotionRequest):
    return run_prediction("postural", data)


@router.post("/rest")
def predict_rest(data: MotionRequest):
    return run_prediction("rest", data)


@router.get("/health")
def motion_health():
    return {
        "status": "ok",
        "module": "Motion Analysis",
        "loaded_models": list(models.keys()),
        "feature_count": 48
    }
