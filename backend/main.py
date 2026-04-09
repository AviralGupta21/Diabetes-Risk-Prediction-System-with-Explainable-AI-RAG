import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import run_pipeline

app = FastAPI(
    title       = "Diabetes Risk Prediction API",
    description = "RAG-augmented explainable diabetes risk assessment",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

class PredictionRequest(BaseModel):
    Pregnancies             : float = Field(..., ge=0,  le=20,  example=2)
    Glucose                 : float = Field(..., ge=0,  le=300, example=150)
    BloodPressure           : float = Field(..., ge=0,  le=200, example=80)
    SkinThickness           : float = Field(..., ge=0,  le=100, example=25)
    Insulin                 : float = Field(..., ge=0,  le=900, example=100)
    BMI                     : float = Field(..., ge=0,  le=70,  example=30.0)
    DiabetesPedigreeFunction: float = Field(..., ge=0,  le=3,   example=0.5)
    Age                     : float = Field(..., ge=1,  le=120, example=35)


class FeatureImportance(BaseModel):
    feature : str
    shap_value: float


class PredictionResponse(BaseModel):
    prediction               : int
    probability              : float
    confidence               : float
    risk_label               : str
    top_features             : list[FeatureImportance]
    all_features             : list[FeatureImportance]
    explanation_text         : str
    advice_text              : str
    explanation_sources      : list[str]
    advice_sources           : list[str]
    actionable_features      : list[str]
    non_actionable_features  : list[str]


@app.get("/")
def root():
    return {
        "message": "Diabetes Risk Prediction API is running.",
        "docs"   : "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        user_input = request.model_dump()

        result = run_pipeline(user_input)

        return PredictionResponse(
            prediction              = result["prediction"],
            probability             = result["probability"],
            confidence              = result["confidence"],
            risk_label              = "High Risk" if result["prediction"] == 1 else "Low Risk",
            top_features            = [
                FeatureImportance(feature=f, shap_value=v)
                for f, v in result["top_features"]
            ],
            all_features            = [
                FeatureImportance(feature=f, shap_value=v)
                for f, v in result["all_features"]
            ],
            explanation_text        = result["explanation_text"],
            advice_text             = result["advice_text"],
            explanation_sources     = result["explanation_sources"],
            advice_sources          = result["advice_sources"],
            actionable_features     = result["actionable_features"],
            non_actionable_features = result["non_actionable_features"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))