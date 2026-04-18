"""
AquaSmart FastAPI backend.

Thin API layer above the shared recommendation engine so the Streamlit UI
and the HTTP backend rely on the same business rules.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.recommendation_engine import (
    CROP_DISPLAY_NAMES,
    GROWTH_STAGE_NAMES,
    ModelNotReadyError,
    RecommendationError,
    SUPPORTED_CROPS,
    CityNotFoundError,
    ExternalServiceError,
    UnknownCropError,
    UnknownStageError,
    build_farmer_recommendation,
    get_model_info as load_model_info,
    get_model_status,
    run_prediction,
    validate_crop,
)

app = FastAPI(
    title="AquaSmart API",
    description=(
        "Smart irrigation recommendations powered by ML. "
        "5 crop-specific models trained on weather and soil signals."
    ),
    version="3.1.0",
)


class FarmerRequest(BaseModel):
    city: str = Field(..., description="City or town name near the field", example="Chartres")
    surface_hectares: float = Field(..., description="Field surface area in hectares", gt=0, example=5.0)
    crop: str = Field(..., description="Crop type", example="winter_wheat")
    growth_stage: str | None = Field(None, description="Optional growth stage override")


class FarmerResponse(BaseModel):
    irrigate: bool
    amount_mm: float
    amount_liters: float
    confidence: str
    message: str
    location: str
    crop: str
    crop_display: str
    growth_stage: str
    weather_summary: str
    data_sources: dict
    recommendation_date: str


class PredictRequest(BaseModel):
    crop: str = Field(..., description="Crop type", example="winter_wheat")
    temperature_2m_mean: float = Field(..., example=18.5)
    relative_humidity_2m_mean: float = Field(..., example=55.0)
    precipitation_sum: float = Field(..., example=0.0)
    et0_fao_evapotranspiration: float = Field(..., example=4.5)
    wind_speed_10m_max: float = Field(..., example=15.0)
    soil_moisture_0_to_7cm_mean: float = Field(..., example=0.25)
    growth_stage_encoded: int = Field(..., ge=0, le=4, example=3)


class PredictResponse(BaseModel):
    irrigate: bool
    amount_mm: float
    confidence: str
    message: str
    crop: str


class CropModelInfo(BaseModel):
    crop: str
    crop_display: str
    classifier: str
    classifier_f1: float
    regressor: str
    regressor_rmse: float
    regressor_r2: float
    baseline_improvement: float


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, CityNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ModelNotReadyError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, ExternalServiceError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, (UnknownCropError, UnknownStageError, RecommendationError)):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    status = get_model_status()
    return {
        "status": "healthy" if status["loaded_crops"] else "no_models_loaded",
        **status,
    }


@app.get("/model-info", response_model=list[CropModelInfo])
def model_info():
    return [CropModelInfo(**item) for item in load_model_info()]


@app.post("/recommend", response_model=FarmerResponse)
def recommend(request: FarmerRequest):
    try:
        payload = build_farmer_recommendation(
            city=request.city,
            surface_hectares=request.surface_hectares,
            crop=request.crop,
            growth_stage=request.growth_stage,
        )
        return FarmerResponse(**payload)
    except Exception as exc:  # pragma: no cover - mapped at runtime
        _raise_http_error(exc)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        crop = validate_crop(request.crop)
        prediction = run_prediction(
            crop=crop,
            features_dict={
                "temperature_2m_mean": request.temperature_2m_mean,
                "relative_humidity_2m_mean": request.relative_humidity_2m_mean,
                "precipitation_sum": request.precipitation_sum,
                "et0_fao_evapotranspiration": request.et0_fao_evapotranspiration,
                "wind_speed_10m_max": request.wind_speed_10m_max,
                "soil_moisture_0_to_7cm_mean": request.soil_moisture_0_to_7cm_mean,
            },
            growth_stage_encoded=request.growth_stage_encoded,
        )
    except Exception as exc:  # pragma: no cover - mapped at runtime
        _raise_http_error(exc)

    stage_name = GROWTH_STAGE_NAMES.get(request.growth_stage_encoded, "unknown")
    crop_display = CROP_DISPLAY_NAMES.get(crop, crop)
    if prediction["irrigate"]:
        message = f"Irrigate {prediction['amount_mm']:.1f} mm for {crop_display} ({stage_name} stage)."
    else:
        message = f"No irrigation needed for {crop_display} ({stage_name} stage)."

    return PredictResponse(
        irrigate=prediction["irrigate"],
        amount_mm=prediction["amount_mm"],
        confidence=prediction["confidence"],
        message=message,
        crop=crop,
    )
