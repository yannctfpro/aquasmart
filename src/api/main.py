"""
AquaSmart — FastAPI Backend (v3 Multi-Crop)
=============================================
Serves crop-specific irrigation predictions from trained ML models.
Each crop has its own classifier, regressor, and scaler.

Endpoints:
    POST /recommend     → Farmer-facing: city + crop + surface → liters
    POST /predict       → Internal: raw features + crop → mm
    GET  /health        → Health check + loaded crops
    GET  /model-info    → Performance metrics for all crops

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import date
import joblib
import numpy as np
import json
import requests as http_requests

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="AquaSmart API",
    description="Smart irrigation recommendations powered by ML. "
                "5 crop-specific models trained on 10 cities across France.",
    version="3.0.0",
)

# ============================================================
# LOAD ALL CROP MODELS (once at startup)
# ============================================================

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

SUPPORTED_CROPS = ["winter_wheat", "corn", "barley", "rapeseed", "sunflower"]

# Store all models in a dict: models["corn"]["classifier"], etc.
models = {}

for crop in SUPPORTED_CROPS:
    crop_dir = MODEL_DIR / crop
    try:
        models[crop] = {
            "classifier": joblib.load(crop_dir / "classifier.joblib"),
            "regressor": joblib.load(crop_dir / "regressor.joblib"),
            "scaler": joblib.load(crop_dir / "scaler.joblib"),
            "results": json.load(open(crop_dir / "results.json")),
        }
        print(f"  ✅ {crop} models loaded")
    except Exception as e:
        print(f"  ⚠️  {crop} failed to load: {e}")
        models[crop] = None

loaded_crops = [c for c in SUPPORTED_CROPS if models.get(c) is not None]
print(f"\n✅ {len(loaded_crops)}/{len(SUPPORTED_CROPS)} crop models ready: {', '.join(loaded_crops)}")


# ============================================================
# GROWTH STAGE CALENDARS & Kc (FAO-56)
# ============================================================

CROP_STAGE_CALENDAR = {
    "winter_wheat": {
        10: "initial", 11: "initial", 12: "initial",
        1: "development", 2: "development", 3: "development",
        4: "mid_season", 5: "mid_season",
        6: "late_season", 7: "late_season",
        8: "fallow", 9: "fallow",
    },
    "corn": {
        4: "initial", 5: "development",
        6: "mid_season", 7: "mid_season", 8: "mid_season",
        9: "late_season",
        10: "fallow", 11: "fallow", 12: "fallow",
        1: "fallow", 2: "fallow", 3: "fallow",
    },
    "barley": {
        10: "initial", 11: "initial", 12: "initial",
        1: "development", 2: "development", 3: "development",
        4: "mid_season", 5: "mid_season",
        6: "late_season",
        7: "fallow", 8: "fallow", 9: "fallow",
    },
    "rapeseed": {
        9: "initial", 10: "initial", 11: "initial",
        12: "development", 1: "development", 2: "development",
        3: "mid_season", 4: "mid_season", 5: "mid_season",
        6: "late_season", 7: "late_season",
        8: "fallow",
    },
    "sunflower": {
        4: "initial", 5: "development",
        6: "mid_season", 7: "mid_season", 8: "mid_season",
        9: "late_season",
        10: "fallow", 11: "fallow", 12: "fallow",
        1: "fallow", 2: "fallow", 3: "fallow",
    },
}

GROWTH_STAGE_ENCODING = {
    "fallow": 0, "initial": 1, "development": 2, "mid_season": 3, "late_season": 4,
}
GROWTH_STAGE_NAMES = {v: k for k, v in GROWTH_STAGE_ENCODING.items()}

CROP_DISPLAY_NAMES = {
    "winter_wheat": "Winter Wheat (Blé tendre)",
    "corn": "Corn (Maïs)",
    "barley": "Barley (Orge)",
    "rapeseed": "Rapeseed (Colza)",
    "sunflower": "Sunflower (Tournesol)",
}


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class FarmerRequest(BaseModel):
    """What the farmer provides."""
    city: str = Field(..., description="City or town name near the field", example="Chartres")
    surface_hectares: float = Field(..., description="Field surface area in hectares", gt=0, example=5.0)
    crop: str = Field(..., description="Crop type: winter_wheat, corn, barley, rapeseed, sunflower", example="winter_wheat")
    growth_stage: str | None = Field(None, description="Optional: override auto-detected growth stage")


class FarmerResponse(BaseModel):
    """What the farmer receives."""
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


class PredictRequest(BaseModel):
    """Internal: raw ML prediction input."""
    crop: str = Field(..., description="Crop type", example="winter_wheat")
    temperature_2m_mean: float = Field(..., example=18.5)
    relative_humidity_2m_mean: float = Field(..., example=55.0)
    precipitation_sum: float = Field(..., example=0.0)
    et0_fao_evapotranspiration: float = Field(..., example=4.5)
    wind_speed_10m_max: float = Field(..., example=15.0)
    soil_moisture_0_to_7cm_mean: float = Field(..., example=0.25)
    growth_stage_encoded: int = Field(..., ge=0, le=4, example=3)


class PredictResponse(BaseModel):
    """Internal prediction output."""
    irrigate: bool
    amount_mm: float
    confidence: str
    message: str
    crop: str


class CropModelInfo(BaseModel):
    """Performance metrics for one crop."""
    crop: str
    crop_display: str
    classifier: str
    classifier_f1: float
    regressor: str
    regressor_rmse: float
    regressor_r2: float
    baseline_improvement: str


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_confidence(probabilities: np.ndarray) -> str:
    max_prob = probabilities.max()
    if max_prob >= 0.85:
        return "high"
    elif max_prob >= 0.65:
        return "medium"
    return "low"


def get_growth_stage(crop: str, target_date: date) -> str:
    calendar = CROP_STAGE_CALENDAR.get(crop)
    if calendar is None:
        raise ValueError(f"Unknown crop: {crop}")
    return calendar.get(target_date.month, "fallow")


def geocode_city(city: str) -> dict:
    """Convert city name to coordinates via Open-Meteo Geocoding API."""
    response = http_requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results_list = data.get("results", [])
    if not results_list:
        raise ValueError(f"City '{city}' not found. Please check the spelling.")
    match = results_list[0]
    return {
        "name": match.get("name", city),
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "country": match.get("country", ""),
    }


def fetch_today_weather(lat: float, lon: float) -> dict:
    """Fetch today's weather from Open-Meteo Forecast API."""
    today = date.today().isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,et0_fao_evapotranspiration,wind_speed_10m_max",
        "hourly": "soil_moisture_0_to_1cm",
        "start_date": today, "end_date": today,
        "timezone": "auto",
    }
    response = http_requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})
    soil_values = hourly.get("soil_moisture_0_to_1cm", [])
    soil_moisture = np.mean([v for v in soil_values if v is not None]) if soil_values else 0.30

    weather = {
        "temperature_2m_mean": daily.get("temperature_2m_mean", [None])[0],
        "relative_humidity_2m_mean": daily.get("relative_humidity_2m_mean", [None])[0],
        "precipitation_sum": daily.get("precipitation_sum", [None])[0],
        "et0_fao_evapotranspiration": daily.get("et0_fao_evapotranspiration", [None])[0],
        "wind_speed_10m_max": daily.get("wind_speed_10m_max", [None])[0],
        "soil_moisture_0_to_7cm_mean": round(float(soil_moisture), 4),
    }
    for key, val in weather.items():
        if val is None:
            raise ValueError(f"Open-Meteo returned no data for '{key}'.")
    return weather


def mm_to_liters(mm: float, hectares: float) -> float:
    """1 mm on 1 m² = 1 liter. 1 ha = 10,000 m²."""
    return round(mm * hectares * 10_000, 0)


def run_prediction(crop: str, features_dict: dict, growth_stage_encoded: int) -> dict:
    """Run the two-stage ML pipeline for a specific crop."""
    crop_models = models.get(crop)
    if crop_models is None:
        raise RuntimeError(f"Models for '{crop}' are not loaded.")

    clf = crop_models["classifier"]
    reg = crop_models["regressor"]
    scl = crop_models["scaler"]

    features = np.array([[
        features_dict["temperature_2m_mean"],
        features_dict["relative_humidity_2m_mean"],
        features_dict["precipitation_sum"],
        features_dict["et0_fao_evapotranspiration"],
        features_dict["wind_speed_10m_max"],
        features_dict["soil_moisture_0_to_7cm_mean"],
        growth_stage_encoded,
    ]])

    features_scaled = scl.transform(features)

    irrigate_pred = clf.predict(features_scaled)[0]
    probabilities = clf.predict_proba(features_scaled)[0]
    confidence = get_confidence(probabilities)

    amount_mm = 0.0
    if irrigate_pred == 1:
        amount_mm = float(reg.predict(features_scaled)[0])
        amount_mm = max(0.0, round(amount_mm, 2))

    return {
        "irrigate": bool(irrigate_pred == 1),
        "amount_mm": amount_mm,
        "confidence": confidence,
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """Check API status and which crop models are loaded."""
    return {
        "status": "healthy" if loaded_crops else "no_models_loaded",
        "crops_loaded": loaded_crops,
        "crops_total": len(SUPPORTED_CROPS),
    }


@app.get("/model-info", response_model=list[CropModelInfo])
def model_info():
    """Return performance metrics for all loaded crop models."""
    infos = []
    for crop in loaded_crops:
        r = models[crop]["results"]
        best_cls = r["best_classifier"]
        best_reg = r["best_regressor"]
        cls_metrics = r["classification"][best_cls]

        if best_reg != "None" and best_reg in r["regression"]:
            reg_metrics = r["regression"][best_reg]
            baseline_mae = r["baseline"]["reg_mae"]
            ai_mae = reg_metrics["mae"]
            improvement = (1 - ai_mae / baseline_mae) * 100 if baseline_mae > 0 else 0
            infos.append(CropModelInfo(
                crop=crop,
                crop_display=CROP_DISPLAY_NAMES.get(crop, crop),
                classifier=best_cls,
                classifier_f1=round(cls_metrics["f1"], 4),
                regressor=best_reg,
                regressor_rmse=round(reg_metrics["rmse"], 4),
                regressor_r2=round(reg_metrics["r2"], 4),
                baseline_improvement=f"{improvement:.1f}%",
            ))
    return infos


@app.post("/recommend", response_model=FarmerResponse)
def recommend(request: FarmerRequest):
    """
    🌱 Smart irrigation recommendation for farmers.

    The farmer provides: city, field surface, crop type.
    The system: geocodes → fetches weather → detects growth stage → predicts → converts to liters.
    """
    # Validate crop
    if request.crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{request.crop}'. Available: {SUPPORTED_CROPS}")
    if models.get(request.crop) is None:
        raise HTTPException(status_code=503, detail=f"Model for '{request.crop}' is not loaded.")

    # Geocode city
    try:
        location = geocode_city(request.city)
        lat, lon = location["latitude"], location["longitude"]
        location_label = f"{location['name']}, {location['country']}"
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {str(e)}")

    # Growth stage
    if request.growth_stage:
        if request.growth_stage not in GROWTH_STAGE_ENCODING:
            raise HTTPException(status_code=400, detail=f"Unknown stage '{request.growth_stage}'.")
        stage_name = request.growth_stage
    else:
        stage_name = get_growth_stage(request.crop, date.today())
    stage_encoded = GROWTH_STAGE_ENCODING[stage_name]

    # Fetch weather
    try:
        weather = fetch_today_weather(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {str(e)}")

    # Predict using crop-specific model
    try:
        prediction = run_prediction(request.crop, weather, stage_encoded)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Convert to liters
    amount_liters = mm_to_liters(prediction["amount_mm"], request.surface_hectares)

    # Build response
    weather_summary = (
        f"{weather['temperature_2m_mean']:.1f}°C, "
        f"{weather['precipitation_sum']:.1f}mm rain, "
        f"{weather['relative_humidity_2m_mean']:.0f}% humidity, "
        f"ET₀={weather['et0_fao_evapotranspiration']:.1f}mm/day"
    )

    crop_display = CROP_DISPLAY_NAMES.get(request.crop, request.crop)

    if stage_name == "fallow":
        message = f"No crop in the field (fallow period for {crop_display}). No irrigation needed."
    elif not prediction["irrigate"]:
        message = (
            f"No irrigation needed today for your {crop_display} "
            f"near {location_label} ({stage_name} stage). Soil moisture is sufficient."
        )
    else:
        message = (
            f"Irrigate your {crop_display} near {location_label} today: "
            f"{prediction['amount_mm']:.1f} mm across {request.surface_hectares} ha "
            f"= {amount_liters:,.0f} liters. "
            f"Crop is in {stage_name} stage."
        )

    return FarmerResponse(
        irrigate=prediction["irrigate"],
        amount_mm=prediction["amount_mm"],
        amount_liters=amount_liters,
        confidence=prediction["confidence"],
        message=message,
        location=location_label,
        crop=request.crop,
        crop_display=crop_display,
        growth_stage=stage_name,
        weather_summary=weather_summary,
        data_sources=weather,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Internal ML prediction endpoint with crop-specific model routing."""
    if request.crop not in SUPPORTED_CROPS:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{request.crop}'.")
    if models.get(request.crop) is None:
        raise HTTPException(status_code=503, detail=f"Model for '{request.crop}' not loaded.")

    features_dict = {
        "temperature_2m_mean": request.temperature_2m_mean,
        "relative_humidity_2m_mean": request.relative_humidity_2m_mean,
        "precipitation_sum": request.precipitation_sum,
        "et0_fao_evapotranspiration": request.et0_fao_evapotranspiration,
        "wind_speed_10m_max": request.wind_speed_10m_max,
        "soil_moisture_0_to_7cm_mean": request.soil_moisture_0_to_7cm_mean,
    }

    prediction = run_prediction(request.crop, features_dict, request.growth_stage_encoded)

    stage_name = GROWTH_STAGE_NAMES.get(request.growth_stage_encoded, "unknown")
    if not prediction["irrigate"]:
        message = f"No irrigation needed for {request.crop} ({stage_name} stage)."
    else:
        message = f"Irrigate {prediction['amount_mm']:.1f} mm for {request.crop} ({stage_name} stage)."

    return PredictResponse(
        irrigate=prediction["irrigate"],
        amount_mm=prediction["amount_mm"],
        confidence=prediction["confidence"],
        message=message,
        crop=request.crop,
    )