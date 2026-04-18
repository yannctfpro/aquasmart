"""Shared recommendation engine used by the API and Streamlit app."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

SUPPORTED_CROPS = ["winter_wheat", "corn", "barley", "rapeseed", "sunflower"]
FEATURE_COLS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "growth_stage_encoded",
]

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
    "fallow": 0,
    "initial": 1,
    "development": 2,
    "mid_season": 3,
    "late_season": 4,
}
GROWTH_STAGE_NAMES = {value: key for key, value in GROWTH_STAGE_ENCODING.items()}

CROP_DISPLAY_NAMES = {
    "winter_wheat": "Winter Wheat (Blé tendre)",
    "corn": "Corn (Maïs)",
    "barley": "Barley (Orge)",
    "rapeseed": "Rapeseed (Colza)",
    "sunflower": "Sunflower (Tournesol)",
}


class RecommendationError(RuntimeError):
    """Base recommendation error."""


class UnknownCropError(RecommendationError):
    """Raised when the crop is unsupported."""


class UnknownStageError(RecommendationError):
    """Raised when the growth stage is unsupported."""


class CityNotFoundError(RecommendationError):
    """Raised when the city cannot be geocoded."""


class ExternalServiceError(RecommendationError):
    """Raised when an external API request fails."""


class ModelNotReadyError(RecommendationError):
    """Raised when model artifacts are missing."""


def validate_crop(crop: str) -> str:
    if crop not in SUPPORTED_CROPS:
        raise UnknownCropError(
            f"Unknown crop '{crop}'. Available crops: {', '.join(SUPPORTED_CROPS)}."
        )
    return crop


def validate_stage(stage_name: str) -> str:
    if stage_name not in GROWTH_STAGE_ENCODING:
        raise UnknownStageError(
            f"Unknown growth stage '{stage_name}'. "
            f"Available stages: {', '.join(GROWTH_STAGE_ENCODING)}."
        )
    return stage_name


@lru_cache(maxsize=1)
def load_model_store() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load all available crop model bundles once."""
    loaded_models: dict[str, dict[str, Any]] = {}
    model_errors: dict[str, str] = {}

    for crop in SUPPORTED_CROPS:
        crop_dir = MODEL_DIR / crop
        try:
            loaded_models[crop] = {
                "classifier": joblib.load(crop_dir / "classifier.joblib"),
                "regressor": joblib.load(crop_dir / "regressor.joblib"),
                "scaler": joblib.load(crop_dir / "scaler.joblib"),
                "results": json.loads((crop_dir / "results.json").read_text(encoding="utf-8")),
            }
        except Exception as exc:
            model_errors[crop] = str(exc)

    return loaded_models, model_errors


def refresh_model_store() -> None:
    load_model_store.cache_clear()


def get_model_status() -> dict[str, Any]:
    loaded_models, model_errors = load_model_store()
    loaded_crops = [crop for crop in SUPPORTED_CROPS if crop in loaded_models]
    missing_crops = [crop for crop in SUPPORTED_CROPS if crop not in loaded_models]
    return {
        "loaded_crops": loaded_crops,
        "missing_crops": missing_crops,
        "model_errors": model_errors,
        "crops_total": len(SUPPORTED_CROPS),
    }


def get_growth_stage(crop: str, target_date: date | None = None) -> str:
    crop = validate_crop(crop)
    current_date = target_date or date.today()
    calendar = CROP_STAGE_CALENDAR[crop]
    return calendar.get(current_date.month, "fallow")


def geocode_city(city: str) -> dict[str, Any]:
    """Convert a city name to coordinates via Open-Meteo geocoding."""
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExternalServiceError(f"Geocoding failed: {exc}") from exc

    data = response.json()
    results_list = data.get("results", [])
    if not results_list:
        raise CityNotFoundError(f"City '{city}' was not found. Please check the spelling.")

    match = results_list[0]
    return {
        "name": match.get("name", city),
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "country": match.get("country", ""),
        "admin1": match.get("admin1", ""),
    }


def _daily_value(daily_block: dict[str, list[Any]], key: str) -> float:
    values = daily_block.get(key, [])
    value = values[0] if values else None
    if value is None:
        raise ExternalServiceError(f"Open-Meteo returned no value for '{key}'.")
    return float(value)


def _mean_non_null(values: list[Any]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(float(sum(filtered) / len(filtered)), 4)


def fetch_today_weather(lat: float, lon: float, target_date: date | None = None) -> dict[str, Any]:
    """Fetch same-day weather features from Open-Meteo forecast."""
    selected_date = (target_date or date.today()).isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "precipitation_sum",
                "et0_fao_evapotranspiration",
                "wind_speed_10m_max",
            ]
        ),
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_0_to_1cm",
        "start_date": selected_date,
        "end_date": selected_date,
        "timezone": "auto",
    }

    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExternalServiceError(f"Weather fetch failed: {exc}") from exc

    data = response.json()
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    soil_moisture_7cm = _mean_non_null(hourly.get("soil_moisture_0_to_7cm", []))
    soil_moisture_1cm = _mean_non_null(hourly.get("soil_moisture_0_to_1cm", []))

    soil_moisture = soil_moisture_7cm
    soil_source = "hourly soil_moisture_0_to_7cm mean"
    if soil_moisture is None and soil_moisture_1cm is not None:
        soil_moisture = soil_moisture_1cm
        soil_source = "hourly soil_moisture_0_to_1cm mean (fallback)"
    if soil_moisture is None:
        soil_moisture = 0.30
        soil_source = "default fallback (0.30 volumetric water content)"

    return {
        "temperature_2m_mean": _daily_value(daily, "temperature_2m_mean"),
        "relative_humidity_2m_mean": _daily_value(daily, "relative_humidity_2m_mean"),
        "precipitation_sum": _daily_value(daily, "precipitation_sum"),
        "et0_fao_evapotranspiration": _daily_value(daily, "et0_fao_evapotranspiration"),
        "wind_speed_10m_max": _daily_value(daily, "wind_speed_10m_max"),
        "soil_moisture_0_to_7cm_mean": soil_moisture,
        "soil_moisture_source": soil_source,
        "forecast_date": selected_date,
    }


def get_confidence(probabilities: np.ndarray) -> str:
    max_probability = float(probabilities.max())
    if max_probability >= 0.85:
        return "high"
    if max_probability >= 0.65:
        return "medium"
    return "low"


def mm_to_liters(mm: float, hectares: float) -> float:
    return round(mm * hectares * 10_000, 0)


def run_prediction(crop: str, features_dict: dict[str, float], growth_stage_encoded: int) -> dict[str, Any]:
    crop = validate_crop(crop)
    loaded_models, model_errors = load_model_store()
    crop_models = loaded_models.get(crop)
    if crop_models is None:
        details = model_errors.get(crop, "missing model files")
        raise ModelNotReadyError(f"Model for '{crop}' is not ready: {details}")

    features = pd.DataFrame(
        [[
            features_dict["temperature_2m_mean"],
            features_dict["relative_humidity_2m_mean"],
            features_dict["precipitation_sum"],
            features_dict["et0_fao_evapotranspiration"],
            features_dict["wind_speed_10m_max"],
            features_dict["soil_moisture_0_to_7cm_mean"],
            growth_stage_encoded,
        ]],
        columns=FEATURE_COLS,
    )

    features_scaled = pd.DataFrame(
        crop_models["scaler"].transform(features),
        columns=FEATURE_COLS,
    )
    irrigate_prediction = crop_models["classifier"].predict(features_scaled)[0]
    probabilities = crop_models["classifier"].predict_proba(features_scaled)[0]

    amount_mm = 0.0
    if int(irrigate_prediction) == 1:
        amount_mm = float(crop_models["regressor"].predict(features_scaled)[0])
        amount_mm = max(0.0, round(amount_mm, 2))

    return {
        "irrigate": bool(int(irrigate_prediction) == 1),
        "amount_mm": amount_mm,
        "confidence": get_confidence(probabilities),
    }


def build_farmer_recommendation(
    city: str,
    surface_hectares: float,
    crop: str,
    growth_stage: str | None = None,
    target_date: date | None = None,
) -> dict[str, Any]:
    crop = validate_crop(crop)
    if surface_hectares <= 0:
        raise RecommendationError("Surface area must be greater than 0 hectares.")

    current_date = target_date or date.today()
    location = geocode_city(city)
    location_label = f"{location['name']}, {location['country']}".strip(", ")

    stage_name = validate_stage(growth_stage) if growth_stage else get_growth_stage(crop, current_date)
    stage_encoded = GROWTH_STAGE_ENCODING[stage_name]
    crop_display = CROP_DISPLAY_NAMES.get(crop, crop)

    weather = fetch_today_weather(location["latitude"], location["longitude"], current_date)

    if stage_name == "fallow":
        prediction = {"irrigate": False, "amount_mm": 0.0, "confidence": "high"}
        message = f"No crop in the field (fallow period for {crop_display}). No irrigation needed."
    else:
        prediction = run_prediction(crop, weather, stage_encoded)
        if prediction["irrigate"]:
            amount_liters = mm_to_liters(prediction["amount_mm"], surface_hectares)
            message = (
                f"Irrigate your {crop_display} near {location_label} today: "
                f"{prediction['amount_mm']:.1f} mm across {surface_hectares} ha "
                f"= {amount_liters:,.0f} liters. Crop is in {stage_name} stage."
            )
        else:
            message = (
                f"No irrigation needed today for your {crop_display} near {location_label} "
                f"({stage_name} stage). Soil moisture is sufficient."
            )

    amount_liters = mm_to_liters(prediction["amount_mm"], surface_hectares)
    weather_summary = (
        f"{weather['temperature_2m_mean']:.1f} C, "
        f"{weather['precipitation_sum']:.1f} mm rain, "
        f"{weather['relative_humidity_2m_mean']:.0f}% humidity, "
        f"ET0={weather['et0_fao_evapotranspiration']:.1f} mm/day"
    )

    return {
        "irrigate": prediction["irrigate"],
        "amount_mm": prediction["amount_mm"],
        "amount_liters": amount_liters,
        "confidence": prediction["confidence"],
        "message": message,
        "location": location_label,
        "crop": crop,
        "crop_display": crop_display,
        "growth_stage": stage_name,
        "weather_summary": weather_summary,
        "data_sources": weather,
        "recommendation_date": current_date.isoformat(),
    }


def get_model_info() -> list[dict[str, Any]]:
    loaded_models, _ = load_model_store()
    infos: list[dict[str, Any]] = []

    for crop in SUPPORTED_CROPS:
        crop_models = loaded_models.get(crop)
        if crop_models is None:
            continue

        results = crop_models["results"]
        best_classifier = results["best_classifier"]
        best_regressor = results["best_regressor"]
        classifier_metrics = results["classification"][best_classifier]

        if best_regressor == "None" or best_regressor not in results["regression"]:
            continue

        regressor_metrics = results["regression"][best_regressor]
        baseline_mae = results["baseline"]["reg_mae"]
        ai_mae = regressor_metrics["mae"]
        improvement = (1 - ai_mae / baseline_mae) * 100 if baseline_mae > 0 else 0

        infos.append(
            {
                "crop": crop,
                "crop_display": CROP_DISPLAY_NAMES.get(crop, crop),
                "classifier": best_classifier,
                "classifier_f1": round(float(classifier_metrics["f1"]), 4),
                "regressor": best_regressor,
                "regressor_rmse": round(float(regressor_metrics["rmse"]), 4),
                "regressor_r2": round(float(regressor_metrics["r2"]), 4),
                "baseline_improvement": round(float(improvement), 1),
            }
        )

    return infos
