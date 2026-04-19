"""
AquaSmart — Shared recommendation engine (v4).

Used by both the Streamlit app and the FastAPI backend.

Key differences vs v2/v3:
  - Models are grouped by agronomic CLUSTER, not by individual crop.
    15 crops → 4 clusters → 4 trained model bundles (classifier + regressor).
  - The crop the user selects determines the cluster, but also provides
    static crop features (Kc, RU_max) that differentiate crops within a cluster.
  - Feature vector has 16 dimensions (9 static + 7 temporal history).
  - Users supply recent irrigation history (mm applied in last 7 days,
    days since last irrigation) so temporal features are not defaulted.

Public API (unchanged from v3 for Streamlit/FastAPI compatibility):
  - build_farmer_recommendation(city, surface_hectares, crop, ...)
  - get_model_status()
  - get_model_info()
  - get_growth_stage(crop, date=None)
  - refresh_model_store()
  - SUPPORTED_CROPS, CROP_DISPLAY_NAMES, GROWTH_STAGE_ENCODING
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------
# CROP CONFIGURATION — must match generate_target_v4.py
# ---------------------------------------------------------------
# Each crop belongs to one of 4 agronomic clusters and carries static
# parameters (Kc by stage, RU_max, sowing month) used both to derive the
# growth stage and to build the feature vector at inference time.
CROPS: dict[str, dict[str, Any]] = {
    # --- Cluster 1: winter cereals ---
    "winter_wheat": {
        "cluster": 1, "ru_max": 120, "sowing_month": 10,
        "display": "Winter Wheat (Blé tendre)",
        "kc": {"initial": 0.40, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.40, "fallow": 0.0},
    },
    "durum_wheat": {
        "cluster": 1, "ru_max": 115, "sowing_month": 10,
        "display": "Durum Wheat (Blé dur)",
        "kc": {"initial": 0.40, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.40, "fallow": 0.0},
    },
    "winter_barley": {
        "cluster": 1, "ru_max": 110, "sowing_month": 10,
        "display": "Winter Barley (Orge d'hiver)",
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.25, "fallow": 0.0},
    },
    "oats": {
        "cluster": 1, "ru_max": 105, "sowing_month": 10,
        "display": "Oats (Avoine)",
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.25, "fallow": 0.0},
    },
    "triticale": {
        "cluster": 1, "ru_max": 115, "sowing_month": 10,
        "display": "Triticale",
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.35, "fallow": 0.0},
    },
    # --- Cluster 2: summer deep-rooted crops ---
    "corn": {
        "cluster": 2, "ru_max": 150, "sowing_month": 4,
        "display": "Corn (Maïs)",
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.20,
               "late_season": 0.60, "fallow": 0.0},
    },
    "sunflower": {
        "cluster": 2, "ru_max": 140, "sowing_month": 4,
        "display": "Sunflower (Tournesol)",
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.35, "fallow": 0.0},
    },
    "sorghum": {
        "cluster": 2, "ru_max": 135, "sowing_month": 5,
        "display": "Sorghum (Sorgho)",
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.10,
               "late_season": 0.55, "fallow": 0.0},
    },
    "soybean": {
        "cluster": 2, "ru_max": 130, "sowing_month": 5,
        "display": "Soybean (Soja)",
        "kc": {"initial": 0.40, "development": 0.80, "mid_season": 1.15,
               "late_season": 0.50, "fallow": 0.0},
    },
    # --- Cluster 3: winter oilseeds / legumes ---
    "rapeseed": {
        "cluster": 3, "ru_max": 130, "sowing_month": 9,
        "display": "Rapeseed (Colza)",
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15,
               "late_season": 0.35, "fallow": 0.0},
    },
    "winter_pea": {
        "cluster": 3, "ru_max": 120, "sowing_month": 11,
        "display": "Winter Pea (Pois d'hiver)",
        "kc": {"initial": 0.50, "development": 0.80, "mid_season": 1.15,
               "late_season": 0.30, "fallow": 0.0},
    },
    "faba_bean": {
        "cluster": 3, "ru_max": 125, "sowing_month": 11,
        "display": "Faba Bean (Féverole)",
        "kc": {"initial": 0.50, "development": 0.80, "mid_season": 1.15,
               "late_season": 0.30, "fallow": 0.0},
    },
    # --- Cluster 4: shallow-rooted row crops ---
    "potato": {
        "cluster": 4, "ru_max": 70, "sowing_month": 4,
        "display": "Potato (Pomme de terre)",
        "kc": {"initial": 0.50, "development": 0.75, "mid_season": 1.15,
               "late_season": 0.75, "fallow": 0.0},
    },
    "sugar_beet": {
        "cluster": 4, "ru_max": 85, "sowing_month": 3,
        "display": "Sugar Beet (Betterave sucrière)",
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.20,
               "late_season": 0.70, "fallow": 0.0},
    },
    "field_vegetables": {
        "cluster": 4, "ru_max": 65, "sowing_month": 4,
        "display": "Field Vegetables (Légumes de plein champ)",
        "kc": {"initial": 0.50, "development": 0.75, "mid_season": 1.05,
               "late_season": 0.90, "fallow": 0.0},
    },
}

SUPPORTED_CROPS = list(CROPS.keys())
CROP_DISPLAY_NAMES = {k: v["display"] for k, v in CROPS.items()}
CLUSTER_NAMES = {
    1: "Winter cereals",
    2: "Summer deep-rooted crops",
    3: "Winter oilseeds / legumes",
    4: "Shallow-rooted row crops",
}

GROWTH_STAGE_ENCODING = {
    "fallow": 0,
    "initial": 1,
    "development": 2,
    "mid_season": 3,
    "late_season": 4,
}
GROWTH_STAGE_NAMES = {value: key for key, value in GROWTH_STAGE_ENCODING.items()}

# Feature order MUST match what the models were trained on (preprocess_v4.py)
STATIC_FEATURES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "kc",
    "etc_mm",
    "ru_max",
]
TEMPORAL_FEATURES = [
    "soil_stock_mm_prev",
    "rainfall_cumsum_7d",
    "etc_cumsum_7d",
    "water_balance_7d",
    "irrigation_cumsum_7d",
    "irrigation_cumsum_14d",
    "days_since_last_irrigation",
]
FEATURE_COLS = STATIC_FEATURES + TEMPORAL_FEATURES


# ---------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------
class RecommendationError(RuntimeError):
    """Base recommendation error."""


class UnknownCropError(RecommendationError):
    """Raised when the crop is not in the supported list."""


class UnknownStageError(RecommendationError):
    """Raised when a manually supplied growth stage is not recognized."""


class CityNotFoundError(RecommendationError):
    """Raised when the geocoding API returns no match."""


class ExternalServiceError(RecommendationError):
    """Raised when an external API (geocoding, weather) fails."""


class ModelNotReadyError(RecommendationError):
    """Raised when required model artefacts are missing on disk."""


# ---------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------
def validate_crop(crop: str) -> str:
    if crop not in CROPS:
        raise UnknownCropError(
            f"Unknown crop '{crop}'. Available: {', '.join(SUPPORTED_CROPS)}."
        )
    return crop


def validate_stage(stage_name: str) -> str:
    if stage_name not in GROWTH_STAGE_ENCODING:
        raise UnknownStageError(
            f"Unknown growth stage '{stage_name}'. "
            f"Available: {', '.join(GROWTH_STAGE_ENCODING)}."
        )
    return stage_name


# ---------------------------------------------------------------
# GROWTH STAGE (derived from sowing month, matches generate_target_v4.py)
# ---------------------------------------------------------------
def get_growth_stage(crop: str, target_date: date | None = None) -> str:
    """Map a (crop, date) pair to a growth stage using the sowing-month model."""
    crop = validate_crop(crop)
    d = target_date or date.today()
    sowing = CROPS[crop]["sowing_month"]
    months_since_sowing = (d.month - sowing) % 12
    if months_since_sowing < 2:
        return "initial"
    elif months_since_sowing < 4:
        return "development"
    elif months_since_sowing < 6:
        return "mid_season"
    elif months_since_sowing < 8:
        return "late_season"
    return "fallow"


# ---------------------------------------------------------------
# MODEL STORE (cluster-based bundles)
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model_store() -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """
    Load the 4 cluster bundles.
    Each bundle contains: classifier, regressor, scaler.
    Returns (bundles_by_cluster_id, errors_by_cluster_id).
    """
    bundles: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}

    for cluster_id in [1, 2, 3, 4]:
        cdir = MODEL_DIR / f"cluster_{cluster_id}"
        try:
            classifier = joblib.load(cdir / "classifier.pkl")
            regressor_path = cdir / "regressor.pkl"
            regressor = joblib.load(regressor_path) if regressor_path.exists() else None
            # Scaler is saved by preprocess_v4.py in data/processed/cluster_X/scaler.pkl
            scaler_path = PROJECT_ROOT / "data" / "processed" / f"cluster_{cluster_id}" / "scaler.pkl"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None
            if scaler is None:
                raise FileNotFoundError(
                    f"Scaler not found at {scaler_path}. "
                    "Run src/data/preprocess_v4.py first."
                )
            bundles[cluster_id] = {
                "classifier": classifier,
                "regressor": regressor,
                "scaler": scaler,
            }
        except Exception as exc:  # noqa: BLE001
            errors[cluster_id] = str(exc)

    return bundles, errors


def refresh_model_store() -> None:
    """Clear the cached model store (call after retraining)."""
    load_model_store.cache_clear()


def get_model_status() -> dict[str, Any]:
    """Return the status of all cluster bundles — exposed to the UI."""
    bundles, errors = load_model_store()
    loaded_clusters = sorted(bundles.keys())
    missing_clusters = [c for c in [1, 2, 3, 4] if c not in bundles]

    # For UI compatibility, also expose crop-level view
    loaded_crops = [
        c for c, cfg in CROPS.items() if cfg["cluster"] in loaded_clusters
    ]
    missing_crops = [
        c for c, cfg in CROPS.items() if cfg["cluster"] not in loaded_clusters
    ]

    return {
        "loaded_clusters": loaded_clusters,
        "missing_clusters": missing_clusters,
        "cluster_errors": errors,
        "loaded_crops": loaded_crops,
        "missing_crops": missing_crops,
        "crops_total": len(SUPPORTED_CROPS),
        "clusters_total": 4,
    }


def get_model_info() -> list[dict[str, Any]]:
    """
    Per-cluster performance info.
    Reads models/results_v4.csv (produced by train_v4.py) if available.
    """
    results_file = MODEL_DIR / "results_v4.csv"
    if not results_file.exists():
        return []
    df = pd.read_csv(results_file)
    infos: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cluster_id = int(row["cluster"])
        baseline_mae = float(row["baseline_ru_mae"])
        ml_mae = float(row["ml_mae"])
        improvement = (1 - ml_mae / baseline_mae) * 100 if baseline_mae > 0 else 0.0
        infos.append({
            "cluster": cluster_id,
            "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}"),
            "crops": [
                CROP_DISPLAY_NAMES[c] for c, cfg in CROPS.items()
                if cfg["cluster"] == cluster_id
            ],
            "baseline_ru_f1": round(float(row["baseline_ru_f1"]), 4),
            "baseline_ru_mae": round(baseline_mae, 4),
            "ml_f1": round(float(row["ml_f1"]), 4),
            "ml_mae": round(ml_mae, 4),
            "ml_rmse": round(float(row["ml_rmse"]), 4),
            "baseline_improvement": round(improvement, 1),
        })
    return infos


# ---------------------------------------------------------------
# EXTERNAL APIs (geocoding + weather)
# ---------------------------------------------------------------
def geocode_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates via Open-Meteo's geocoding API."""
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
    results = data.get("results", [])
    if not results:
        raise CityNotFoundError(f"City '{city}' was not found. Check the spelling.")
    m = results[0]
    return {
        "name": m.get("name", city),
        "latitude": m["latitude"],
        "longitude": m["longitude"],
        "country": m.get("country", ""),
        "admin1": m.get("admin1", ""),
    }


def _daily_value(daily: dict[str, list[Any]], key: str) -> float:
    vals = daily.get(key, [])
    v = vals[0] if vals else None
    if v is None:
        raise ExternalServiceError(f"Open-Meteo returned no value for '{key}'.")
    return float(v)


def _mean_non_null(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def fetch_today_weather(lat: float, lon: float,
                        target_date: date | None = None) -> dict[str, Any]:
    """Fetch same-day weather from Open-Meteo's forecast API."""
    d = (target_date or date.today()).isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "wind_speed_10m_max",
        ]),
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_0_to_1cm",
        "start_date": d,
        "end_date": d,
        "timezone": "auto",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise ExternalServiceError(f"Weather fetch failed: {exc}") from exc

    data = r.json()
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    sm_7cm = _mean_non_null(hourly.get("soil_moisture_0_to_7cm", []))
    sm_1cm = _mean_non_null(hourly.get("soil_moisture_0_to_1cm", []))
    if sm_7cm is not None:
        soil_moisture, src = sm_7cm, "hourly 0-7cm mean"
    elif sm_1cm is not None:
        soil_moisture, src = sm_1cm, "hourly 0-1cm mean (fallback)"
    else:
        soil_moisture, src = 0.30, "default 0.30 (fallback)"

    return {
        "temperature_2m_mean": _daily_value(daily, "temperature_2m_mean"),
        "relative_humidity_2m_mean": _daily_value(daily, "relative_humidity_2m_mean"),
        "precipitation_sum": _daily_value(daily, "precipitation_sum"),
        "et0_fao_evapotranspiration": _daily_value(daily, "et0_fao_evapotranspiration"),
        "wind_speed_10m_max": _daily_value(daily, "wind_speed_10m_max"),
        "soil_moisture_0_to_7cm_mean": soil_moisture,
        "soil_moisture_source": src,
        "forecast_date": d,
    }


# ---------------------------------------------------------------
# FEATURE BUILDING
# ---------------------------------------------------------------
def build_feature_vector(
    crop: str,
    stage_name: str,
    weather: dict[str, Any],
    irrigation_last_7d: float,
    irrigation_last_14d: float,
    days_since_last_irrigation: float,
    rainfall_last_7d: float | None = None,
    soil_stock_prev: float | None = None,
) -> pd.DataFrame:
    """
    Assemble the 16-dim feature row used by the cluster models.

    Static features come from today's weather + crop parameters.
    Temporal features come from user-supplied history (or sensible defaults).
    """
    cfg = CROPS[crop]
    kc = cfg["kc"].get(stage_name, 0.0)
    etc = weather["et0_fao_evapotranspiration"] * kc
    ru_max = cfg["ru_max"]

    # Sensible estimates when the user does not provide these
    if rainfall_last_7d is None:
        rainfall_last_7d = weather["precipitation_sum"] * 7  # crude proxy
    etc_cumsum_7d = etc * 7  # assume stable ETc over the week
    water_balance_7d = rainfall_last_7d - etc_cumsum_7d
    if soil_stock_prev is None:
        # Start from a neutral half-full soil if no sensor data
        soil_stock_prev = ru_max * 0.5 + water_balance_7d / 2 + irrigation_last_7d
        soil_stock_prev = float(np.clip(soil_stock_prev, 0.0, ru_max))

    row = {
        "temperature_2m_mean": weather["temperature_2m_mean"],
        "relative_humidity_2m_mean": weather["relative_humidity_2m_mean"],
        "precipitation_sum": weather["precipitation_sum"],
        "et0_fao_evapotranspiration": weather["et0_fao_evapotranspiration"],
        "wind_speed_10m_max": weather["wind_speed_10m_max"],
        "soil_moisture_0_to_7cm_mean": weather["soil_moisture_0_to_7cm_mean"],
        "kc": kc,
        "etc_mm": etc,
        "ru_max": ru_max,
        "soil_stock_mm_prev": soil_stock_prev,
        "rainfall_cumsum_7d": rainfall_last_7d,
        "etc_cumsum_7d": etc_cumsum_7d,
        "water_balance_7d": water_balance_7d,
        "irrigation_cumsum_7d": irrigation_last_7d,
        "irrigation_cumsum_14d": irrigation_last_14d,
        "days_since_last_irrigation": float(min(days_since_last_irrigation, 30.0)),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ---------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------
def get_confidence(probabilities: np.ndarray) -> str:
    max_p = float(probabilities.max())
    if max_p >= 0.85:
        return "high"
    if max_p >= 0.65:
        return "medium"
    return "low"


def mm_to_liters(mm: float, hectares: float) -> float:
    return round(mm * hectares * 10_000, 0)


def run_prediction(crop: str, features: pd.DataFrame) -> dict[str, Any]:
    """Run Stage 1 (classification) and, if positive, Stage 2 (regression)."""
    cfg = CROPS[crop]
    cluster_id = cfg["cluster"]
    bundles, errors = load_model_store()
    bundle = bundles.get(cluster_id)
    if bundle is None:
        detail = errors.get(cluster_id, "missing bundle")
        raise ModelNotReadyError(
            f"Cluster {cluster_id} model not ready: {detail}. "
            "Run src/models/train_v4.py to (re)generate it."
        )

    scaler = bundle["scaler"]
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]

    features_scaled = scaler.transform(features)
    irrigate_flag = int(classifier.predict(features_scaled)[0])

    # Some classifiers (e.g. all-negative train) can lack predict_proba
    if hasattr(classifier, "predict_proba"):
        probas = classifier.predict_proba(features_scaled)[0]
    else:
        probas = np.array([1.0, 0.0]) if irrigate_flag == 0 else np.array([0.0, 1.0])

    amount_mm = 0.0
    if irrigate_flag == 1 and regressor is not None:
        amount_mm = float(regressor.predict(features_scaled)[0])
        amount_mm = max(0.0, round(amount_mm, 2))

    return {
        "irrigate": bool(irrigate_flag == 1),
        "amount_mm": amount_mm,
        "confidence": get_confidence(probas),
        "cluster": cluster_id,
    }


# ---------------------------------------------------------------
# MAIN PUBLIC ENTRY POINT
# ---------------------------------------------------------------
def build_farmer_recommendation(
    city: str,
    surface_hectares: float,
    crop: str,
    growth_stage: str | None = None,
    target_date: date | None = None,
    irrigation_last_7d: float = 0.0,
    irrigation_last_14d: float | None = None,
    days_since_last_irrigation: float = 30.0,
) -> dict[str, Any]:
    """
    Produce a farmer-facing irrigation recommendation.

    Args:
        city: free-text city name (will be geocoded)
        surface_hectares: field area (mm → liters conversion)
        crop: one of SUPPORTED_CROPS
        growth_stage: optional override (else auto-detected from date)
        target_date: optional override (else today)
        irrigation_last_7d: mm of irrigation applied in the past 7 days
        irrigation_last_14d: mm over 14 days (defaults to 7d value if unset)
        days_since_last_irrigation: capped at 30 internally

    Returns a dict with: irrigate, amount_mm, amount_liters, confidence,
    message, location, crop, crop_display, cluster, cluster_name,
    growth_stage, weather_summary, data_sources, recommendation_date.
    """
    crop = validate_crop(crop)
    if surface_hectares <= 0:
        raise RecommendationError("Surface area must be greater than 0 hectares.")
    if irrigation_last_14d is None:
        irrigation_last_14d = irrigation_last_7d

    current_date = target_date or date.today()
    stage_name = (validate_stage(growth_stage) if growth_stage
                  else get_growth_stage(crop, current_date))

    cfg = CROPS[crop]
    cluster_id = cfg["cluster"]
    cluster_name = CLUSTER_NAMES[cluster_id]
    crop_display = cfg["display"]

    location = geocode_city(city)
    location_label = f"{location['name']}, {location['country']}".strip(", ")

    weather = fetch_today_weather(
        location["latitude"], location["longitude"], current_date)

    if stage_name == "fallow":
        prediction = {"irrigate": False, "amount_mm": 0.0,
                      "confidence": "high", "cluster": cluster_id}
        message = (f"No crop in the field (fallow period for {crop_display}). "
                   "No irrigation needed.")
    else:
        features = build_feature_vector(
            crop=crop,
            stage_name=stage_name,
            weather=weather,
            irrigation_last_7d=irrigation_last_7d,
            irrigation_last_14d=irrigation_last_14d,
            days_since_last_irrigation=days_since_last_irrigation,
        )
        prediction = run_prediction(crop, features)
        if prediction["irrigate"]:
            liters = mm_to_liters(prediction["amount_mm"], surface_hectares)
            message = (
                f"Irrigate your {crop_display} near {location_label} today: "
                f"{prediction['amount_mm']:.1f} mm × {surface_hectares} ha "
                f"= {liters:,.0f} liters. Crop is in {stage_name} stage."
            )
        else:
            message = (
                f"No irrigation needed today for your {crop_display} near "
                f"{location_label} ({stage_name} stage). Soil moisture is sufficient."
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
        "cluster": cluster_id,
        "cluster_name": cluster_name,
        "growth_stage": stage_name,
        "weather_summary": weather_summary,
        "data_sources": weather,
        "recommendation_date": current_date.isoformat(),
    }