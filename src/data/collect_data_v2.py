"""
AquaSmart — Multi-Crop Data Collection Pipeline (v2)
=====================================================
Collects data from 10 major agricultural cities across France,
then generates 5 separate datasets (one per crop) with crop-specific
Kc coefficients and growth stage calendars.

Sources:
  1. Open-Meteo Historical API → weather + ET₀ + soil moisture (daily)
  2. FAO-56 Kc tables          → crop coefficients by growth stage

Zone: France (10 cities covering all major agricultural regions)
Crops: winter_wheat, corn, barley, rapeseed, sunflower
Period: 2020-01-01 to 2024-12-31 (5 years, daily)

Usage:
    python src/data/collect_data_v2.py

Output:
    data/raw/aquasmart_v2_{crop}.csv  (one per crop)
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# 10 cities across France's major agricultural regions
LOCATIONS = {
    # Beauce / Centre-Val de Loire (blé #1)
    "Chartres":     {"lat": 48.45, "lon": 1.50},
    # Picardie / Hauts-de-France (blé, betterave, pomme de terre)
    "Amiens":       {"lat": 49.89, "lon": 2.30},
    # Champagne (blé, orge, colza)
    "Reims":        {"lat": 49.25, "lon": 3.88},
    # Île-de-France (grandes cultures)
    "Melun":        {"lat": 48.54, "lon": 2.66},
    # Grand Est (blé, orge, maïs)
    "Strasbourg":   {"lat": 48.57, "lon": 7.75},
    # Bourgogne (blé, colza, orge)
    "Dijon":        {"lat": 47.32, "lon": 5.04},
    # Aquitaine / Sud-Ouest (maïs, tournesol)
    "Toulouse":     {"lat": 43.60, "lon": 1.44},
    # Occitanie (tournesol, blé dur)
    "Montpellier":  {"lat": 43.61, "lon": 3.87},
    # Ouest / Bretagne (blé, orge, colza)
    "Rennes":       {"lat": 48.11, "lon": -1.68},
    # Nouvelle-Aquitaine (maïs, tournesol, colza)
    "Bordeaux":     {"lat": 44.84, "lon": -0.58},
}

DATE_START = "2020-01-01"
DATE_END = "2024-12-31"

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================
# OPEN-METEO CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
]

# ============================================================
# FAO-56 Kc COEFFICIENTS & GROWTH CALENDARS
# ============================================================
# Source: FAO Irrigation and Drainage Paper 56

CROPS = {
    "winter_wheat": {
        "name": "Winter Wheat (Blé tendre)",
        "kc": {"initial": 0.40, "development": 0.75, "mid_season": 1.15, "late_season": 0.40, "fallow": 0.0},
        "calendar": {
            10: "initial", 11: "initial", 12: "initial",
            1: "development", 2: "development", 3: "development",
            4: "mid_season", 5: "mid_season",
            6: "late_season", 7: "late_season",
            8: "fallow", 9: "fallow",
        },
    },
    "corn": {
        "name": "Corn (Maïs grain)",
        "kc": {"initial": 0.30, "development": 0.75, "mid_season": 1.20, "late_season": 0.60, "fallow": 0.0},
        "calendar": {
            4: "initial", 5: "development",
            6: "mid_season", 7: "mid_season", 8: "mid_season",
            9: "late_season",
            10: "fallow", 11: "fallow", 12: "fallow",
            1: "fallow", 2: "fallow", 3: "fallow",
        },
    },
    "barley": {
        "name": "Winter Barley (Orge d'hiver)",
        "kc": {"initial": 0.40, "development": 0.75, "mid_season": 1.15, "late_season": 0.25, "fallow": 0.0},
        "calendar": {
            10: "initial", 11: "initial", 12: "initial",
            1: "development", 2: "development", 3: "development",
            4: "mid_season", 5: "mid_season",
            6: "late_season",
            7: "fallow", 8: "fallow", 9: "fallow",
        },
    },
    "rapeseed": {
        "name": "Rapeseed (Colza)",
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.15, "late_season": 0.35, "fallow": 0.0},
        "calendar": {
            9: "initial", 10: "initial", 11: "initial",
            12: "development", 1: "development", 2: "development",
            3: "mid_season", 4: "mid_season", 5: "mid_season",
            6: "late_season", 7: "late_season",
            8: "fallow",
        },
    },
    "sunflower": {
        "name": "Sunflower (Tournesol)",
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.15, "late_season": 0.35, "fallow": 0.0},
        "calendar": {
            4: "initial", 5: "development",
            6: "mid_season", 7: "mid_season", 8: "mid_season",
            9: "late_season",
            10: "fallow", 11: "fallow", 12: "fallow",
            1: "fallow", 2: "fallow", 3: "fallow",
        },
    },
}

# ============================================================
# DATA COLLECTION FUNCTIONS
# ============================================================

def fetch_open_meteo_single(lat, lon, start, end, max_retries=5):
    """Fetch daily weather data for a single date range with retry logic."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Paris",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            if response.status_code == 429:
                wait = attempt * 30
                print(f"\n    ⚠️  Rate limited (429). Waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df.rename(columns={"time": "date"}, inplace=True)
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = attempt * 15
            print(f"\n    ⚠️  Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
            if attempt < max_retries:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Failed after {max_retries} attempts for ({lat}, {lon})")


def fetch_open_meteo(lat, lon, start, end):
    """Fetch daily weather data, splitting by year to avoid timeouts."""
    start_year = int(start[:4])
    end_year = int(end[:4])

    chunks = []
    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01" if year > start_year else start
        y_end = f"{year}-12-31" if year < end_year else end
        print(f"    {year}...", end=" ", flush=True)
        chunk = fetch_open_meteo_single(lat, lon, y_start, y_end)
        chunks.append(chunk)
        print(f"✓ ({len(chunk)} rows)")
        time.sleep(3)  # 3s between calls to stay under rate limit

    return pd.concat(chunks, ignore_index=True)


def compute_crop_features(df, crop_key):
    """
    Add crop-specific features: growth_stage, kc, etc_mm, water_need_index.
    """
    crop = CROPS[crop_key]
    df = df.copy()

    # Growth stage from date
    df["growth_stage"] = df["date"].dt.month.map(crop["calendar"])

    # Kc coefficient
    df["kc"] = df["growth_stage"].map(crop["kc"])

    # Crop evapotranspiration: ETc = ET₀ × Kc
    df["etc_mm"] = df["et0_fao_evapotranspiration"] * df["kc"]

    # Effective rainfall (USDA simplified: ~80% usable)
    df["effective_rainfall_mm"] = df["precipitation_sum"].clip(lower=0) * 0.8

    # Water Need Index (mm/day) = max(0, ETc - effective_rainfall)
    df["water_need_index"] = (df["etc_mm"] - df["effective_rainfall_mm"]).clip(lower=0)

    # Binary irrigation flag (threshold: 0.5 mm/day)
    df["irrigation_needed"] = (df["water_need_index"] > 0.5).astype(int)

    # Encode growth stage for ML
    stage_encoding = {"fallow": 0, "initial": 1, "development": 2, "mid_season": 3, "late_season": 4}
    df["growth_stage_encoded"] = df["growth_stage"].map(stage_encoding)

    # Add crop label
    df["crop"] = crop_key

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_collection():
    """Execute the full multi-location, multi-crop collection pipeline."""
    print("=" * 60)
    print("AquaSmart v2 — Multi-Crop Data Collection")
    print("=" * 60)
    print(f"Locations: {len(LOCATIONS)} cities across France")
    print(f"Crops:     {len(CROPS)} ({', '.join(CROPS.keys())})")
    print(f"Period:    {DATE_START} to {DATE_END}")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Collect weather data (same for all crops) ----
    print("\n📡 STEP 1: Collecting weather data from Open-Meteo...")

    # Check if we already have partial weather data (resume support)
    weather_cache_path = RAW_DIR / "weather_cache.csv"
    if weather_cache_path.exists():
        cached = pd.read_csv(weather_cache_path, parse_dates=["date"])
        cached_locations = set(cached["location"].unique())
        print(f"  📦 Found cached weather for: {', '.join(cached_locations)}")
        all_weather = [cached]
    else:
        cached_locations = set()
        all_weather = []

    for location_name, coords in LOCATIONS.items():
        if location_name in cached_locations:
            print(f"\n  📍 {location_name} — already cached, skipping")
            continue

        print(f"\n  📍 {location_name} ({coords['lat']}, {coords['lon']})")
        weather_df = fetch_open_meteo(
            coords["lat"], coords["lon"], DATE_START, DATE_END
        )
        weather_df["location"] = location_name
        weather_df["latitude"] = coords["lat"]
        weather_df["longitude"] = coords["lon"]
        all_weather.append(weather_df)
        print(f"    ✅ {len(weather_df)} daily records")

        # Save progress after each city
        progress = pd.concat(all_weather, ignore_index=True)
        progress.to_csv(weather_cache_path, index=False)
        print(f"    💾 Progress saved ({len(progress)} total rows)")

    weather_all = pd.concat(all_weather, ignore_index=True)
    print(f"\n  Total weather records: {len(weather_all)}")

    # ---- Step 2: Generate one dataset per crop ----
    print("\n🌾 STEP 2: Generating crop-specific datasets...")

    for crop_key, crop_info in CROPS.items():
        print(f"\n  🌱 {crop_info['name']}...")

        crop_df = compute_crop_features(weather_all, crop_key)

        # Save
        filename = f"aquasmart_v2_{crop_key}.csv"
        crop_df.to_csv(RAW_DIR / filename, index=False)

        # Stats
        irr_pct = crop_df["irrigation_needed"].mean() * 100
        wni_mean = crop_df["water_need_index"].mean()
        print(f"    Records: {len(crop_df)}")
        print(f"    Irrigation days: {irr_pct:.1f}%")
        print(f"    Mean Water Need: {wni_mean:.2f} mm/day")
        print(f"    Saved: {filename}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  Weather collected for {len(LOCATIONS)} locations")
    print(f"  Datasets generated for {len(CROPS)} crops")
    print(f"  Files saved in: {RAW_DIR}")
    print("\n  Next step: run preprocessing + training")
    print("    python src/data/preprocess_v2.py")
    print("    python src/models/train_v2.py")

    return weather_all


if __name__ == "__main__":
    run_collection()