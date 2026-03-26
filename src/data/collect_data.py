"""
AquaSmart — Data Collection Pipeline (Sprint 1)
================================================
Collects and assembles a real dataset from official sources:
  1. Open-Meteo Historical API → weather + ET₀ + soil moisture (daily)
  2. ISRIC SoilGrids REST API  → static soil properties (texture, pH, organic carbon)
  3. FAO-56 Kc tables          → crop coefficients by growth stage

Target variable: Water Need Index (mm/day) = max(0, ETc - effective_rainfall)
Where ETc = ET₀ × Kc (FAO Penman-Monteith method)

Zone: Beauce (France), major wheat-producing region
Crop: Winter wheat (blé tendre)
Period: 2020-01-01 to 2024-12-31 (5 years, daily)

Usage:
    python src/data/collect_data.py
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Beauce region — 5 locations to add spatial diversity
LOCATIONS = {
    "Chartres":     {"lat": 48.45, "lon": 1.50},
    "Orleans":      {"lat": 47.90, "lon": 1.90},
    "Pithiviers":   {"lat": 48.17, "lon": 2.25},
    "Chateaudun":   {"lat": 48.07, "lon": 1.34},
    "Etampes":      {"lat": 48.43, "lon": 2.16},
}

DATE_START = "2020-01-01"
DATE_END = "2024-12-31"

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 1. OPEN-METEO: Weather + ET₀ + Soil Moisture
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
    "sunshine_duration",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
]


def fetch_open_meteo_single(lat: float, lon: float, start: str, end: str,
                            max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch daily weather data for a single date range with retry logic.
    """
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
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df.rename(columns={"time": "date"}, inplace=True)
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = attempt * 10
            print(f"    ⚠️  Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
            if attempt < max_retries:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily weather data from Open-Meteo Historical API.
    Splits the request year by year to avoid timeouts.
    """
    start_year = int(start[:4])
    end_year = int(end[:4])

    chunks = []
    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01" if year > start_year else start
        y_end = f"{year}-12-31" if year < end_year else end
        print(f"  Fetching Open-Meteo: ({lat}, {lon}) — {year}...")
        chunk = fetch_open_meteo_single(lat, lon, y_start, y_end)
        chunks.append(chunk)
        time.sleep(1)  # Be nice to the API

    return pd.concat(chunks, ignore_index=True)


# ============================================================
# 2. SOILGRIDS: Static Soil Properties
# ============================================================

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

SOIL_PROPERTIES = ["clay", "sand", "silt", "phh2o", "ocd", "bdod", "cec"]
# clay/sand/silt: texture (g/kg)
# phh2o: pH in water (pH*10)
# ocd: organic carbon density (g/dm³)
# bdod: bulk density (cg/cm³)
# cec: cation exchange capacity (mmol(c)/kg)


def fetch_soilgrids(lat: float, lon: float) -> dict:
    """
    Fetch soil properties from ISRIC SoilGrids REST API.
    Free, no API key. Rate limited to 5 requests/minute.
    Returns mean values at 0-30cm depth (relevant for root zone).
    """
    params = {
        "lat": lat,
        "lon": lon,
        "property": SOIL_PROPERTIES,
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
        "value": "mean",
    }

    print(f"  Fetching SoilGrids: ({lat}, {lon})...")
    response = requests.get(SOILGRIDS_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Average across the 3 depth layers (0-30cm)
    soil = {}
    for layer in data.get("properties", {}).get("layers", []):
        prop_name = layer["name"]
        values = []
        for depth in layer.get("depths", []):
            val = depth.get("values", {}).get("mean")
            if val is not None:
                values.append(val)
        if values:
            soil[f"soil_{prop_name}"] = np.mean(values)

    # Convert units to human-readable
    if "soil_phh2o" in soil:
        soil["soil_ph"] = soil.pop("soil_phh2o") / 10  # pH*10 → pH
    if "soil_bdod" in soil:
        soil["soil_bulk_density"] = soil.pop("soil_bdod") / 100  # cg/cm³ → g/cm³
    if "soil_clay" in soil:
        soil["soil_clay_pct"] = soil.pop("soil_clay") / 10  # g/kg → %
    if "soil_sand" in soil:
        soil["soil_sand_pct"] = soil.pop("soil_sand") / 10
    if "soil_silt" in soil:
        soil["soil_silt_pct"] = soil.pop("soil_silt") / 10

    return soil


# ============================================================
# 3. FAO-56 Kc COEFFICIENTS — Winter Wheat (France)
# ============================================================
# Source: FAO Irrigation and Drainage Paper 56, Table 12
# Winter wheat in temperate climate (France/Europe)
#
# Growth stages for winter wheat in Beauce:
#   - Sowing/Initial:  Oct 15 - Dec 31  (dormancy through winter)
#   - Development:     Jan 1 - Mar 31   (tillering, stem elongation)
#   - Mid-season:      Apr 1 - May 31   (heading, flowering, grain fill)
#   - Late season:     Jun 1 - Jul 15   (ripening, harvest)
#   - Fallow:          Jul 16 - Oct 14  (no crop)

FAO_KC_WHEAT = {
    "initial":     0.4,   # Kc_ini: sparse canopy, low transpiration
    "development": 0.75,  # Kc increasing as canopy grows
    "mid_season":  1.15,  # Kc_mid: full canopy, peak water demand
    "late_season": 0.40,  # Kc_end: senescence, drying down
    "fallow":      0.0,   # No crop present
}


def assign_growth_stage(date: pd.Timestamp) -> str:
    """Assign wheat growth stage based on date (Beauce calendar)."""
    month, day = date.month, date.day

    if (month == 10 and day >= 15) or month in [11, 12]:
        return "initial"
    elif month in [1, 2, 3]:
        return "development"
    elif month in [4, 5]:
        return "mid_season"
    elif month == 6 or (month == 7 and day <= 15):
        return "late_season"
    else:
        return "fallow"


def get_kc(stage: str) -> float:
    """Return FAO Kc coefficient for the given growth stage."""
    return FAO_KC_WHEAT.get(stage, 0.0)


# ============================================================
# 4. WATER NEED INDEX CALCULATION
# ============================================================

def compute_water_need_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Water Need Index (target variable) using FAO method.

    Water Need Index = max(0, ETc - effective_rainfall)
    Where:
        ETc = ET₀ × Kc
        effective_rainfall ≈ 0.8 × precipitation (USDA-SCS method, simplified)

    Also adds soil moisture deficit as complementary feature.
    """
    df = df.copy()

    # Growth stage and Kc
    df["growth_stage"] = df["date"].apply(assign_growth_stage)
    df["kc"] = df["growth_stage"].apply(get_kc)

    # Crop evapotranspiration
    df["etc_mm"] = df["et0_fao_evapotranspiration"] * df["kc"]

    # Effective rainfall (simplified USDA method: ~80% of rain is usable)
    df["effective_rainfall_mm"] = df["precipitation_sum"].clip(lower=0) * 0.8

    # Water Need Index (mm/day) — THE TARGET
    df["water_need_index"] = (df["etc_mm"] - df["effective_rainfall_mm"]).clip(lower=0)

    # Binary irrigation flag (useful for classification sub-task)
    df["irrigation_needed"] = (df["water_need_index"] > 0.5).astype(int)

    return df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_collection():
    """Execute the full data collection pipeline."""
    print("=" * 60)
    print("AquaSmart — Data Collection Pipeline")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []

    for location_name, coords in LOCATIONS.items():
        print(f"\n📍 Processing {location_name}...")

        # --- Weather + ET₀ ---
        weather_df = fetch_open_meteo(
            coords["lat"], coords["lon"], DATE_START, DATE_END
        )
        weather_df["location"] = location_name
        weather_df["latitude"] = coords["lat"]
        weather_df["longitude"] = coords["lon"]

        # --- Soil properties (static, one call per location) ---
        try:
            soil_props = fetch_soilgrids(coords["lat"], coords["lon"])
            for key, value in soil_props.items():
                weather_df[key] = value
            print(f"  ✅ SoilGrids: {len(soil_props)} properties fetched")
        except Exception as e:
            print(f"  ⚠️  SoilGrids failed: {e}")
            print(f"     → Continuing without soil data for {location_name}")

        # Respect SoilGrids rate limit (5 req/min)
        time.sleep(13)

        # --- Growth stage + Kc + Water Need Index ---
        weather_df = compute_water_need_index(weather_df)

        all_data.append(weather_df)
        print(f"  ✅ {len(weather_df)} daily records collected")

    # --- Merge all locations ---
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"Total: {len(df)} records across {len(LOCATIONS)} locations")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {len(df.columns)}")

    # --- Save raw data ---
    raw_path = RAW_DIR / "aquasmart_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n💾 Raw data saved to: {raw_path}")

    # --- Quick data quality report ---
    print(f"\n{'=' * 60}")
    print("DATA QUALITY REPORT")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None! ✅")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count / len(df) * 100:.1f}%)")

    print(f"\nTarget variable (water_need_index):")
    print(f"  Mean:   {df['water_need_index'].mean():.2f} mm/day")
    print(f"  Median: {df['water_need_index'].median():.2f} mm/day")
    print(f"  Max:    {df['water_need_index'].max():.2f} mm/day")
    print(f"  Zeros:  {(df['water_need_index'] == 0).sum()} "
          f"({(df['water_need_index'] == 0).mean() * 100:.1f}%)")

    print(f"\nGrowth stages:")
    print(df["growth_stage"].value_counts().to_string())

    return df


if __name__ == "__main__":
    df = run_collection()
    print("\n✅ Collection complete! Next step: run notebooks/01_eda.ipynb")