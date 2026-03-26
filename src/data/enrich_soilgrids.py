"""
AquaSmart — SoilGrids Enrichment Script
========================================
Fetches static soil properties from ISRIC SoilGrids REST API
and adds them to the existing aquasmart_raw.csv.

Run this script after collect_data.py if SoilGrids was unavailable.

Usage:
    python src/data/enrich_soilgrids.py
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "aquasmart_raw.csv"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["clay", "sand", "silt", "phh2o", "ocd", "bdod", "cec"]

LOCATIONS = {
    "Chartres":    {"lat": 48.45, "lon": 1.50},
    "Orleans":     {"lat": 47.90, "lon": 1.90},
    "Pithiviers":  {"lat": 48.17, "lon": 2.25},
    "Chateaudun":  {"lat": 48.07, "lon": 1.34},
    "Etampes":     {"lat": 48.43, "lon": 2.16},
}


def fetch_soilgrids(lat: float, lon: float, max_retries: int = 3) -> dict:
    """Fetch soil properties with retry logic."""
    params = {
        "lat": lat,
        "lon": lon,
        "property": SOIL_PROPERTIES,
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
        "value": "mean",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(SOILGRIDS_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

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

            # Convert units
            if "soil_phh2o" in soil:
                soil["soil_ph"] = round(soil.pop("soil_phh2o") / 10, 2)
            if "soil_bdod" in soil:
                soil["soil_bulk_density"] = round(soil.pop("soil_bdod") / 100, 3)
            if "soil_clay" in soil:
                soil["soil_clay_pct"] = round(soil.pop("soil_clay") / 10, 1)
            if "soil_sand" in soil:
                soil["soil_sand_pct"] = round(soil.pop("soil_sand") / 10, 1)
            if "soil_silt" in soil:
                soil["soil_silt_pct"] = round(soil.pop("soil_silt") / 10, 1)
            if "soil_ocd" in soil:
                soil["soil_organic_carbon"] = round(soil.pop("soil_ocd"), 1)
            if "soil_cec" in soil:
                soil["soil_cec"] = round(soil["soil_cec"], 1)

            return soil

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = attempt * 15
            print(f"    Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
            if attempt < max_retries:
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                wait = attempt * 20
                print(f"    503 Service Unavailable (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            else:
                raise


def run_enrichment():
    print("=" * 60)
    print("AquaSmart — SoilGrids Enrichment")
    print("=" * 60)

    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found. Run collect_data.py first.")
        return

    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH.name}")

    # Check if soil data already exists
    soil_cols = [c for c in df.columns if c.startswith("soil_") and c not in [
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_moisture_28_to_100cm_mean"
    ]]
    if soil_cols:
        print(f"Soil columns already present: {soil_cols}")
        print("Overwriting with fresh SoilGrids data...")
        df.drop(columns=soil_cols, inplace=True)

    # Fetch soil data for each location
    soil_data = {}
    for name, coords in LOCATIONS.items():
        print(f"\nFetching SoilGrids for {name} ({coords['lat']}, {coords['lon']})...")
        try:
            soil = fetch_soilgrids(coords["lat"], coords["lon"])
            soil_data[name] = soil
            print(f"  OK: {soil}")
            time.sleep(13)  # Rate limit: 5 req/min
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  SoilGrids is still unavailable. Try again later.")
            return

    # Merge into dataframe
    for name, soil in soil_data.items():
        mask = df["location"] == name
        for key, value in soil.items():
            df.loc[mask, key] = value

    # Save
    df.to_csv(RAW_PATH, index=False)
    print(f"\nUpdated {RAW_PATH.name} with soil properties")

    # Summary
    new_cols = [c for c in df.columns if c.startswith("soil_") and c not in [
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_moisture_28_to_100cm_mean"
    ]]
    print(f"\nNew soil columns added: {new_cols}")
    for col in new_cols:
        print(f"  {col}: {df[col].unique()}")

    print("\nDone!")


if __name__ == "__main__":
    run_enrichment()
