"""
AquaSmart — Multi-Crop Preprocessing (v2)
==========================================
Processes each crop's raw dataset into train/test splits.
Reuses the same feature selection and scaling approach as v1,
applied independently per crop.

Usage:
    python src/data/preprocess_v2.py

Input:  data/raw/aquasmart_v2_{crop}.csv
Output: data/processed/{crop}/train.csv, test.csv, train_irrigation.csv, test_irrigation.csv
        models/{crop}/scaler.joblib
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Same 7 features as v1 (proven to work, no leakage)
FEATURE_COLS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "growth_stage_encoded",
]

TARGET_CLS = "irrigation_needed"
TARGET_REG = "water_need_index"

CROPS = ["winter_wheat", "corn", "barley", "rapeseed", "sunflower"]


def preprocess_crop(crop_key: str):
    """Preprocess a single crop dataset."""
    print(f"\n  🌱 Processing {crop_key}...")

    # Load raw data
    raw_path = RAW_DIR / f"aquasmart_v2_{crop_key}.csv"
    if not raw_path.exists():
        print(f"    ⚠️  File not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, parse_dates=["date"])
    print(f"    Loaded: {len(df)} rows")

    # Temporal split: 2020-2023 train, 2024 test
    train = df[df["date"].dt.year <= 2023].copy()
    test = df[df["date"].dt.year == 2024].copy()
    print(f"    Train: {len(train)} rows (2020-2023)")
    print(f"    Test:  {len(test)} rows (2024)")

    # Irrigation-only subsets (for regression stage)
    train_irr = train[train[TARGET_CLS] == 1].copy()
    test_irr = test[test[TARGET_CLS] == 1].copy()
    print(f"    Train irrigation days: {len(train_irr)} ({len(train_irr)/len(train)*100:.1f}%)")
    print(f"    Test irrigation days:  {len(test_irr)} ({len(test_irr)/len(test)*100:.1f}%)")

    # Scale features
    scaler = StandardScaler()
    train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
    test[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

    # Apply same scaling to irrigation subsets
    train_irr[FEATURE_COLS] = scaler.transform(train_irr[FEATURE_COLS])
    test_irr[FEATURE_COLS] = scaler.transform(test_irr[FEATURE_COLS])

    # Create output directories
    crop_processed_dir = PROCESSED_DIR / crop_key
    crop_model_dir = MODEL_DIR / crop_key
    crop_processed_dir.mkdir(parents=True, exist_ok=True)
    crop_model_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets (only features + targets)
    cols_to_save = FEATURE_COLS + [TARGET_CLS, TARGET_REG]

    train[cols_to_save].to_csv(crop_processed_dir / "train.csv", index=False)
    test[cols_to_save].to_csv(crop_processed_dir / "test.csv", index=False)
    train_irr[cols_to_save].to_csv(crop_processed_dir / "train_irrigation.csv", index=False)
    test_irr[cols_to_save].to_csv(crop_processed_dir / "test_irrigation.csv", index=False)

    # Save scaler
    joblib.dump(scaler, crop_model_dir / "scaler.joblib")

    print(f"    ✅ Saved to {crop_processed_dir}/")
    print(f"    ✅ Scaler saved to {crop_model_dir}/scaler.joblib")


def main():
    print("=" * 60)
    print("AquaSmart v2 — Multi-Crop Preprocessing")
    print("=" * 60)

    for crop_key in CROPS:
        preprocess_crop(crop_key)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Processed {len(CROPS)} crops")
    print(f"\n  Next step: python src/models/train_v2.py")


if __name__ == "__main__":
    main()
