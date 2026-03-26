"""
AquaSmart — Data Loading Utilities
Handles loading raw datasets and basic validation.
"""

import pandas as pd
import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load a raw dataset from data/raw/.
    
    Parameters
    ----------
    filename : str
        Name of the file (e.g., 'irrigation_data.csv')
    
    Returns
    -------
    pd.DataFrame
        Raw dataframe
    """
    raw_path = get_project_root() / "data" / "raw" / filename
    
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. "
            f"Please place your CSV file in data/raw/"
        )
    
    # Detect file type
    suffix = raw_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(raw_path)
    elif suffix in (".xls", ".xlsx"):
        df = pd.read_excel(raw_path)
    elif suffix == ".json":
        df = pd.read_json(raw_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    print(f"✅ Loaded {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def quick_summary(df: pd.DataFrame) -> dict:
    """
    Generate a quick summary of the dataframe for initial inspection.
    
    Returns
    -------
    dict
        Summary statistics including shape, dtypes, missing values, etc.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return summary


def validate_irrigation_dataset(df: pd.DataFrame) -> list:
    """
    Validate that the dataset contains expected columns for irrigation modeling.
    Returns a list of warnings.
    """
    expected_categories = {
        "weather": ["temperature", "temp", "humidity", "rainfall", "precipitation", "wind"],
        "soil": ["soil_moisture", "soil_type", "ph", "clay", "sand"],
        "crop": ["crop", "crop_type", "growth_stage", "crop_days"],
        "target": ["water_requirement", "irrigation", "water_need", "water_amount"],
    }
    
    warnings = []
    col_lower = [c.lower().replace(" ", "_") for c in df.columns]
    
    for category, keywords in expected_categories.items():
        found = any(
            any(kw in col for kw in keywords)
            for col in col_lower
        )
        if not found:
            warnings.append(
                f"⚠️  No '{category}' variable detected. "
                f"Expected keywords: {keywords}"
            )
        else:
            matched = [
                df.columns[i] for i, col in enumerate(col_lower)
                if any(kw in col for kw in keywords)
            ]
            print(f"✅ {category}: found {matched}")
    
    return warnings
