"""
AquaSmart v4 — Target generation with VARIABLE doses + crop clustering.

v4 fixes:
  - Reads ALL 10 cities from weather_cache.csv (not just Melun)
  - 15 French crops instead of 5
  - Each crop is assigned to one of 4 agronomic clusters
  - Irrigation dose is VARIABLE (function of soil deficit)
  - Adds soil_stock_mm column (useful as feature later)

Output: data/raw/aquasmart_v4_{crop}.csv for each crop
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------
# Crop configuration (15 crops → 4 agronomic clusters)
# ---------------------------------------------------------------
CROPS = {
    # --- Cluster 1: winter cereals ---
    "winter_wheat": {
        "cluster": 1, "ru_max": 120, "sowing_month": 10,
        "kc": {"initial": 0.40, "development": 0.70, "mid_season": 1.15, "late_season": 0.40, "fallow": 0.0},
    },
    "durum_wheat": {
        "cluster": 1, "ru_max": 115, "sowing_month": 10,
        "kc": {"initial": 0.40, "development": 0.70, "mid_season": 1.15, "late_season": 0.40, "fallow": 0.0},
    },
    "winter_barley": {
        "cluster": 1, "ru_max": 110, "sowing_month": 10,
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.15, "late_season": 0.25, "fallow": 0.0},
    },
    "oats": {
        "cluster": 1, "ru_max": 105, "sowing_month": 10,
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.15, "late_season": 0.25, "fallow": 0.0},
    },
    "triticale": {
        "cluster": 1, "ru_max": 115, "sowing_month": 10,
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15, "late_season": 0.35, "fallow": 0.0},
    },

    # --- Cluster 2: summer deep-rooted crops ---
    "corn": {
        "cluster": 2, "ru_max": 150, "sowing_month": 4,
        "kc": {"initial": 0.30, "development": 0.70, "mid_season": 1.20, "late_season": 0.60, "fallow": 0.0},
    },
    "sunflower": {
        "cluster": 2, "ru_max": 140, "sowing_month": 4,
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15, "late_season": 0.35, "fallow": 0.0},
    },
    "sorghum": {
        "cluster": 2, "ru_max": 135, "sowing_month": 5,
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.10, "late_season": 0.55, "fallow": 0.0},
    },
    "soybean": {
        "cluster": 2, "ru_max": 130, "sowing_month": 5,
        "kc": {"initial": 0.40, "development": 0.80, "mid_season": 1.15, "late_season": 0.50, "fallow": 0.0},
    },

    # --- Cluster 3: winter oilseeds / legumes ---
    "rapeseed": {
        "cluster": 3, "ru_max": 130, "sowing_month": 9,
        "kc": {"initial": 0.35, "development": 0.70, "mid_season": 1.15, "late_season": 0.35, "fallow": 0.0},
    },
    "winter_pea": {
        "cluster": 3, "ru_max": 120, "sowing_month": 11,
        "kc": {"initial": 0.50, "development": 0.80, "mid_season": 1.15, "late_season": 0.30, "fallow": 0.0},
    },
    "faba_bean": {
        "cluster": 3, "ru_max": 125, "sowing_month": 11,
        "kc": {"initial": 0.50, "development": 0.80, "mid_season": 1.15, "late_season": 0.30, "fallow": 0.0},
    },

    # --- Cluster 4: shallow-rooted row crops ---
    "potato": {
        "cluster": 4, "ru_max": 70, "sowing_month": 4,
        "kc": {"initial": 0.50, "development": 0.75, "mid_season": 1.15, "late_season": 0.75, "fallow": 0.0},
    },
    "sugar_beet": {
        "cluster": 4, "ru_max": 85, "sowing_month": 3,
        "kc": {"initial": 0.35, "development": 0.75, "mid_season": 1.20, "late_season": 0.70, "fallow": 0.0},
    },
    "field_vegetables": {
        "cluster": 4, "ru_max": 65, "sowing_month": 4,
        "kc": {"initial": 0.50, "development": 0.75, "mid_season": 1.05, "late_season": 0.90, "fallow": 0.0},
    },
}

# Simulation parameters
TRIGGER_FRAC = 0.50
TARGET_FRAC_LOW = 0.70
TARGET_FRAC_HIGH = 0.90
DOSE_MIN = 15.0
DOSE_MAX = 40.0
EFF_RAIN = 0.8

WEATHER_COLS = [
    "location", "date", "temperature_2m_mean", "relative_humidity_2m_mean",
    "precipitation_sum", "et0_fao_evapotranspiration", "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
]


def assign_stage(date: pd.Timestamp, sowing_month: int) -> str:
    """Map a date to a growth stage based on sowing month (8-month cycle)."""
    months_since_sowing = (date.month - sowing_month) % 12
    if months_since_sowing < 2:
        return "initial"
    elif months_since_sowing < 4:
        return "development"
    elif months_since_sowing < 6:
        return "mid_season"
    elif months_since_sowing < 8:
        return "late_season"
    return "fallow"


def simulate_location(df_loc: pd.DataFrame, crop_cfg: dict) -> pd.DataFrame:
    """Run daily water balance simulation for one location."""
    df = df_loc.sort_values("date").reset_index(drop=True).copy()
    ru_max = crop_cfg["ru_max"]
    kc_table = crop_cfg["kc"]
    sowing = crop_cfg["sowing_month"]

    df["growth_stage"] = df["date"].apply(lambda d: assign_stage(d, sowing))
    df["kc"] = df["growth_stage"].map(kc_table).fillna(0.0)
    df["etc_mm"] = df["et0_fao_evapotranspiration"] * df["kc"]
    df["eff_rain_mm"] = df["precipitation_sum"].clip(lower=0) * EFF_RAIN

    n = len(df)
    stock = np.zeros(n)
    irrigation = np.zeros(n)
    stock_prev = ru_max * 0.5

    etc_arr = df["etc_mm"].values
    rain_arr = df["eff_rain_mm"].values

    for i in range(n):
        s = stock_prev + rain_arr[i] - etc_arr[i]
        s = max(0.0, min(ru_max, s))

        dose = 0.0
        if s < TRIGGER_FRAC * ru_max:
            lo = max(0, i - 6)
            etc_7d = etc_arr[lo:i + 1].sum()
            pressure = float(np.clip(etc_7d / 35.0, 0.0, 1.0))
            target_level = (TARGET_FRAC_LOW + (TARGET_FRAC_HIGH - TARGET_FRAC_LOW) * pressure) * ru_max
            raw_need = target_level - s
            dose = float(np.clip(raw_need, DOSE_MIN, DOSE_MAX))
            s = min(ru_max, s + dose)

        stock[i] = s
        irrigation[i] = dose
        stock_prev = s

    df["soil_stock_mm"] = stock
    df["irrigation_applied_mm"] = irrigation
    df["irrigation_needed"] = (irrigation > 0).astype(int)
    return df


def load_weather(raw_dir: Path) -> pd.DataFrame:
    """
    Load the shared weather dataset.
    Priority: weather_cache.csv (all 10 cities) > any aquasmart_v2_*.csv
    """
    cache = raw_dir / "weather_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        print(f"  ✓ Loaded weather_cache.csv ({len(df)} rows, "
              f"{df['location'].nunique()} cities)")
        return df[WEATHER_COLS].copy()

    # Fallback: read any v2 file that already contains all 10 cities
    v2_files = sorted(raw_dir.glob("aquasmart_v2_*.csv"))
    if not v2_files:
        raise FileNotFoundError("No weather source found in data/raw/")

    df = pd.read_csv(v2_files[0], parse_dates=["date"])
    cities = df["location"].nunique() if "location" in df.columns else 1
    if cities < 2:
        raise ValueError(
            f"Weather source {v2_files[0].name} only contains {cities} city. "
            "Please re-run collect_data_v2.py first."
        )
    print(f"  ✓ Loaded {v2_files[0].name} ({len(df)} rows, {cities} cities)")
    return df[WEATHER_COLS].copy()


def process_crop(crop: str, cfg: dict, weather: pd.DataFrame, raw_dir: Path) -> None:
    print(f"\n🌱 {crop}  (cluster {cfg['cluster']}, RU_max={cfg['ru_max']}mm)")

    out = (weather.groupby("location", group_keys=False, sort=False)
                  .apply(lambda g: simulate_location(g, cfg), include_groups=True)
                  .reset_index(drop=True))

    out["crop"] = crop
    out["cluster"] = cfg["cluster"]
    out["ru_max"] = cfg["ru_max"]

    irr_mask = out["irrigation_needed"] == 1
    irr_days = int(irr_mask.sum())
    rate = irr_days / len(out) * 100
    doses = out.loc[irr_mask, "irrigation_applied_mm"]

    print(f"  Irrigation days: {irr_days} / {len(out)} ({rate:.1f}%)")
    if len(doses) > 0:
        print(f"  Doses: min={doses.min():.1f}  median={doses.median():.1f}  "
              f"mean={doses.mean():.1f}  max={doses.max():.1f}  std={doses.std():.2f}")

    outfile = raw_dir / f"aquasmart_v4_{crop}.csv"
    out.to_csv(outfile, index=False)
    print(f"  ✅ {outfile.name}")


def main():
    print("=" * 60)
    print("AquaSmart v4 — Target generation (variable doses, 15 crops)")
    print("=" * 60)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load shared weather ONCE (all 10 cities)
    print("\n📡 Loading shared weather data...")
    weather = load_weather(raw_dir)

    # Apply each crop's coefficients to the same weather
    for crop, cfg in CROPS.items():
        process_crop(crop, cfg, weather, raw_dir)

    print("\n" + "=" * 60)
    print("DONE. Next: python src/data/preprocess_v4.py")
    print("=" * 60)


if __name__ == "__main__":
    main()