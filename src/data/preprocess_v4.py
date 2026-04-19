"""
AquaSmart v4 — Preprocessing with geographic split.

Split strategy (single, clean):
  - Train: 8 cities × all years 2020-2024
  - Test:  2 cities (Montpellier + Rennes) × all years 2020-2024

This evaluates spatial generalization: can the model transfer to new farms
in climates (Mediterranean dry + Atlantic wet) it has never seen?

Output:
  data/processed/cluster_{1..4}/{data.npz, meta_test.csv, scaler.pkl}
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

TEST_CITIES = {"Montpellier", "Rennes"}

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

ALL_FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES
META_COLS = ["location", "date", "crop", "ru_max", "etc_mm", "precipitation_sum"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling / lag features. Applied per (location, crop)."""
    df = df.sort_values(["location", "crop", "date"]).reset_index(drop=True)
    groups = df.groupby(["location", "crop"], sort=False)

    df["soil_stock_mm_prev"] = groups["soil_stock_mm"].shift(1).fillna(0)
    df["rainfall_cumsum_7d"] = groups["precipitation_sum"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    df["etc_cumsum_7d"] = groups["etc_mm"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    df["water_balance_7d"] = df["rainfall_cumsum_7d"] - df["etc_cumsum_7d"]
    df["irrigation_cumsum_7d"] = groups["irrigation_applied_mm"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
    df["irrigation_cumsum_14d"] = groups["irrigation_applied_mm"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).sum()).fillna(0)

    def _days_since(s: pd.Series) -> pd.Series:
        days = np.zeros(len(s), dtype=float)
        counter = 999.0
        prev = s.shift(1).fillna(0).to_numpy()
        for i, v in enumerate(prev):
            counter = 0.0 if v > 0 else counter + 1.0
            days[i] = min(counter, 30.0)
        return pd.Series(days, index=s.index)

    df["days_since_last_irrigation"] = groups["irrigation_applied_mm"].transform(_days_since)
    return df


def process_cluster(cluster_id: int, raw_dir: Path, proc_dir: Path) -> None:
    frames = []
    for f in sorted(raw_dir.glob("aquasmart_v4_*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        if "cluster" not in df.columns or df["cluster"].iloc[0] != cluster_id:
            continue
        frames.append(df)

    if not frames:
        print(f"\n⚠ Cluster {cluster_id}: no crops found, skipping")
        return

    big = pd.concat(frames, ignore_index=True)
    crops = sorted(big["crop"].unique())
    cities = sorted(big["location"].unique()) if "location" in big.columns else []
    print(f"\n📦 Cluster {cluster_id}: {len(crops)} crops, {len(cities)} cities")
    print(f"   Crops: {crops}")
    print(f"   Total rows: {len(big)}")

    big = add_temporal_features(big)
    big = big.reset_index(drop=True)

    missing = [c for c in ALL_FEATURES if c not in big.columns]
    if missing:
        print(f"   ⚠ Missing features: {missing}")
        return

    # --- Geographic split ---
    test_mask = big["location"].isin(TEST_CITIES)
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        print(f"   ⚠ No test cities ({TEST_CITIES}) found in data, skipping")
        return

    train_cities = sorted(big.loc[train_mask, "location"].unique())
    test_cities = sorted(big.loc[test_mask, "location"].unique())
    print(f"   Train cities ({len(train_cities)}): {train_cities}")
    print(f"   Test cities  ({len(test_cities)}): {test_cities}")

    X_train = big.loc[train_mask, ALL_FEATURES].to_numpy()
    X_test = big.loc[test_mask, ALL_FEATURES].to_numpy()
    y_tr_dec = big.loc[train_mask, "irrigation_needed"].to_numpy()
    y_te_dec = big.loc[test_mask, "irrigation_needed"].to_numpy()
    y_tr_amt = big.loc[train_mask, "irrigation_applied_mm"].to_numpy()
    y_te_amt = big.loc[test_mask, "irrigation_applied_mm"].to_numpy()
    meta_test = big.loc[test_mask, META_COLS].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    out_dir = proc_dir / f"cluster_{cluster_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "data.npz",
        X_train=X_train_s, X_test=X_test_s,
        y_train_decision=y_tr_dec, y_test_decision=y_te_dec,
        y_train_amount=y_tr_amt, y_test_amount=y_te_amt,
        feature_names=np.array(ALL_FEATURES),
    )
    meta_test.to_csv(out_dir / "meta_test.csv", index=False)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    print(f"   Train: {len(X_train):>6} rows ({int(y_tr_dec.sum())} irrigation days, "
          f"{y_tr_dec.mean()*100:.1f}%)")
    print(f"   Test:  {len(X_test):>6} rows ({int(y_te_dec.sum())} irrigation days, "
          f"{y_te_dec.mean()*100:.1f}%)")
    print(f"   ✅ Saved to {out_dir}")


def main():
    print("=" * 60)
    print("AquaSmart v4 — Preprocessing (geographic split)")
    print("=" * 60)
    print(f"Features: {len(ALL_FEATURES)} total")
    print(f"  Static:   {STATIC_FEATURES}")
    print(f"  Temporal: {TEMPORAL_FEATURES}")
    print(f"Split: train = 8 cities × 5 years  |  test = {sorted(TEST_CITIES)} × 5 years")

    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id in [1, 2, 3, 4]:
        process_cluster(cluster_id, raw_dir, proc_dir)

    print("\n" + "=" * 60)
    print("DONE. Next: python src/models/train_v4.py")
    print("=" * 60)


if __name__ == "__main__":
    main()