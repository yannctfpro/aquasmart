"""
AquaSmart — Water Balance Baseline (Checkbook Method)
======================================================
Implements the FAO-56 / USDA-NRCS "checkbook method" as a realistic
baseline for irrigation decisions. Replaces the naive fixed-schedule
baseline that produced inflated ML improvement metrics.

Method:
    Stock(t) = min(RU_max, max(0, Stock(t-1) + Effective_Rain - ETc + Irrigation(t-1)))
    Irrigate if Stock(t) < threshold × RU_max

This baseline represents what a professional farmer using a field
notebook or spreadsheet would actually do. It integrates:
  - Soil (via plant-available water capacity RU_max)
  - Growth stage (via Kc and variable threshold)
  - Weather (via ET₀ and precipitation)
  - Irrigation system (via realistic max dose)

Usage:
    python src/baselines/water_balance.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

# ============================================================
# CROP-SPECIFIC PARAMETERS
# ============================================================
# RU_max: plant-available water in root zone (mm)
# Sources: FAO-56 Table 22, USDA-NRCS, Arvalis references

CROP_PARAMS = {
    "winter_wheat": {"ru_max": 130, "max_dose": 30, "root_depth_m": 1.1},
    "corn":         {"ru_max": 150, "max_dose": 30, "root_depth_m": 1.3},
    "barley":       {"ru_max": 125, "max_dose": 30, "root_depth_m": 1.0},
    "rapeseed":     {"ru_max": 140, "max_dose": 30, "root_depth_m": 1.3},
    "sunflower":    {"ru_max": 170, "max_dose": 30, "root_depth_m": 1.8},
}

# Depletion threshold by growth stage
# Stricter during sensitive reproductive stages (flowering/grain fill)
STAGE_THRESHOLDS = {
    "fallow": 1.00,
    "initial": 0.50,
    "development": 0.55,
    "mid_season": 0.65,      # CRITICAL — flowering
    "late_season": 0.50,
}

EFFECTIVE_RAIN_COEF = 0.8    # USDA-SCS simplified

# Kc coefficients (FAO-56 Table 12)
CROP_KC = {
    "winter_wheat": {"fallow": 0.0, "initial": 0.40, "development": 0.75, "mid_season": 1.15, "late_season": 0.40},
    "corn":         {"fallow": 0.0, "initial": 0.30, "development": 0.75, "mid_season": 1.20, "late_season": 0.60},
    "barley":       {"fallow": 0.0, "initial": 0.40, "development": 0.75, "mid_season": 1.15, "late_season": 0.25},
    "rapeseed":     {"fallow": 0.0, "initial": 0.35, "development": 0.75, "mid_season": 1.15, "late_season": 0.35},
    "sunflower":    {"fallow": 0.0, "initial": 0.35, "development": 0.75, "mid_season": 1.15, "late_season": 0.35},
}

STAGE_NAMES = {0: "fallow", 1: "initial", 2: "development", 3: "mid_season", 4: "late_season"}


# ============================================================
# WATER BALANCE SIMULATION
# ============================================================

def simulate_water_balance(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Simulate daily water balance and irrigation decisions for one location × one crop."""
    params = CROP_PARAMS[crop]
    ru_max = params["ru_max"]
    max_dose = params["max_dose"]
    kc_table = CROP_KC[crop]

    df = df.sort_values("date").reset_index(drop=True).copy()

    n = len(df)
    stock = np.zeros(n)
    etc_arr = np.zeros(n)
    irrigation = np.zeros(n)
    decision = np.zeros(n, dtype=int)

    # Initialize: full soil reserve at start of season
    stock_prev = ru_max

    for t in range(n):
        stage_code = int(df.loc[t, "growth_stage_encoded"])
        stage_name = STAGE_NAMES[stage_code]
        kc = kc_table[stage_name]
        threshold_frac = STAGE_THRESHOLDS[stage_name]

        etc_t = df.loc[t, "et0_fao_evapotranspiration"] * kc
        eff_rain = EFFECTIVE_RAIN_COEF * max(0, df.loc[t, "precipitation_sum"])

        # Update stock BEFORE today's decision
        stock_today = stock_prev + eff_rain - etc_t
        stock_today = min(ru_max, max(0, stock_today))

        threshold_mm = threshold_frac * ru_max

        if stage_name == "fallow":
            irr_t = 0
            dec_t = 0
        elif stock_today < threshold_mm:
            needed = ru_max - stock_today
            irr_t = min(max_dose, needed)
            dec_t = 1
        else:
            irr_t = 0
            dec_t = 0

        stock[t] = stock_today
        etc_arr[t] = etc_t
        irrigation[t] = irr_t
        decision[t] = dec_t

        # Apply irrigation for next iteration
        stock_prev = min(ru_max, stock_today + irr_t)

    df["baseline_stock_mm"] = stock
    df["baseline_etc_mm"] = etc_arr
    df["baseline_irrigation_mm"] = irrigation
    df["baseline_decision"] = decision

    return df


def compute_baseline_metrics(df: pd.DataFrame) -> dict:
    """Compute metrics for the baseline against ground truth."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    y_true_cls = df["irrigation_needed"].values
    y_pred_cls = df["baseline_decision"].values
    y_true_reg = df["water_need_index"].values
    y_pred_reg = df["baseline_irrigation_mm"].values

    return {
        "cls_accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "cls_precision": float(precision_score(y_true_cls, y_pred_cls, zero_division=0)),
        "cls_recall": float(recall_score(y_true_cls, y_pred_cls, zero_division=0)),
        "cls_f1": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "reg_rmse": float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))),
        "reg_mae": float(mean_absolute_error(y_true_reg, y_pred_reg)),
        "reg_r2": float(r2_score(y_true_reg, y_pred_reg)) if len(y_true_reg) > 1 else 0.0,
    }


def run_baseline_for_crop(crop: str) -> dict:
    """Run the water balance baseline for a crop and compare with ML."""
    print(f"\n{'='*60}")
    print(f"🌱 Water Balance Baseline: {crop}")
    print(f"{'='*60}")

    raw_path = RAW_DIR / f"aquasmart_v2_{crop}.csv"
    if not raw_path.exists():
        print(f"  ⚠️  Raw file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path, parse_dates=["date"])
    print(f"  Loaded: {len(df)} rows × {df['location'].nunique()} locations")

    # Simulate water balance per location
    dfs = []
    for loc in df["location"].unique():
        sub = df[df["location"] == loc].copy()
        sub = simulate_water_balance(sub, crop)
        dfs.append(sub)
    df_sim = pd.concat(dfs, ignore_index=True)

    # Evaluate on 2024 only (test year, matching ML evaluation)
    df_test = df_sim[df_sim["date"].dt.year == 2024].copy()
    print(f"  Test (2024): {len(df_test)} rows")

    baseline_metrics = compute_baseline_metrics(df_test)

    print(f"\n  Classification:")
    print(f"    Accuracy:  {baseline_metrics['cls_accuracy']:.3f}")
    print(f"    Precision: {baseline_metrics['cls_precision']:.3f}")
    print(f"    Recall:    {baseline_metrics['cls_recall']:.3f}")
    print(f"    F1:        {baseline_metrics['cls_f1']:.3f}")
    print(f"  Regression:")
    print(f"    RMSE: {baseline_metrics['reg_rmse']:.3f} mm/day")
    print(f"    MAE:  {baseline_metrics['reg_mae']:.3f} mm/day")
    print(f"    R²:   {baseline_metrics['reg_r2']:.3f}")

    # Compare with ML
    ml_results_path = MODEL_DIR / crop / "results.json"
    if ml_results_path.exists():
        with open(ml_results_path) as f:
            ml_results = json.load(f)

        best_reg = ml_results["best_regressor"]
        if best_reg != "None" and best_reg in ml_results["regression"]:
            ml_mae = ml_results["regression"][best_reg]["mae"]
            ml_rmse = ml_results["regression"][best_reg]["rmse"]
            ml_f1 = ml_results["classification"][ml_results["best_classifier"]]["f1"]

            mae_imp = (1 - ml_mae / baseline_metrics["reg_mae"]) * 100 if baseline_metrics["reg_mae"] > 0 else 0
            rmse_imp = (1 - ml_rmse / baseline_metrics["reg_rmse"]) * 100 if baseline_metrics["reg_rmse"] > 0 else 0
            f1_imp = (ml_f1 - baseline_metrics["cls_f1"]) * 100

            print(f"\n  📊 ML vs Water Balance Baseline:")
            print(f"    F1:   baseline={baseline_metrics['cls_f1']:.3f}  →  ML={ml_f1:.3f}  ({f1_imp:+.1f} pts)")
            print(f"    MAE:  baseline={baseline_metrics['reg_mae']:.3f}  →  ML={ml_mae:.3f}  ({mae_imp:+.1f}%)")
            print(f"    RMSE: baseline={baseline_metrics['reg_rmse']:.3f}  →  ML={ml_rmse:.3f}  ({rmse_imp:+.1f}%)")

            return {
                "crop": crop,
                "baseline": baseline_metrics,
                "ml": {
                    "classifier": ml_results["best_classifier"],
                    "regressor": best_reg,
                    "f1": ml_f1,
                    "mae": ml_mae,
                    "rmse": ml_rmse,
                },
                "improvement": {
                    "f1_points": f1_imp,
                    "mae_percent": mae_imp,
                    "rmse_percent": rmse_imp,
                },
            }

    return {"crop": crop, "baseline": baseline_metrics, "ml": None}


def main():
    print("=" * 60)
    print("AquaSmart — Water Balance Baseline (Checkbook Method)")
    print("=" * 60)

    all_results = []
    for crop in CROP_PARAMS.keys():
        result = run_baseline_for_crop(crop)
        if result:
            all_results.append(result)

    # Summary table
    print("\n" + "=" * 88)
    print("FINAL COMPARISON — ML vs Water Balance Baseline (test 2024)")
    print("=" * 88)
    print(f"{'Crop':<16s} {'Baseline F1':>12s} {'ML F1':>8s} {'Baseline MAE':>14s} {'ML MAE':>10s} {'MAE gain':>10s}")
    print("-" * 88)

    for r in all_results:
        if r.get("ml"):
            print(f"{r['crop']:<16s} "
                  f"{r['baseline']['cls_f1']:>12.3f} "
                  f"{r['ml']['f1']:>8.3f} "
                  f"{r['baseline']['reg_mae']:>14.3f} "
                  f"{r['ml']['mae']:>10.3f} "
                  f"{r['improvement']['mae_percent']:>9.1f}%")

    output_path = PROJECT_ROOT / "reports" / "water_balance_baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ Full results saved to: {output_path}")


if __name__ == "__main__":
    main()