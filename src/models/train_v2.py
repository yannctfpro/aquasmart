"""
AquaSmart — Multi-Crop Training Pipeline (v2)
===============================================
Trains independent classifier + regressor for each crop.
Same 2-stage approach as v1, applied per crop.

Usage:
    python src/models/train_v2.py

Input:  data/processed/{crop}/train.csv, test.csv, etc.
Output: models/{crop}/classifier.joblib, regressor.joblib, results.json
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "wind_speed_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "growth_stage_encoded",
]

CROPS = ["winter_wheat", "corn", "barley", "rapeseed", "sunflower"]


def train_crop(crop_key: str):
    """Train models for a single crop."""
    print(f"\n{'='*60}")
    print(f"🌱 Training: {crop_key}")
    print(f"{'='*60}")

    crop_dir = PROCESSED_DIR / crop_key
    model_dir = MODEL_DIR / crop_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train = pd.read_csv(crop_dir / "train.csv")
    test = pd.read_csv(crop_dir / "test.csv")
    train_irr = pd.read_csv(crop_dir / "train_irrigation.csv")
    test_irr = pd.read_csv(crop_dir / "test_irrigation.csv")

    X_train = train[FEATURE_COLS]
    y_train_cls = train["irrigation_needed"]
    X_test = test[FEATURE_COLS]
    y_test_cls = test["irrigation_needed"]

    X_train_irr = train_irr[FEATURE_COLS]
    y_train_irr = train_irr["water_need_index"]
    X_test_irr = test_irr[FEATURE_COLS]
    y_test_irr = test_irr["water_need_index"]

    print(f"  Train: {len(train)} | Test: {len(test)}")
    print(f"  Train irrigation: {len(train_irr)} | Test irrigation: {len(test_irr)}")

    # ---- Baseline ----
    baseline_cls = np.ones_like(y_test_cls)
    mean_irrigation = y_train_irr.mean() if len(y_train_irr) > 0 else 1.65
    baseline_reg = np.where(test["water_need_index"] > 0, mean_irrigation, 0)

    baseline = {
        "cls_accuracy": accuracy_score(y_test_cls, baseline_cls),
        "cls_precision": precision_score(y_test_cls, baseline_cls, zero_division=0),
        "cls_recall": recall_score(y_test_cls, baseline_cls, zero_division=0),
        "cls_f1": f1_score(y_test_cls, baseline_cls, zero_division=0),
        "reg_rmse": np.sqrt(mean_squared_error(test["water_need_index"], baseline_reg)),
        "reg_mae": mean_absolute_error(test["water_need_index"], baseline_reg),
        "reg_r2": r2_score(test["water_need_index"], baseline_reg) if len(test) > 1 else 0,
    }
    print(f"\n  Baseline F1: {baseline['cls_f1']:.3f} | MAE: {baseline['reg_mae']:.3f}")

    # ---- Stage 1: Classification ----
    print(f"\n  --- Stage 1: Classification ---")
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    cls_results = {}
    best_f1 = 0
    best_cls_model = None
    best_cls_name = None

    for name, model in classifiers.items():
        model.fit(X_train, y_train_cls)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test_cls, y_pred, zero_division=0)
        cls_results[name] = {
            "accuracy": accuracy_score(y_test_cls, y_pred),
            "precision": precision_score(y_test_cls, y_pred, zero_division=0),
            "recall": recall_score(y_test_cls, y_pred, zero_division=0),
            "f1": f1,
        }
        print(f"    {name}: F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_cls_model = model
            best_cls_name = name

    print(f"  ✅ Best classifier: {best_cls_name} (F1={best_f1:.3f})")
    joblib.dump(best_cls_model, model_dir / "classifier.joblib")

    # ---- Stage 2: Regression ----
    print(f"\n  --- Stage 2: Regression ---")

    if len(X_train_irr) < 10 or len(X_test_irr) < 5:
        print(f"  ⚠️  Not enough irrigation data for regression. Skipping.")
        reg_results = {}
        best_reg_name = "None"
        # Save a dummy result
        results = {
            "crop": crop_key,
            "baseline": baseline,
            "classification": cls_results,
            "regression": {},
            "best_classifier": best_cls_name,
            "best_regressor": "None",
        }
        with open(model_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return

    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    reg_results = {}
    best_rmse = float("inf")
    best_reg_model = None
    best_reg_name = None

    for name, model in regressors.items():
        model.fit(X_train_irr, y_train_irr)
        y_pred = model.predict(X_test_irr)
        rmse = np.sqrt(mean_squared_error(y_test_irr, y_pred))
        reg_results[name] = {
            "rmse": rmse,
            "mae": mean_absolute_error(y_test_irr, y_pred),
            "r2": r2_score(y_test_irr, y_pred),
        }
        print(f"    {name}: RMSE={rmse:.3f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_reg_model = model
            best_reg_name = name

    print(f"  ✅ Best regressor: {best_reg_name} (RMSE={best_rmse:.3f})")
    joblib.dump(best_reg_model, model_dir / "regressor.joblib")

    # ---- Improvement over baseline ----
    if baseline["reg_mae"] > 0:
        ai_mae = reg_results[best_reg_name]["mae"]
        improvement = (1 - ai_mae / baseline["reg_mae"]) * 100
        print(f"\n  📊 MAE improvement over baseline: {improvement:.1f}%")

    # ---- Save results ----
    results = {
        "crop": crop_key,
        "baseline": baseline,
        "classification": cls_results,
        "regression": reg_results,
        "best_classifier": best_cls_name,
        "best_regressor": best_reg_name,
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✅ Results saved to {model_dir}/results.json")


def main():
    print("=" * 60)
    print("AquaSmart v2 — Multi-Crop Training Pipeline")
    print("=" * 60)

    for crop_key in CROPS:
        train_crop(crop_key)

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — ALL CROPS")
    print("=" * 60)

    for crop_key in CROPS:
        results_path = MODEL_DIR / crop_key / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                r = json.load(f)
            best_cls = r["best_classifier"]
            cls_f1 = r["classification"][best_cls]["f1"]
            best_reg = r["best_regressor"]
            if best_reg != "None" and best_reg in r["regression"]:
                reg_rmse = r["regression"][best_reg]["rmse"]
                reg_mae = r["regression"][best_reg]["mae"]
                baseline_mae = r["baseline"]["reg_mae"]
                improvement = (1 - reg_mae / baseline_mae) * 100 if baseline_mae > 0 else 0
                print(f"  {crop_key:<20s} | Cls F1={cls_f1:.3f} | Reg RMSE={reg_rmse:.3f} | vs baseline: {improvement:.1f}%")
            else:
                print(f"  {crop_key:<20s} | Cls F1={cls_f1:.3f} | Reg: insufficient data")

    print(f"\n  Models saved in: {MODEL_DIR}/{{crop}}/")
    print(f"  Each crop has: classifier.joblib, regressor.joblib, scaler.joblib, results.json")


if __name__ == "__main__":
    main()
