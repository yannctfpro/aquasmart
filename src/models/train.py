"""
AquaSmart — Model Training Pipeline
====================================
Two-stage modeling approach:
  Stage 1: Classification — does this day require irrigation? (binary)
  Stage 2: Regression — how much water is needed? (mm/day, irrigation days only)

Includes baseline comparison (fixed schedule) for evaluation.

Usage:
    python src/models/train.py

Input:  data/processed/train.csv, test.csv, train_irrigation.csv, test_irrigation.csv
Output: models/classifier.joblib, models/regressor.joblib, reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)

sns.set_theme(style="whitegrid")

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

for d in [MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    """Load all preprocessed datasets."""
    print("=" * 60)
    print("Loading preprocessed data...")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train_irr = pd.read_csv(DATA_DIR / "train_irrigation.csv")
    test_irr = pd.read_csv(DATA_DIR / "test_irrigation.csv")

    feature_cols = [c for c in train.columns if c not in ["water_need_index", "irrigation_needed"]]

    X_train = train[feature_cols]
    y_train_cls = train["irrigation_needed"]
    y_train_reg = train["water_need_index"]

    X_test = test[feature_cols]
    y_test_cls = test["irrigation_needed"]
    y_test_reg = test["water_need_index"]

    X_train_irr = train_irr[feature_cols]
    y_train_irr = train_irr["water_need_index"]
    X_test_irr = test_irr[feature_cols]
    y_test_irr = test_irr["water_need_index"]

    print(f"  Full train:       {X_train.shape}")
    print(f"  Full test:        {X_test.shape}")
    print(f"  Irrigation train: {X_train_irr.shape}")
    print(f"  Irrigation test:  {X_test_irr.shape}")
    print(f"  Features: {feature_cols}")

    return (X_train, y_train_cls, y_train_reg, X_test, y_test_cls, y_test_reg,
            X_train_irr, y_train_irr, X_test_irr, y_test_irr, feature_cols)


# ============================================================
# 2. BASELINE: FIXED SCHEDULE
# ============================================================
def evaluate_baseline(y_test_cls, y_test_reg):
    """
    Baseline: a fixed irrigation schedule.
    Irrigates every day during growing season (growth_stage > 0)
    with a fixed amount = historical mean of irrigation days.
    """
    print("\n" + "=" * 60)
    print("BASELINE: Fixed Schedule")
    print("=" * 60)

    # The baseline always says "irrigate" during growing season
    # Since we don't have growth_stage in test directly, use the
    # fact that irrigation_needed captures the same idea
    # Baseline prediction: always irrigate (optimistic fixed schedule)
    baseline_cls = np.ones_like(y_test_cls)

    # Fixed amount: mean of all training irrigation amounts
    mean_irrigation = 1.65  # from EDA: mean of non-zero WNI
    baseline_reg = np.where(y_test_reg > 0, mean_irrigation, 0)

    # Classification metrics
    acc = accuracy_score(y_test_cls, baseline_cls)
    prec = precision_score(y_test_cls, baseline_cls)
    rec = recall_score(y_test_cls, baseline_cls)
    f1 = f1_score(y_test_cls, baseline_cls)

    print(f"\n  Classification (irrigate or not?):")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1 Score:  {f1:.3f}")

    # Regression metrics (on days where irrigation actually happens)
    mask = y_test_reg > 0
    if mask.sum() > 0:
        rmse = np.sqrt(mean_squared_error(y_test_reg[mask], baseline_reg[mask]))
        mae = mean_absolute_error(y_test_reg[mask], baseline_reg[mask])
        r2 = r2_score(y_test_reg[mask], baseline_reg[mask])
        print(f"\n  Regression (how much? — irrigation days):")
        print(f"    RMSE:      {rmse:.3f} mm/day")
        print(f"    MAE:       {mae:.3f} mm/day")
        print(f"    R2:        {r2:.3f}")

    return {
        "name": "Fixed Schedule",
        "cls_accuracy": acc, "cls_precision": prec,
        "cls_recall": rec, "cls_f1": f1,
        "reg_rmse": rmse, "reg_mae": mae, "reg_r2": r2,
    }


# ============================================================
# 3. STAGE 1: CLASSIFICATION
# ============================================================
def train_classifiers(X_train, y_train, X_test, y_test):
    """Train and evaluate classification models."""
    print("\n" + "=" * 60)
    print("STAGE 1: Classification (irrigate or not?)")
    print("=" * 60)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
    }

    results = {}
    best_f1 = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
        }

        print(f"    Accuracy:  {acc:.3f}")
        print(f"    Precision: {prec:.3f}")
        print(f"    Recall:    {rec:.3f}")
        print(f"    F1 Score:  {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\n  Best classifier: {best_name} (F1={best_f1:.3f})")

    # Save best classifier
    joblib.dump(best_model, MODEL_DIR / "classifier.joblib")
    print(f"  Saved to models/classifier.joblib")

    # Confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=["No irrigation", "Irrigation"],
                yticklabels=["No irrigation", "Irrigation"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        importances.plot(kind="barh", color="#1D9E75", ax=ax)
        ax.set_title(f"Feature Importance — {best_name} (Classification)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "09_feature_importance_cls.png", dpi=150)
        plt.close()

    return results, best_model, best_name


# ============================================================
# 4. STAGE 2: REGRESSION
# ============================================================
def train_regressors(X_train, y_train, X_test, y_test):
    """Train and evaluate regression models on irrigation days only."""
    print("\n" + "=" * 60)
    print("STAGE 2: Regression (how much water? — irrigation days)")
    print("=" * 60)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
    }

    results = {}
    best_rmse = float("inf")
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}

        print(f"    RMSE:  {rmse:.3f} mm/day")
        print(f"    MAE:   {mae:.3f} mm/day")
        print(f"    R2:    {r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    print(f"\n  Best regressor: {best_name} (RMSE={best_rmse:.3f})")

    # Save best regressor
    joblib.dump(best_model, MODEL_DIR / "regressor.joblib")
    print(f"  Saved to models/regressor.joblib")

    # Predicted vs actual plot
    y_pred_best = best_model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred_best, alpha=0.3, s=10, color="#378ADD")
    max_val = max(y_test.max(), y_pred_best.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    axes[0].set_xlabel("Actual (mm/day)")
    axes[0].set_ylabel("Predicted (mm/day)")
    axes[0].set_title(f"Predicted vs Actual — {best_name}")
    axes[0].legend()

    # Residuals
    residuals = y_test.values - y_pred_best
    axes[1].hist(residuals, bins=40, color="#1D9E75", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (mm/day)")
    axes[1].set_title(f"Residual Distribution — {best_name}")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_regression_results.png", dpi=150)
    plt.close()

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        importances.plot(kind="barh", color="#378ADD", ax=ax)
        ax.set_title(f"Feature Importance — {best_name} (Regression)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "11_feature_importance_reg.png", dpi=150)
        plt.close()

    return results, best_model, best_name


# ============================================================
# 5. COMPARISON SUMMARY
# ============================================================
def generate_summary(baseline, cls_results, reg_results, best_cls, best_reg):
    """Generate final comparison report."""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Stage 1: Classification (irrigate or not?) ---")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Baseline (always yes)':<25s} {baseline['cls_accuracy']:>10.3f} {baseline['cls_precision']:>10.3f} {baseline['cls_recall']:>10.3f} {baseline['cls_f1']:>10.3f}")
    for name, metrics in cls_results.items():
        marker = " <-- BEST" if name == best_cls else ""
        print(f"  {name:<25s} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f}{marker}")

    print(f"\n--- Stage 2: Regression (how much? — irrigation days) ---")
    print(f"  {'Model':<25s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (fixed mean)':<25s} {baseline['reg_rmse']:>10.3f} {baseline['reg_mae']:>10.3f} {baseline['reg_r2']:>10.3f}")
    for name, metrics in reg_results.items():
        marker = " <-- BEST" if name == best_reg else ""
        print(f"  {name:<25s} {metrics['rmse']:>10.3f} {metrics['mae']:>10.3f} {metrics['r2']:>10.3f}{marker}")

    # Water savings estimation
    baseline_total = baseline["reg_mae"] * 365  # mm/year error
    best_reg_metrics = reg_results[best_reg]
    ai_total = best_reg_metrics["mae"] * 365

    print(f"\n--- Water Savings Estimate ---")
    print(f"  Baseline MAE over a year:  {baseline_total:.0f} mm of error")
    print(f"  AI model MAE over a year:  {ai_total:.0f} mm of error")
    print(f"  Improvement:               {(1 - ai_total/baseline_total)*100:.1f}% more precise")

    # Save results as JSON
    results = {
        "baseline": baseline,
        "classification": cls_results,
        "regression": reg_results,
        "best_classifier": best_cls,
        "best_regressor": best_reg,
    }
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to models/results.json")

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification F1
    names_cls = ["Baseline"] + list(cls_results.keys())
    f1_scores = [baseline["cls_f1"]] + [m["f1"] for m in cls_results.values()]
    colors_cls = ["#B4B2A9"] + ["#1D9E75" if n == best_cls else "#5DCAA5" for n in cls_results]
    axes[0].barh(names_cls, f1_scores, color=colors_cls, edgecolor="white")
    axes[0].set_xlabel("F1 Score")
    axes[0].set_title("Stage 1: Classification")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(f1_scores):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va="center")

    # Regression RMSE
    names_reg = ["Baseline"] + list(reg_results.keys())
    rmses = [baseline["reg_rmse"]] + [m["rmse"] for m in reg_results.values()]
    colors_reg = ["#B4B2A9"] + ["#378ADD" if n == best_reg else "#85B7EB" for n in reg_results]
    axes[1].barh(names_reg, rmses, color=colors_reg, edgecolor="white")
    axes[1].set_xlabel("RMSE (mm/day) — lower is better")
    axes[1].set_title("Stage 2: Regression")
    for i, v in enumerate(rmses):
        axes[1].text(v + 0.02, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_model_comparison.png", dpi=150)
    plt.close()
    print(f"  Comparison chart saved to reports/figures/12_model_comparison.png")


# ============================================================
# MAIN
# ============================================================
def main():
    # Load
    (X_train, y_train_cls, y_train_reg, X_test, y_test_cls, y_test_reg,
     X_train_irr, y_train_irr, X_test_irr, y_test_irr, feature_cols) = load_data()

    # Baseline
    baseline = evaluate_baseline(y_test_cls, y_test_reg)

    # Stage 1: Classification
    cls_results, best_cls_model, best_cls_name = train_classifiers(
        X_train, y_train_cls, X_test, y_test_cls
    )

    # Stage 2: Regression (irrigation days only)
    reg_results, best_reg_model, best_reg_name = train_regressors(
        X_train_irr, y_train_irr, X_test_irr, y_test_irr
    )

    # Summary
    generate_summary(baseline, cls_results, reg_results, best_cls_name, best_reg_name)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Classifier: models/classifier.joblib ({best_cls_name})")
    print(f"  Regressor:  models/regressor.joblib ({best_reg_name})")
    print(f"  Scaler:     models/scaler.joblib")
    print(f"  Results:    models/results.json")
    print(f"  Figures:    reports/figures/08-12_*.png")


if __name__ == "__main__":
    main()
