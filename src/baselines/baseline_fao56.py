"""
AquaSmart — FAO-56 dynamic baselines.

Replaces the naive "historical mean" baseline with real agronomic baselines.

Two baselines are provided:

  1. fao56_simple(df)
       Daily water need from raw FAO-56 formula, no soil memory.
       Decision: irrigate if need > 0.5 mm
       Amount:   need itself (mm)

  2. fao56_with_ru(df, ru_max)
       Full water-balance simulation WITHOUT machine learning.
       Same simulation logic as the target generator, but applied directly
       as a decision rule. This is the professional reference ("checkbook
       irrigation"). Our ML models must BEAT this baseline to prove value.

Both baselines take the preprocessed dataframe (with etc_mm, precipitation,
etc.) and return (y_decision_pred, y_amount_pred).
"""

import numpy as np
import pandas as pd


# ---- Baseline 1: simple daily FAO-56 ---------------------------------

def fao56_simple(df: pd.DataFrame,
                 eff_rain_coeff: float = 0.8,
                 decision_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw FAO-56 daily rule, no soil memory.

    Args:
        df: must contain 'etc_mm' and 'precipitation_sum'
        eff_rain_coeff: fraction of rainfall considered usable by the crop
        decision_threshold: mm/day below which irrigation is not triggered

    Returns:
        (decision_array, amount_array)
    """
    etc = df["etc_mm"].to_numpy()
    rain = df["precipitation_sum"].clip(lower=0).to_numpy()
    need = np.maximum(0.0, etc - eff_rain_coeff * rain)

    decision = (need > decision_threshold).astype(int)
    amount = np.where(decision == 1, need, 0.0)
    return decision, amount


# ---- Baseline 2: FAO-56 + soil water reserve (checkbook) -------------

def fao56_with_ru(df: pd.DataFrame,
                  ru_max: float,
                  eff_rain_coeff: float = 0.8,
                  trigger_frac: float = 0.5,
                  target_frac: float = 0.8,
                  dose_min: float = 15.0,
                  dose_max: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Full water-balance baseline (no ML).

    Maintains a daily soil stock. When stock drops below trigger_frac * RU_max,
    irrigate to refill up to target_frac * RU_max, bounded by [dose_min, dose_max].

    Args:
        df: must be sorted by (location, date) and contain
            'etc_mm', 'precipitation_sum', and optionally 'location'
        ru_max: soil water holding capacity (mm)
        trigger_frac: fraction of RU_max below which irrigation triggers
        target_frac: fraction of RU_max to refill to
        dose_min/dose_max: operational dose bounds

    Returns:
        (decision_array, amount_array)
    """
    decision = np.zeros(len(df), dtype=int)
    amount = np.zeros(len(df), dtype=float)

    def _simulate(group_idx: np.ndarray):
        etc = df.loc[group_idx, "etc_mm"].to_numpy()
        rain = df.loc[group_idx, "precipitation_sum"].clip(lower=0).to_numpy()
        eff_rain = eff_rain_coeff * rain

        stock = ru_max * 0.5  # start at half-full
        for k, idx in enumerate(group_idx):
            s = stock + eff_rain[k] - etc[k]
            s = max(0.0, min(ru_max, s))

            if s < trigger_frac * ru_max:
                raw_need = target_frac * ru_max - s
                dose = float(np.clip(raw_need, dose_min, dose_max))
                s = min(ru_max, s + dose)
                decision[idx] = 1
                amount[idx] = dose

            stock = s

    # Process per-location if available; otherwise one global pass
    if "location" in df.columns:
        for _, sub in df.groupby("location", sort=False):
            _simulate(sub.index.to_numpy())
    else:
        _simulate(df.index.to_numpy())

    return decision, amount


# ---- Evaluation helper -----------------------------------------------

def evaluate_baseline(y_true_decision: np.ndarray,
                      y_true_amount: np.ndarray,
                      y_pred_decision: np.ndarray,
                      y_pred_amount: np.ndarray) -> dict:
    """Compute F1 / MAE / RMSE for a baseline's predictions."""
    from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

    f1 = f1_score(y_true_decision, y_pred_decision, zero_division=0)
    mae = mean_absolute_error(y_true_amount, y_pred_amount)
    rmse = float(np.sqrt(mean_squared_error(y_true_amount, y_pred_amount)))
    return {"f1": f1, "mae": mae, "rmse": rmse}
