"""
AquaSmart v4 — Training: one model per cluster, geographic split.

Two-stage ML pipeline per cluster:
  Stage 1 (classification): "irrigate today or not?"  — F1-selected
  Stage 2 (regression):     "how many mm?"            — RMSE-selected

Comparison references (the credible baselines):
  - fao56_simple   : raw daily FAO-56 formula, no soil memory
  - fao56_with_ru  : full water-balance simulation (professional reference)

Evaluation: geographic split (train on 8 cities, test on Montpellier + Rennes).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

# Make baseline_fao56 importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from baseline_fao56 import fao56_simple, fao56_with_ru, evaluate_baseline
except ImportError:
    from baselines.baseline_fao56 import fao56_simple, fao56_with_ru, evaluate_baseline


RANDOM_STATE = 42


def train_classifiers(X_train, y_train, X_test, y_test):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        f1 = f1_score(y_test, pred, zero_division=0)
        results[name] = {"model": m, "f1": f1, "pred": pred}
        print(f"    {name}: F1={f1:.3f}")
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"  ✅ Best classifier: {best} (F1={results[best]['f1']:.3f})")
    return best, results


def train_regressors(X_train, y_train, X_test, y_test):
    if len(X_train) < 10 or len(X_test) < 1:
        print("    ⚠ Not enough irrigation days to train regressor")
        return None, {}
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, random_state=RANDOM_STATE),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results[name] = {"model": m, "rmse": rmse, "pred": pred}
        print(f"    {name}: RMSE={rmse:.3f}")
    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"  ✅ Best regressor: {best} (RMSE={results[best]['rmse']:.3f})")
    return best, results


def eval_two_stage(clf_pred, X_test, y_test_dec, y_test_amt,
                   regressor, mean_dose):
    """Combine Stage 1 + Stage 2 into final dose predictions."""
    final_pred = np.zeros(len(y_test_amt))
    mask = clf_pred == 1
    if mask.sum() > 0 and regressor is not None:
        final_pred[mask] = regressor.predict(X_test[mask])
    elif mask.sum() > 0:
        final_pred[mask] = mean_dose
    final_pred = np.clip(final_pred, 0, 50)

    f1 = f1_score(y_test_dec, clf_pred, zero_division=0)
    mae = mean_absolute_error(y_test_amt, final_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test_amt, final_pred)))
    return {"f1": f1, "mae": mae, "rmse": rmse}


def evaluate_fao_baselines(meta_test: pd.DataFrame,
                           y_te_dec: np.ndarray,
                           y_te_amt: np.ndarray) -> dict:
    """Run both FAO-56 baselines on the test set."""
    dec1, amt1 = fao56_simple(meta_test)
    b1 = evaluate_baseline(y_te_dec, y_te_amt, dec1, amt1)

    # RU baseline must run per (location, crop) with its own ru_max and chronological order
    dec2 = np.zeros(len(meta_test), dtype=int)
    amt2 = np.zeros(len(meta_test), dtype=float)
    meta_sorted = meta_test.sort_values(["location", "crop", "date"])
    for (loc, crop), sub in meta_sorted.groupby(["location", "crop"], sort=False):
        ru = float(sub["ru_max"].iloc[0])
        sub_reset = sub.reset_index(drop=False)  # keep original index in 'index' column
        d, a = fao56_with_ru(sub_reset, ru_max=ru)
        original_idx = sub_reset["index"].to_numpy()
        dec2[original_idx] = d
        amt2[original_idx] = a
    b2 = evaluate_baseline(y_te_dec, y_te_amt, dec2, amt2)

    return {"fao56_simple": b1, "fao56_with_ru": b2}


def process_cluster(cluster_id: int, proc_dir: Path, models_dir: Path) -> dict | None:
    cdir = proc_dir / f"cluster_{cluster_id}"
    data_file = cdir / "data.npz"
    meta_file = cdir / "meta_test.csv"
    if not data_file.exists():
        print(f"\n⚠ Cluster {cluster_id}: data missing, skipping")
        return None

    print("\n" + "=" * 60)
    print(f"🌾 Cluster {cluster_id}")
    print("=" * 60)

    data = np.load(data_file, allow_pickle=True)
    X_train, X_test = data["X_train"], data["X_test"]
    y_tr_dec, y_te_dec = data["y_train_decision"], data["y_test_decision"]
    y_tr_amt, y_te_amt = data["y_train_amount"], data["y_test_amount"]
    meta_test = pd.read_csv(meta_file, parse_dates=["date"])

    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Irrigation days — train: {int(y_tr_dec.sum())} | test: {int(y_te_dec.sum())}")

    if int(y_te_dec.sum()) == 0:
        print("  ⚠ No irrigation days in test — this cluster cannot be evaluated")
        return None

    # --- Baselines ---
    print("\n  📏 FAO-56 baselines:")
    baselines = evaluate_fao_baselines(meta_test, y_te_dec, y_te_amt)
    for name, m in baselines.items():
        print(f"    {name}: F1={m['f1']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")

    # --- Stage 1 ---
    print("\n  Stage 1 — Classification:")
    best_clf_name, clf_res = train_classifiers(X_train, y_tr_dec, X_test, y_te_dec)
    best_clf = clf_res[best_clf_name]["model"]
    clf_pred = clf_res[best_clf_name]["pred"]

    # --- Stage 2 ---
    print("\n  Stage 2 — Regression (irrigation days only):")
    train_irr = y_tr_dec == 1
    test_irr = y_te_dec == 1
    best_reg_name, reg_res = train_regressors(
        X_train[train_irr], y_tr_amt[train_irr],
        X_test[test_irr], y_te_amt[test_irr],
    )
    best_reg = reg_res[best_reg_name]["model"] if best_reg_name else None
    mean_dose = float(y_tr_amt[train_irr].mean()) if train_irr.sum() > 0 else 0.0

    # --- Combined eval ---
    ml = eval_two_stage(clf_pred, X_test, y_te_dec, y_te_amt, best_reg, mean_dose)
    print(f"\n  📊 ML pipeline: F1={ml['f1']:.3f}  MAE={ml['mae']:.3f}  RMSE={ml['rmse']:.3f}")

    ref = baselines["fao56_with_ru"]
    f1_gain = ml["f1"] - ref["f1"]
    mae_gain = (ref["mae"] - ml["mae"]) / max(ref["mae"], 1e-9) * 100
    print(f"  📈 vs fao56_with_ru: ΔF1={f1_gain:+.3f}  |  MAE gain={mae_gain:+.1f}%")

    # --- Save models ---
    out_models = models_dir / f"cluster_{cluster_id}"
    out_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, out_models / "classifier.pkl")
    if best_reg is not None:
        joblib.dump(best_reg, out_models / "regressor.pkl")
    print(f"  💾 Models saved to {out_models}")

    return {
        "cluster": cluster_id,
        "baseline_simple": baselines["fao56_simple"],
        "baseline_with_ru": baselines["fao56_with_ru"],
        "ml": ml,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 95)
    print("FINAL — ML vs FAO-56 baselines (test: Montpellier + Rennes, 2020-2024)")
    print("=" * 95)
    hdr = (f"{'Cluster':<10}{'Simple F1':>11}{'RU F1':>9}{'ML F1':>9}"
           f"{'Simple MAE':>13}{'RU MAE':>10}{'ML MAE':>10}{'MAE gain':>12}")
    print(hdr)
    print("-" * 95)
    for r in rows:
        s, ru, ml = r["baseline_simple"], r["baseline_with_ru"], r["ml"]
        gain = (ru["mae"] - ml["mae"]) / max(ru["mae"], 1e-9) * 100
        print(f"{r['cluster']:<10}{s['f1']:>11.3f}{ru['f1']:>9.3f}{ml['f1']:>9.3f}"
              f"{s['mae']:>13.3f}{ru['mae']:>10.3f}{ml['mae']:>10.3f}{gain:>11.1f}%")


def main():
    print("=" * 60)
    print("AquaSmart v4 — Training (geographic split)")
    print("=" * 60)

    proc_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in [1, 2, 3, 4]:
        res = process_cluster(cid, proc_dir, models_dir)
        if res is not None:
            rows.append(res)

    if rows:
        print_summary(rows)
        df_out = pd.DataFrame([
            {
                "cluster": r["cluster"],
                "baseline_simple_f1": r["baseline_simple"]["f1"],
                "baseline_simple_mae": r["baseline_simple"]["mae"],
                "baseline_ru_f1": r["baseline_with_ru"]["f1"],
                "baseline_ru_mae": r["baseline_with_ru"]["mae"],
                "ml_f1": r["ml"]["f1"],
                "ml_mae": r["ml"]["mae"],
                "ml_rmse": r["ml"]["rmse"],
            }
            for r in rows
        ])
        df_out.to_csv(models_dir / "results_v4.csv", index=False)
        print(f"\n💾 Results saved to {models_dir / 'results_v4.csv'}")


if __name__ == "__main__":
    main()