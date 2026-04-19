"""
AquaSmart — Generate report figures from v4 results.

Produces 6 figures saved to reports/figures/v4/:
  01_ml_vs_baselines.png       — Bar chart: ML F1 vs FAO-56 baselines per cluster
  02_mae_gains.png              — Bar chart: MAE improvement per cluster (%)
  03_target_distribution.png    — Histogram: distribution of irrigation doses in training
  04_irrigation_rate_by_crop.png — Bar chart: % irrigation days per crop
  05_feature_correlation.png    — Heatmap: correlation of features with target
  06_confusion_matrix_cluster4.png — Confusion matrix for the best-performing cluster

Run from project root:
    python src/visualization/generate_report_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "models" / "results_v4.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "reports" / "figures" / "v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Styling ----
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
})

CLUSTER_NAMES = {
    1: "C1 — Winter cereals",
    2: "C2 — Summer deep-rooted",
    3: "C3 — Winter oilseeds",
    4: "C4 — Shallow-rooted",
}

TEAL = "#0f7c6b"
LEAF = "#6a9444"
SUN = "#e5b54b"
ALERT = "#bf5b2c"
WATER = "#3b86a8"


def fig_01_ml_vs_baselines():
    """Grouped bar chart: F1 score per cluster, baselines vs ML."""
    df = pd.read_csv(RESULTS_CSV)
    df["cluster_label"] = df["cluster"].map(CLUSTER_NAMES)

    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, df["baseline_simple_f1"], width,
           label="FAO-56 simple (naive)", color=ALERT, alpha=0.85)
    ax.bar(x, df["baseline_ru_f1"], width,
           label="FAO-56 + soil reserve (professional)", color=SUN, alpha=0.9)
    ax.bar(x + width, df["ml_f1"], width,
           label="AquaSmart ML pipeline", color=TEAL, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(df["cluster_label"], rotation=15, ha="right")
    ax.set_ylabel("F1 score (irrigation decision)")
    ax.set_title("Classification performance: ML vs FAO-56 baselines per cluster")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    for i, (s, r, m) in enumerate(zip(df["baseline_simple_f1"],
                                       df["baseline_ru_f1"], df["ml_f1"])):
        ax.text(i - width, s + 0.02, f"{s:.2f}", ha="center", fontsize=9)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width, m + 0.02, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "01_ml_vs_baselines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_02_mae_gains():
    """Bar chart: MAE improvement (%) of ML vs RU baseline per cluster."""
    df = pd.read_csv(RESULTS_CSV)
    df["cluster_label"] = df["cluster"].map(CLUSTER_NAMES)
    df["mae_gain_pct"] = (
        (df["baseline_ru_mae"] - df["ml_mae"]) / df["baseline_ru_mae"] * 100
    )

    colors = [LEAF if g > 0 else ALERT for g in df["mae_gain_pct"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df["cluster_label"], df["mae_gain_pct"], color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MAE improvement vs FAO-56 w/ soil reserve (%)")
    ax.set_title("Water-use accuracy: ML gain over the professional baseline")
    ax.set_xticklabels(df["cluster_label"], rotation=15, ha="right")

    for bar, gain in zip(bars, df["mae_gain_pct"]):
        y = bar.get_height()
        offset = 2.5 if y >= 0 else -4.5
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset,
                f"{gain:+.1f}%", ha="center",
                fontweight="bold", fontsize=11,
                color="#19362d")

    plt.tight_layout()
    out = OUT_DIR / "02_mae_gains.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_03_target_distribution():
    """Histogram: distribution of irrigation doses (mm) across all crops."""
    doses = []
    for csv_file in sorted(RAW_DIR.glob("aquasmart_v4_*.csv")):
        df = pd.read_csv(csv_file, usecols=["irrigation_applied_mm"])
        mask = df["irrigation_applied_mm"] > 0
        doses.extend(df.loc[mask, "irrigation_applied_mm"].tolist())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(doses, bins=50, color=WATER, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Irrigation dose (mm)")
    ax.set_ylabel("Number of irrigation events")
    ax.set_title(
        f"Distribution of target irrigation doses\n"
        f"(all crops combined, n={len(doses):,} irrigation days)"
    )
    ax.axvline(np.mean(doses), color=ALERT, linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(doses):.1f} mm")
    ax.axvline(np.median(doses), color=LEAF, linestyle="--", linewidth=2,
               label=f"Median = {np.median(doses):.1f} mm")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "03_target_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_04_irrigation_rate_by_crop():
    """Bar chart: % of days that are irrigation days, per crop."""
    rates = []
    for csv_file in sorted(RAW_DIR.glob("aquasmart_v4_*.csv")):
        crop = csv_file.stem.replace("aquasmart_v4_", "")
        df = pd.read_csv(csv_file, usecols=["irrigation_needed", "cluster"])
        rate = df["irrigation_needed"].mean() * 100
        cluster = int(df["cluster"].iloc[0])
        rates.append({"crop": crop, "rate": rate, "cluster": cluster})

    df = pd.DataFrame(rates).sort_values(["cluster", "rate"])
    cluster_colors = {1: SUN, 2: ALERT, 3: WATER, 4: LEAF}
    colors = [cluster_colors[c] for c in df["cluster"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(df["crop"], df["rate"], color=colors, alpha=0.85)
    ax.set_xlabel("Share of days requiring irrigation (%)")
    ax.set_title("Irrigation demand per crop (5 years × 10 cities)")

    # Cluster legend
    for cid, color in cluster_colors.items():
        ax.bar(0, 0, color=color, alpha=0.85, label=CLUSTER_NAMES[cid])
    ax.legend(loc="lower right", title="Cluster", frameon=True)

    for bar, rate in zip(bars, df["rate"]):
        ax.text(rate + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "04_irrigation_rate_by_crop.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_05_feature_correlation():
    """
    Correlation heatmap: all features vs the irrigation target.
    Uses processed cluster_2 data (largest signal, most balanced).
    """
    data_file = PROCESSED_DIR / "cluster_2" / "data.npz"
    if not data_file.exists():
        print(f"  ⚠ Skipping fig_05: {data_file} not found")
        return

    data = np.load(data_file, allow_pickle=True)
    X = data["X_train"]
    y = data["y_train_amount"]
    feature_names = list(data["feature_names"])

    # Build a DataFrame for correlation analysis
    df = pd.DataFrame(X, columns=feature_names)
    df["target_irrigation_mm"] = y
    corr = df.corr()["target_irrigation_mm"].drop("target_irrigation_mm")
    corr = corr.sort_values()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [ALERT if c < 0 else TEAL for c in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors, alpha=0.85)
    ax.set_xlabel("Correlation with irrigation amount (mm)")
    ax.set_title(
        "Feature correlation with irrigation target\n"
        "(cluster 2 — summer deep-rooted crops, scaled features)"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, corr.values):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "05_feature_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig_06_confusion_matrix_cluster4():
    """
    Confusion matrix for cluster 4 (best ML performance).
    Recreates predictions by loading the classifier.
    """
    import joblib

    data_file = PROCESSED_DIR / "cluster_4" / "data.npz"
    clf_file = PROJECT_ROOT / "models" / "cluster_4" / "classifier.pkl"
    if not data_file.exists() or not clf_file.exists():
        print(f"  ⚠ Skipping fig_06: cluster_4 data or model missing")
        return

    data = np.load(data_file, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test_decision"]
    clf = joblib.load(clf_file)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=["No irrigation", "Irrigation"],
        yticklabels=["No irrigation", "Irrigation"],
        cbar=False, ax=ax,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — Cluster 4 (shallow-rooted crops)\n"
                 "Test set: Montpellier + Rennes, 2020-2024")

    # Add precision/recall annotation
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ax.text(1.02, -0.15,
            f"Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}",
            transform=ax.transAxes, fontsize=10,
            ha="right", va="top", fontweight="bold", color=TEAL)

    plt.tight_layout()
    out = OUT_DIR / "06_confusion_matrix_cluster4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def main():
    print("=" * 60)
    print("AquaSmart — Generating v4 report figures")
    print("=" * 60)
    print(f"Output: {OUT_DIR}\n")

    if not RESULTS_CSV.exists():
        print(f"⚠ {RESULTS_CSV} not found. Run train_v4.py first.")
        return

    fig_01_ml_vs_baselines()
    fig_02_mae_gains()
    fig_03_target_distribution()
    fig_04_irrigation_rate_by_crop()
    fig_05_feature_correlation()
    fig_06_confusion_matrix_cluster4()

    print("\n" + "=" * 60)
    print("DONE. Figures ready for the technical report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
