"""
AquaSmart — Visualization Utilities
Reusable plotting functions for EDA and reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Style configuration
AQUASMART_COLORS = {
    "primary": "#1D9E75",     # Teal green
    "secondary": "#378ADD",   # Blue
    "warning": "#EF9F27",     # Amber
    "danger": "#E24B4A",      # Red
    "neutral": "#888780",     # Gray
}

def set_style():
    """Apply AquaSmart visual style to all plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_missing_values(df: pd.DataFrame, save_path: str = None):
    """Bar chart of missing values per column."""
    set_style()
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        print("✅ No missing values found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    missing_pct = (missing / len(df) * 100)
    missing_pct.plot(kind="barh", color=AQUASMART_COLORS["warning"], ax=ax)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column")
    
    for i, (val, pct) in enumerate(zip(missing, missing_pct)):
        ax.text(pct + 0.5, i, f"{val} ({pct:.1f}%)", va="center", fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_distributions(df: pd.DataFrame, columns: list = None, save_path: str = None):
    """Histogram + KDE for numeric columns."""
    set_style()
    numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                     color=AQUASMART_COLORS["primary"], alpha=0.6)
        axes[i].set_title(col)
    
    # Hide unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Feature Distributions", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Heatmap of feature correlations."""
    set_style()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", center=0,
                cmap="RdYlGn", ax=ax, square=True, linewidths=0.5,
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_target_analysis(df: pd.DataFrame, target_col: str,
                         feature_col: str = None, save_path: str = None):
    """Analyze the target variable distribution and its relationship with a feature."""
    set_style()
    
    if feature_col:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Target distribution
    sns.histplot(df[target_col].dropna(), kde=True, ax=ax1,
                 color=AQUASMART_COLORS["primary"], alpha=0.6)
    ax1.axvline(df[target_col].mean(), color=AQUASMART_COLORS["danger"],
                linestyle="--", label=f"Mean: {df[target_col].mean():.2f}")
    ax1.axvline(df[target_col].median(), color=AQUASMART_COLORS["warning"],
                linestyle="--", label=f"Median: {df[target_col].median():.2f}")
    ax1.legend()
    ax1.set_title(f"Distribution of {target_col}")
    
    # Scatter vs feature
    if feature_col:
        ax2.scatter(df[feature_col], df[target_col], alpha=0.3,
                    color=AQUASMART_COLORS["secondary"], s=10)
        ax2.set_xlabel(feature_col)
        ax2.set_ylabel(target_col)
        ax2.set_title(f"{target_col} vs {feature_col}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
