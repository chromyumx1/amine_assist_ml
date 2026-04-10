"""
plot_results.py — Generate all analysis plots.

Usage:
    python plot_results.py                   # from merged CSV
    python plot_results.py --csv path.csv    # custom CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from config import (
    PLOT_DIR, CSV_DIR, MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS,
    STATES, SENSITIVITY_RADII, ensure_dirs,
)


# ============================================================
# Utilities
# ============================================================
def add_yield_info(df):
    """Add methanol yield and category columns."""
    df = df.copy()
    df["MeOH_yield"] = df["amine"].map(MEOH_YIELD)
    df["yield_cat"] = df["amine"].map(YIELD_CATEGORY).fillna("unknown")
    return df


def color_by_category(df):
    """Return list of colors based on yield category."""
    return [CATEGORY_COLORS.get(cat, "#7f7f7f") for cat in df["yield_cat"]]


def annotate_amines(ax, df, x_col, y_col, fontsize=7):
    """Add amine name labels to scatter points."""
    for _, row in df.iterrows():
        ax.annotate(
            row["amine"], (row[x_col], row[y_col]),
            fontsize=fontsize, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points",
            alpha=0.8,
        )


def add_regression(ax, x, y, color="black"):
    """Add linear regression line and R² annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color=color, alpha=0.5, lw=1)
    ax.text(
        0.05, 0.95, f"R² = {r**2:.3f}\np = {p:.2e}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def category_legend(ax):
    """Add yield category legend."""
    handles = [
        Patch(facecolor=CATEGORY_COLORS["high"], label="High MeOH"),
        Patch(facecolor=CATEGORY_COLORS["medium"], label="Medium MeOH"),
        Patch(facecolor=CATEGORY_COLORS["low"], label="Low MeOH"),
        Patch(facecolor=CATEGORY_COLORS["unknown"], label="No data"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.8)


# ============================================================
# Individual plot functions
# ============================================================

def plot_vbur_vs_yield(df, state="2A"):
    """Plot %V_bur vs methanol yield for a given state."""
    sub = df[(df["state"] == state) & df["MeOH_yield"].notna()].copy()
    if len(sub) < 3:
        print(f"  [SKIP] Not enough data for Vbur vs yield ({state})")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = color_by_category(sub)
    ax.scatter(sub["Vbur_pct"], sub["MeOH_yield"], c=colors, s=60, edgecolors="k", lw=0.5, zorder=3)
    annotate_amines(ax, sub, "Vbur_pct", "MeOH_yield")
    add_regression(ax, sub["Vbur_pct"].values, sub["MeOH_yield"].values)
    category_legend(ax)

    ax.set_xlabel(f"%V_bur (R = 3.5 Å) — {state} state", fontsize=11)
    ax.set_ylabel("Methanol yield (%)", fontsize=11)
    ax.set_title(f"Amine steric footprint vs. methanol yield ({state})", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"Vbur_vs_yield_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: Vbur_vs_yield_{state}.png")


def plot_vbur_comparison_across_states(df):
    """
    Grouped bar chart: %V_bur for each amine across states.
    Useful to see how the amine's steric footprint changes along the pathway.
    """
    # Pivot: rows=amine, columns=state
    pivot = df.pivot_table(index="amine", columns="state", values="Vbur_pct")
    # Order columns by reaction pathway
    col_order = [s for s in ["2A", "TS3A", "3A", "H1", "TSH2", "H3", "4A", "TS5A", "5A"] if s in pivot.columns]
    pivot = pivot[col_order]

    if len(pivot) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 0.6), 6))
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="k", linewidth=0.3)
    ax.set_ylabel("%V_bur", fontsize=11)
    ax.set_xlabel("Amine", fontsize=11)
    ax.set_title("Buried volume across reaction states", fontsize=12)
    ax.legend(title="State", fontsize=8, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "Vbur_across_states.png"), dpi=300)
    plt.close(fig)
    print("  Saved: Vbur_across_states.png")


def plot_quadrant_analysis(df, state="2A"):
    """Stacked bar of quadrant %V_bur for each amine."""
    sub = df[df["state"] == state].copy()
    q_cols = [c for c in ["Q_SW", "Q_NW", "Q_NE", "Q_SE"] if c in sub.columns]
    if len(sub) == 0 or len(q_cols) == 0:
        return

    sub = sub.sort_values("Vbur_pct")

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.5), 5))
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    colors_q = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, qcol in enumerate(q_cols):
        vals = sub[qcol].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=qcol.replace("Q_", ""), color=colors_q[i],
               edgecolor="k", linewidth=0.3, width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sub["amine"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Quadrant %V_bur", fontsize=11)
    ax.set_title(f"Directional steric profile — {state} state", fontsize=12)
    ax.legend(title="Quadrant", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"quadrant_analysis_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: quadrant_analysis_{state}.png")


def plot_sasa_vs_yield(df, state="2A"):
    """SASA of amine fragment vs methanol yield."""
    sub = df[(df["state"] == state) & df["MeOH_yield"].notna()].copy()
    if len(sub) < 3 or "SASA_amine" not in sub.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total amine SASA
    colors = color_by_category(sub)
    axes[0].scatter(sub["SASA_amine"], sub["MeOH_yield"], c=colors, s=60,
                    edgecolors="k", lw=0.5)
    annotate_amines(axes[0], sub, "SASA_amine", "MeOH_yield")
    add_regression(axes[0], sub["SASA_amine"].values, sub["MeOH_yield"].values)
    axes[0].set_xlabel("SASA (amine fragment, Å²)", fontsize=10)
    axes[0].set_ylabel("Methanol yield (%)", fontsize=10)
    axes[0].set_title(f"Amine SASA vs. yield ({state})")
    axes[0].grid(True, alpha=0.3)

    # Right: amine N SASA
    sub2 = sub[sub["SASA_amine_N"].notna()]
    if len(sub2) >= 3:
        colors2 = color_by_category(sub2)
        axes[1].scatter(sub2["SASA_amine_N"], sub2["MeOH_yield"], c=colors2, s=60,
                        edgecolors="k", lw=0.5)
        annotate_amines(axes[1], sub2, "SASA_amine_N", "MeOH_yield")
        add_regression(axes[1], sub2["SASA_amine_N"].values, sub2["MeOH_yield"].values)
        axes[1].set_xlabel("SASA (amine N atom, Å²)", fontsize=10)
        axes[1].set_ylabel("Methanol yield (%)", fontsize=10)
        axes[1].set_title(f"Amine N accessibility vs. yield ({state})")
        axes[1].grid(True, alpha=0.3)

    category_legend(axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"SASA_vs_yield_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: SASA_vs_yield_{state}.png")


def plot_sterimol_vs_yield(df, state="2A"):
    """Sterimol L, B1, B5 vs methanol yield."""
    sub = df[(df["state"] == state) & df["MeOH_yield"].notna()].copy()
    sterimol_cols = ["Sterimol_L", "Sterimol_B1", "Sterimol_B5"]
    avail = [c for c in sterimol_cols if c in sub.columns and sub[c].notna().sum() >= 3]
    if len(avail) == 0:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]

    for ax, col in zip(axes, avail):
        sub2 = sub[sub[col].notna()]
        colors = color_by_category(sub2)
        ax.scatter(sub2[col], sub2["MeOH_yield"], c=colors, s=60,
                   edgecolors="k", lw=0.5)
        annotate_amines(ax, sub2, col, "MeOH_yield")
        add_regression(ax, sub2[col].values, sub2["MeOH_yield"].values)
        ax.set_xlabel(f"{col} (Å)", fontsize=10)
        ax.set_ylabel("Methanol yield (%)", fontsize=10)
        ax.set_title(f"{col} vs. yield ({state})")
        ax.grid(True, alpha=0.3)

    category_legend(axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"Sterimol_vs_yield_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: Sterimol_vs_yield_{state}.png")


def plot_vbur_vs_sterimol(df, state="2A"):
    """%V_bur vs Sterimol B5 — bound steric footprint vs intrinsic amine width."""
    sub = df[(df["state"] == state)].copy()
    if "Sterimol_B5" not in sub.columns:
        return
    sub = sub[sub["Sterimol_B5"].notna() & sub["Vbur_pct"].notna()]
    if len(sub) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = color_by_category(sub)
    ax.scatter(sub["Sterimol_B5"], sub["Vbur_pct"], c=colors, s=60,
               edgecolors="k", lw=0.5)
    annotate_amines(ax, sub, "Sterimol_B5", "Vbur_pct")
    add_regression(ax, sub["Sterimol_B5"].values, sub["Vbur_pct"].values)
    category_legend(ax)

    ax.set_xlabel("Sterimol B5 (Å)", fontsize=11)
    ax.set_ylabel(f"%V_bur ({state})", fontsize=11)
    ax.set_title(f"Amine width vs. buried volume ({state})", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"Vbur_vs_Sterimol_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: Vbur_vs_Sterimol_{state}.png")


def plot_sensitivity(csv_path=None):
    """Plot %V_bur ranking stability across different sphere radii."""
    if csv_path is None:
        csv_path = os.path.join(CSV_DIR, "Vbur_sensitivity.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] No sensitivity data found")
        return

    df = pd.read_csv(csv_path)

    # Only plot for 2A (or the first state available)
    states_avail = df["state"].unique()
    for state in ["2A"] + list(states_avail):
        sub = df[df["state"] == state]
        if len(sub) == 0:
            continue

        pivot = sub.pivot(index="amine", columns="radius", values="Vbur_pct")

        fig, ax = plt.subplots(figsize=(8, 6))
        for amine in pivot.index:
            ax.plot(pivot.columns, pivot.loc[amine], "o-", markersize=4, label=amine)

        ax.set_xlabel("Sphere radius (Å)", fontsize=11)
        ax.set_ylabel("%V_bur", fontsize=11)
        ax.set_title(f"Sensitivity to sphere radius — {state}", fontsize=12)
        ax.legend(fontsize=6, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"sensitivity_{state}.png"), dpi=300)
        plt.close(fig)
        print(f"  Saved: sensitivity_{state}.png")
        break  # just plot for one state


def plot_descriptor_heatmap(df):
    """
    Correlation heatmap of all numerical descriptors.
    Useful for identifying redundant vs. complementary descriptors.
    """
    # Use 2A state only for clean comparison
    sub = df[df["state"] == "2A"].copy()

    num_cols = [c for c in sub.columns if sub[c].dtype in [np.float64, np.int64]
                and c not in ["radius", "amine_N_index"]]
    if len(num_cols) < 3:
        return

    corr = sub[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(num_cols, fontsize=7)

    # Annotate cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Descriptor correlation matrix (2A state)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "descriptor_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    print("  Saved: descriptor_correlation_heatmap.png")


def plot_ru_n_distance_vs_yield(df, state="2A"):
    """Ru–N(amine) distance vs methanol yield."""
    sub = df[(df["state"] == state) & df["MeOH_yield"].notna()].copy()
    if "Ru_N_distance" not in sub.columns:
        return
    sub = sub[sub["Ru_N_distance"].notna()]
    if len(sub) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = color_by_category(sub)
    ax.scatter(sub["Ru_N_distance"], sub["MeOH_yield"], c=colors, s=60,
               edgecolors="k", lw=0.5)
    annotate_amines(ax, sub, "Ru_N_distance", "MeOH_yield")
    add_regression(ax, sub["Ru_N_distance"].values, sub["MeOH_yield"].values)
    category_legend(ax)

    ax.set_xlabel("Ru–N(amine) distance (Å)", fontsize=11)
    ax.set_ylabel("Methanol yield (%)", fontsize=11)
    ax.set_title(f"Bond distance vs. yield ({state})", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"RuN_dist_vs_yield_{state}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: RuN_dist_vs_yield_{state}.png")


# ============================================================
# Master plot generator
# ============================================================

def generate_all_plots(df=None):
    """Generate all plots from merged descriptor DataFrame."""
    ensure_dirs()

    if df is None:
        csv_path = os.path.join(CSV_DIR, "all_descriptors_merged.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] Cannot find {csv_path}. Run calculations first.")
            return
        df = pd.read_csv(csv_path)

    df = add_yield_info(df)

    print("\nGenerating plots...\n")

    # Per-state plots (focus on 2A, but also do others if data exists)
    for state in df["state"].unique():
        plot_vbur_vs_yield(df, state=state)
        plot_quadrant_analysis(df, state=state)
        plot_sasa_vs_yield(df, state=state)
        plot_sterimol_vs_yield(df, state=state)
        plot_vbur_vs_sterimol(df, state=state)
        plot_ru_n_distance_vs_yield(df, state=state)

    # Cross-state plots
    plot_vbur_comparison_across_states(df)
    plot_descriptor_heatmap(df)
    plot_sensitivity()

    print("\nAll plots saved to:", PLOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Generate all analysis plots")
    parser.add_argument("--csv", default=None, help="Path to merged CSV")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = None

    generate_all_plots(df)


if __name__ == "__main__":
    main()
