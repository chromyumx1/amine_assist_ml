"""
plot_electronic.py — Plots and correlation analysis for all electronic
descriptors (WBI, NBO E2, NEDA, hydricity, NLMO/E_pwx) from dataset_exported.xlsx.

Outputs saved to: output/electronic_par/

Usage:
    python plot_electronic.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(CODE_DIR, ".."))
DATA_PATH = os.path.join(CODE_DIR, "dataset_exported.xlsx")
OUT_DIR   = os.path.join(BASE_DIR, "output", "electronic_par")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
from config import MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS

# ── descriptor groups ─────────────────────────────────────────────────────────
STATES_WBI   = ["2A", "TS3A", "3A", "4A"]
STATES_E2    = ["2A", "TS3A", "3A", "4A"]
STATES_NEDA  = ["2A", "3A", "4A", "5A", "H1", "H3", "TS3A", "TS5A", "TSH2"]
STATES_NLMO  = ["2A", "TS3A", "3A", "4A", "5A"]
STATES_HYD   = ["2A", "3A", "4A", "5A"]   # hyd2→2A, hyd3→3A …

# NEDA component prefixes (column name = prefix + "_" + state)
NEDA_COMPONENTS = [
    "E_CT", "E_ES", "E_POL", "E_XC",
    "E_DEF_frag1", "E_DEF_frag2",
    "E_SE_frag1",  "E_SE_frag2",
    "E_elec", "E_core", "E_int",
]
NEDA_LABELS = {
    "E_CT":       "Charge Transfer",
    "E_ES":       "Electrostatic",
    "E_POL":      "Polarization",
    "E_XC":       "Exchange-Correlation",
    "E_DEF_frag1":"Deformation (cat.)",
    "E_DEF_frag2":"Deformation (amine)",
    "E_SE_frag1": "Self-Energy (cat.)",
    "E_SE_frag2": "Self-Energy (amine)",
    "E_elec":     "Elec. Interaction",
    "E_core":     "Core Repulsion",
    "E_int":      "Total Interaction",
}
NEDA_UNITS = "kcal/mol"

# ── colour palette ────────────────────────────────────────────────────────────
STATE_COLORS = {
    "2A":"#1f77b4","TS3A":"#aec7e8","3A":"#ff7f0e",
    "4A":"#2ca02c","TS5A":"#98df8a","5A":"#d62728",
    "H1":"#9467bd","TSH2":"#c5b0d5","H3":"#8c564b",
}
NEDA_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_excel(DATA_PATH, header=0)
    df = df.dropna(subset=["Amine"])
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    # rename Amine column to canonical key used in config
    df = df.rename(columns={"Amine": "amine", "Yield": "yield"})
    return df


def scatter_vs_yield(ax, x, y, amines, title, xlabel, ylabel="MeOH yield (%)"):
    """Generic scatter + regression on a given axes."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        ax.set_title(title + "\n[insufficient data]", fontsize=9)
        return
    colors = [CATEGORY_COLORS.get(YIELD_CATEGORY.get(a, "unknown"), "#7f7f7f")
              for a in amines]
    ax.scatter(x[mask], y[mask], c=[colors[i] for i in np.where(mask)[0]],
               s=55, edgecolors="k", lw=0.4, zorder=3)
    for i in np.where(mask)[0]:
        ax.annotate(amines[i], (x[i], y[i]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", alpha=0.8)
    # regression
    if np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sl, ic, r, p, _ = stats.linregress(x[mask], y[mask])
        xl = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, sl * xl + ic, "--", color="k", alpha=0.4, lw=1)
        ax.text(0.05, 0.95, f"R² = {r**2:.3f}\np = {p:.2e}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)


def category_legend_handles():
    return [
        Patch(facecolor=CATEGORY_COLORS["high"],    label="High yield"),
        Patch(facecolor=CATEGORY_COLORS["medium"],  label="Medium yield"),
        Patch(facecolor=CATEGORY_COLORS["low"],     label="Low yield"),
        Patch(facecolor=CATEGORY_COLORS["unknown"], label="No data"),
    ]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WBI — Wiberg Bond Index (Ru–N)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wbi(df):
    amines = df["amine"].values
    yield_  = df["yield"].values

    # 1a. WBI vs yield — one panel per state
    fig, axes = plt.subplots(1, len(STATES_WBI), figsize=(4.5 * len(STATES_WBI), 4.5),
                             sharey=True)
    for ax, state in zip(axes, STATES_WBI):
        col = f"WBI_{state}"
        x = pd.to_numeric(df[col], errors="coerce").values
        scatter_vs_yield(ax, x, yield_, amines,
                         title=f"WBI ({state})",
                         xlabel="Wiberg Bond Index (Ru–N)")
    axes[0].set_ylabel("MeOH yield (%)", fontsize=9)
    fig.legend(handles=category_legend_handles(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Ru–N Wiberg Bond Index vs. MeOH Yield", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "WBI_vs_yield.png")

    # 1b. WBI across states (line plot per amine, coloured by yield category)
    fig, ax = plt.subplots(figsize=(8, 5))
    wbi_data = {state: pd.to_numeric(df[f"WBI_{state}"], errors="coerce").values
                for state in STATES_WBI}
    x_pos = np.arange(len(STATES_WBI))
    for i, amine in enumerate(amines):
        vals = [wbi_data[s][i] for s in STATES_WBI]
        if any(np.isfinite(v) for v in vals):
            cat = YIELD_CATEGORY.get(amine, "unknown")
            color = CATEGORY_COLORS.get(cat, "#7f7f7f")
            ax.plot(x_pos, vals, "o-", color=color, alpha=0.65,
                    lw=1.2, ms=5, label=amine)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STATES_WBI, fontsize=10)
    ax.set_xlabel("Reaction State", fontsize=11)
    ax.set_ylabel("WBI (Ru–N)", fontsize=11)
    ax.set_title("Ru–N Bond Order Evolution Along Pathway", fontsize=12)
    ax.legend(handles=category_legend_handles(), fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "WBI_across_states.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NBO E2 — second-order perturbation energies
# ═══════════════════════════════════════════════════════════════════════════════

def plot_e2(df):
    amines = df["amine"].values
    yield_ = df["yield"].values
    e2_cols = {
        "2A":   "E2_int2A",
        "TS3A": "E2_TS3A",
        "3A":   "E2_int3A",
        "4A":   "E2_int4A",
    }

    fig, axes = plt.subplots(1, len(e2_cols), figsize=(4.5 * len(e2_cols), 4.5),
                             sharey=True)
    for ax, (state, col) in zip(axes, e2_cols.items()):
        x = pd.to_numeric(df[col], errors="coerce").values
        scatter_vs_yield(ax, x, yield_, amines,
                         title=f"E2 ({state})",
                         xlabel="E₂ (kcal/mol)")
    axes[0].set_ylabel("MeOH yield (%)", fontsize=9)
    fig.legend(handles=category_legend_handles(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("NBO 2nd-Order Perturbation Energy (N→Ru) vs. MeOH Yield",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "E2_vs_yield.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NEDA — Natural Energy Decomposition Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _neda_col(component, state):
    return f"{component}_{state}"


def plot_neda_vs_yield(df):
    """One figure per NEDA component; subplots = states."""
    amines = df["amine"].values
    yield_ = df["yield"].values

    for comp, label in NEDA_LABELS.items():
        available_states = [s for s in STATES_NEDA
                            if _neda_col(comp, s) in df.columns
                            and pd.to_numeric(df[_neda_col(comp, s)],
                                              errors="coerce").notna().sum() >= 3]
        if not available_states:
            continue

        ncols = min(4, len(available_states))
        nrows = (len(available_states) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4.2 * nrows),
                                 sharey=True, squeeze=False)
        ax_flat = axes.flatten()

        for i, state in enumerate(available_states):
            x = pd.to_numeric(df[_neda_col(comp, state)], errors="coerce").values
            scatter_vs_yield(ax_flat[i], x, yield_, amines,
                             title=state,
                             xlabel=f"{comp} ({NEDA_UNITS})")
            if i % ncols == 0:
                ax_flat[i].set_ylabel("MeOH yield (%)", fontsize=8)

        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].set_visible(False)

        fig.legend(handles=category_legend_handles(), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"NEDA: {label} vs. MeOH Yield", fontsize=13, y=1.01)
        fig.tight_layout()
        save(fig, f"NEDA_{comp}_vs_yield.png")


def plot_neda_decomposition(df):
    """Stacked-bar NEDA decomposition per amine for each state."""
    amines = df["amine"].values
    # Only plot the primary interaction components (not frag1/frag2 duplicates)
    plot_comps = ["E_CT", "E_ES", "E_POL", "E_XC", "E_DEF_frag1",
                  "E_DEF_frag2", "E_SE_frag1", "E_SE_frag2"]

    for state in STATES_NEDA:
        cols = [_neda_col(c, state) for c in plot_comps]
        available = [c for c in cols if c in df.columns]
        if not available:
            continue

        sub = df[[c for c in ["amine", "yield"] + available]].copy()
        sub = sub.sort_values("yield", ascending=False, na_position="last")
        sub_amines = sub["amine"].values
        x = np.arange(len(sub_amines))

        fig, ax = plt.subplots(figsize=(max(12, len(sub_amines) * 0.55), 5))
        pos_bottom = np.zeros(len(sub_amines))
        neg_bottom = np.zeros(len(sub_amines))

        for j, (col, comp) in enumerate(zip(available, plot_comps)):
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0).values
            color = NEDA_COLORS[j % len(NEDA_COLORS)]
            pos_vals = np.where(vals > 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)
            ax.bar(x, pos_vals, bottom=pos_bottom, label=NEDA_LABELS[comp],
                   color=color, edgecolor="k", lw=0.2, width=0.75)
            ax.bar(x, neg_vals, bottom=neg_bottom,
                   color=color, edgecolor="k", lw=0.2, width=0.75)
            pos_bottom += pos_vals
            neg_bottom += neg_vals

        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_amines, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"Energy ({NEDA_UNITS})", fontsize=10)
        ax.set_title(f"NEDA Energy Decomposition — {state} (sorted by yield ↓)",
                     fontsize=11)
        ax.legend(fontsize=7, ncol=2, loc="upper right",
                  bbox_to_anchor=(1.18, 1.0))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save(fig, f"NEDA_decomposition_{state}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hydricity
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hydricity(df):
    amines = df["amine"].values
    yield_ = df["yield"].values
    hyd_map = {"2A": "hyd2", "3A": "hyd3", "4A": "hyd4", "5A": "hyd5"}

    # vs yield per state
    fig, axes = plt.subplots(1, len(hyd_map), figsize=(4.5 * len(hyd_map), 4.5),
                             sharey=True)
    for ax, (state, col) in zip(axes, hyd_map.items()):
        x = pd.to_numeric(df[col], errors="coerce").values
        scatter_vs_yield(ax, x, yield_, amines,
                         title=f"Hydricity ({state})",
                         xlabel="ΔG°H⁻ (kcal/mol)")
    axes[0].set_ylabel("MeOH yield (%)", fontsize=9)
    fig.legend(handles=category_legend_handles(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Hydricity (ΔG°H⁻) vs. MeOH Yield", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "hydricity_vs_yield.png")

    # evolution across states (line plot)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(hyd_map))
    hyd_vals = {state: pd.to_numeric(df[col], errors="coerce").values
                for state, col in hyd_map.items()}
    for i, amine in enumerate(amines):
        vals = [hyd_vals[s][i] for s in hyd_map]
        if any(np.isfinite(v) for v in vals):
            cat = YIELD_CATEGORY.get(amine, "unknown")
            ax.plot(x_pos, vals, "o-",
                    color=CATEGORY_COLORS.get(cat, "#7f7f7f"),
                    alpha=0.65, lw=1.2, ms=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(hyd_map.keys()), fontsize=10)
    ax.set_xlabel("Intermediate state", fontsize=11)
    ax.set_ylabel("ΔG°H⁻ (kcal/mol)", fontsize=11)
    ax.set_title("Hydricity Profile Along Pathway", fontsize=12)
    ax.legend(handles=category_legend_handles(), fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "hydricity_across_states.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NLMO occupation energies and E_pwx
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nlmo(df):
    amines = df["amine"].values
    yield_ = df["yield"].values

    # 5a. SUMEx_NLMO and SUME_pwx (summary descriptors) vs yield per state
    for metric, label, unit in [
        ("SUMEx_NLMO", "ΣEx(NLMO)", "kcal/mol"),
        ("SUME_pwx",   "ΣE_pwx",    "kcal/mol"),
    ]:
        avail_states = [s for s in STATES_NLMO
                        if f"{metric}_{s}" in df.columns]
        if not avail_states:
            continue
        ncols = min(4, len(avail_states))
        nrows = (len(avail_states) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4.2 * nrows),
                                 sharey=True, squeeze=False)
        ax_flat = axes.flatten()
        for i, state in enumerate(avail_states):
            x = pd.to_numeric(df[f"{metric}_{state}"], errors="coerce").values
            scatter_vs_yield(ax_flat[i], x, yield_, amines,
                             title=state,
                             xlabel=f"{label} ({unit})")
            if i % ncols == 0:
                ax_flat[i].set_ylabel("MeOH yield (%)", fontsize=8)
        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].set_visible(False)
        fig.legend(handles=category_legend_handles(), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"{label} vs. MeOH Yield", fontsize=13, y=1.01)
        fig.tight_layout()
        save(fig, f"{metric}_vs_yield.png")

    # 5b. Individual Ex_NLMO(1), Ex_NLMO(2) for each state — one figure per state
    for state in STATES_NLMO:
        nlmo_cols = {k: f"{k}_{state}" for k in
                     ["Ex_NLMO(1)", "Ex_NLMO(2)", "Ex_NLMO(3)"]
                     if f"{k}_{state}" in df.columns}
        pwx_cols  = {k: f"{k}_{state}" for k in
                     ["E_pwx(1)", "E_pwx(2)", "E_pwx(3)",
                      "E_pwx(1,2)", "E_pwx(1,3)", "E_pwx(2,3)"]
                     if f"{k}_{state}" in df.columns}
        all_cols  = {**nlmo_cols, **pwx_cols}
        if not all_cols:
            continue

        n = len(all_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4.2 * nrows),
                                 sharey=True, squeeze=False)
        ax_flat = axes.flatten()
        for i, (key, col) in enumerate(all_cols.items()):
            x = pd.to_numeric(df[col], errors="coerce").values
            scatter_vs_yield(ax_flat[i], x, yield_, amines,
                             title=key,
                             xlabel="Energy (kcal/mol)")
            if i % ncols == 0:
                ax_flat[i].set_ylabel("MeOH yield (%)", fontsize=8)
        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].set_visible(False)
        fig.legend(handles=category_legend_handles(), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"NLMO & E_pwx Components — {state}", fontsize=13, y=1.01)
        fig.tight_layout()
        save(fig, f"NLMO_Epwx_detail_{state}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Correlation heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

def _corr_heatmap(corr, title, filename, annot_fontsize=6):
    n = len(corr)
    size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right",
                       fontsize=annot_fontsize + 1)
    ax.set_yticklabels(corr.index, fontsize=annot_fontsize + 1)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=annot_fontsize,
                        color="white" if abs(val) > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.7)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    save(fig, filename)


def plot_correlation_heatmap_per_state(df):
    """For each state: correlate all its electronic descriptors with yield."""
    for state in STATES_NEDA:
        # Gather all columns belonging to this state
        state_cols = [c for c in df.columns if c.endswith(f"_{state}")]
        if not state_cols:
            continue
        sub = df[["yield"] + state_cols].copy()
        sub = sub.apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna(subset=["yield"])
        # Drop columns with <4 valid values
        sub = sub.loc[:, sub.notna().sum() >= 4]
        if sub.shape[1] < 3:
            continue
        corr = sub.corr(method="pearson")
        _corr_heatmap(
            corr,
            title=f"Correlation Matrix — {state} Electronic Descriptors",
            filename=f"corr_heatmap_{state}.png",
            annot_fontsize=6,
        )


def plot_yield_correlation_bar(df):
    """Bar chart: Pearson r of every electronic descriptor vs yield, all states."""
    all_ecols = [c for c in df.columns if any(
        c.startswith(p) for p in
        ["WBI_", "E2_", "E_CT_", "E_ES_", "E_POL_", "E_XC_",
         "E_DEF_", "E_SE_", "E_elec_", "E_core_", "E_int_",
         "hyd", "Ex_NLMO", "E_pwx", "SUMEx_NLMO", "SUME_pwx"]
    )]

    records = []
    for col in all_ecols:
        sub = df[["yield", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 4 or sub[col].std() == 0:
            continue
        r, p = stats.pearsonr(sub["yield"], sub[col])
        records.append({"descriptor": col, "r": r, "p": p, "r2": r**2})

    if not records:
        return
    rdf = pd.DataFrame(records).sort_values("r", ascending=False)

    # Split into positive and negative r for colour
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in rdf["r"]]

    fig, ax = plt.subplots(figsize=(max(14, len(rdf) * 0.35), 6))
    ax.bar(range(len(rdf)), rdf["r"].values, color=colors, edgecolor="k", lw=0.2)
    ax.set_xticks(range(len(rdf)))
    ax.set_xticklabels(rdf["descriptor"].values, rotation=90, fontsize=5.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.5,  color="green", lw=0.8, ls="--", alpha=0.5, label="|r|=0.5")
    ax.axhline(-0.5, color="green", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Pearson r (vs. MeOH yield)", fontsize=11)
    ax.set_title("Correlation of All Electronic Descriptors with MeOH Yield",
                 fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "all_descriptors_r_vs_yield.png")

    # Save table
    rdf.to_csv(os.path.join(OUT_DIR, "descriptor_yield_correlation.csv"), index=False)
    print("  Saved: descriptor_yield_correlation.csv")

    return rdf


def plot_top_descriptors_vs_yield(df, rdf, top_n=12):
    """Scatter grid for the top |r| descriptors vs yield."""
    if rdf is None or len(rdf) == 0:
        return
    top = rdf.reindex(rdf["r"].abs().sort_values(ascending=False).index).head(top_n)
    amines = df["amine"].values
    yield_ = df["yield"].apply(pd.to_numeric, errors="coerce").values

    ncols = 4
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows),
                             squeeze=False)
    ax_flat = axes.flatten()
    for i, (_, row) in enumerate(top.iterrows()):
        col = row["descriptor"]
        x = pd.to_numeric(df[col], errors="coerce").values
        scatter_vs_yield(ax_flat[i], x, yield_, amines,
                         title=f"{col}\n(r={row['r']:.3f}, p={row['p']:.2e})",
                         xlabel=col)
        if i % ncols == 0:
            ax_flat[i].set_ylabel("MeOH yield (%)", fontsize=8)
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)
    fig.legend(handles=category_legend_handles(), loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Top {top_n} Electronic Descriptors vs. MeOH Yield (by |r|)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "top_descriptors_vs_yield.png")


def plot_global_heatmap(df, rdf):
    """Full correlation heatmap: top descriptors × yield."""
    if rdf is None or len(rdf) == 0:
        return
    # Use top-40 by |r| to keep the figure readable
    top_cols = rdf.reindex(
        rdf["r"].abs().sort_values(ascending=False).index
    ).head(40)["descriptor"].tolist()

    sub = df[["yield"] + top_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(subset=["yield"])
    sub = sub.loc[:, sub.notna().sum() >= 4]
    corr = sub.corr(method="pearson")
    _corr_heatmap(
        corr,
        title="Global Correlation Matrix — Top Electronic Descriptors",
        filename="corr_heatmap_global.png",
        annot_fontsize=5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\nLoading data from {DATA_PATH}")
    df = load_data()
    print(f"  {len(df)} amines loaded\n")
    print(f"Output directory: {OUT_DIR}\n")

    print("=== WBI (Wiberg Bond Index) ===")
    plot_wbi(df)

    print("\n=== NBO E2 (2nd-order perturbation) ===")
    plot_e2(df)

    print("\n=== NEDA — component vs yield ===")
    plot_neda_vs_yield(df)

    print("\n=== NEDA — energy decomposition bars ===")
    plot_neda_decomposition(df)

    print("\n=== Hydricity ===")
    plot_hydricity(df)

    print("\n=== NLMO & E_pwx ===")
    plot_nlmo(df)

    print("\n=== Correlation heatmaps per state ===")
    plot_correlation_heatmap_per_state(df)

    print("\n=== Global r vs yield bar chart ===")
    rdf = plot_yield_correlation_bar(df)

    print("\n=== Top descriptors scatter grid ===")
    plot_top_descriptors_vs_yield(df, rdf, top_n=12)

    print("\n=== Global correlation heatmap (top 40) ===")
    plot_global_heatmap(df, rdf)

    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
