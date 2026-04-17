"""
plot_vbur_correlation.py — Correlation plot: %V_bur (2A state) vs MeOH yield.

Reads:  output/csv/Vbur_2A_summary.csv
Writes: output/electronic_par/vbur_vs_yield_correlation.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))

CSV_IN  = os.path.join(BASE_DIR, "output", "csv", "Vbur_2A_summary.csv")
OUT_PNG = os.path.join(BASE_DIR, "output", "electronic_par",
                       "vbur_vs_yield_correlation.png")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN).dropna(subset=["Vbur_pct", "MeOH_yield"])
x = df["Vbur_pct"].values
y = df["MeOH_yield"].values
labels = df["amine"].values

# ── statistics ────────────────────────────────────────────────────────────────
r, p_val = pearsonr(x, y)
slope, intercept, r_val, p_lin, se = stats.linregress(x, y)

# 95 % confidence band
x_line = np.linspace(x.min() - 0.3, x.max() + 0.3, 300)
y_line = slope * x_line + intercept
n = len(x)
x_mean = x.mean()
se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
t_crit = stats.t.ppf(0.975, df=n - 2)
ci_upper = y_line + t_crit * se_line
ci_lower = y_line - t_crit * se_line

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

# Colour points by MeOH yield (log scale for better visual spread)
norm = matplotlib.colors.Normalize(vmin=0, vmax=y.max())
cmap = matplotlib.colormaps["RdYlGn"]
colors = cmap(norm(y))

sc = ax.scatter(x, y, c=y, cmap="RdYlGn", s=90, zorder=5,
                edgecolors="grey", linewidths=0.5, norm=norm)

# Regression line
ax.plot(x_line, y_line, color="#2c6fad", lw=2, zorder=4,
        label=f"Linear fit  (r = {r:.3f}, p = {p_val:.3f})")

# 95 % CI shading
ax.fill_between(x_line, ci_lower, ci_upper,
                color="#2c6fad", alpha=0.12, zorder=3, label="95 % CI")

# ── amine labels ──────────────────────────────────────────────────────────────
# Slight offsets to avoid overlap; shift high-yield points left/right manually
NUDGE = {
    "THQ":   ( 0.05, -4.5),
    "INDO":  (-0.55,  1.0),
    "NMA":   ( 0.05, -4.5),
    "PIP":   ( 0.08,  1.5),
    "OTL":   ( 0.08,  1.5),
    "MOR":   (-0.6,   1.5),
    "MEPI":  ( 0.08,  1.5),
    "MTL":   ( 0.08,  1.5),
    "BEMT":  ( 0.08,  1.5),
    "AMP":   (-0.6,   1.5),
    "EAE":   ( 0.08, -4.5),
    "DBUA":  (-0.6,  -4.5),
    "ISBZ":  ( 0.05, -5.5),
    "DEA":   ( 0.05,  1.5),
    "BMAE":  (-0.65,  1.5),
}
for xi, yi, lbl in zip(x, y, labels):
    dx, dy = NUDGE.get(lbl, (0.08, 1.5))
    ax.annotate(
        lbl,
        xy=(xi, yi),
        xytext=(xi + dx, yi + dy),
        fontsize=9.5,
        ha="left" if dx >= 0 else "right",
        va="bottom",
        color="black",
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
    )

# ── colourbar ─────────────────────────────────────────────────────────────────
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("MeOH yield (%)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# ── annotations ───────────────────────────────────────────────────────────────
#stats_text = (
#    f"Pearson r = {r:.3f}\n"
#    f"R² = {r_val**2:.3f}\n"
#    f"p = {p_val:.3f}"
#)
#ax.text(0.04, 0.96, stats_text,
#        transform=ax.transAxes, fontsize=11,
#        va="top", ha="left",
#        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))

# ── formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel("%V$_{bur}$ (2A state)", fontsize=13)
ax.set_ylabel("MeOH Yield (%)", fontsize=13)
ax.set_title("%V$_{bur}$ vs MeOH Yield — 2A State", fontsize=14, pad=10)
ax.tick_params(labelsize=11)
#ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(x.min() - 0.8, x.max() + 0.8)
ax.set_ylim(-5, y.max() + 8)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PNG}")
print(f"  Pearson r = {r:.4f}, p = {p_val:.4f}, R² = {r_val**2:.4f}")
