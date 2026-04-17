"""
steric_map_2A_grid.py — Compute %V_bur for all amines in the 2A state and
display their steric maps together in a 4×7 grid with a shared z(Å) colorbar
at the bottom.

Usage:
    python steric_map_2A_grid.py

Output:
    output/electronic_par/steric_map_2A_grid.png   (high-res grid)
    output/csv/Vbur_2A_summary.csv                 (Vbur values)
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

from morfeus import BuriedVolume, read_xyz
from morfeus.buried_volume import rotate_coordinates

# ── paths ─────────────────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
sys.path.insert(0, CODE_DIR)

from config import (
    RU_INDEX, Z_AXIS_ATOMS, XZ_PLANE_ATOMS,
    SPHERE_RADIUS, INCLUDE_H, RADII_TYPE, RADII_SCALE,
    N_CATALYST_ATOMS, XYZ_DIR, MEOH_YIELD,
    ensure_dirs,
)
from parse_structures import load_registry

OUT_PLOT = os.path.join(BASE_DIR, "output", "electronic_par", "steric_map_2A_grid.png")
OUT_CSV  = os.path.join(BASE_DIR, "output", "csv", "Vbur_2A_summary.csv")

# ── grid layout ───────────────────────────────────────────────────────────────
NCOLS = 7
NROWS = 4                   # 4×7 = 28 panels; 26 amines + 2 spare
GRID  = 120                 # grid resolution for steric map
LEVELS = 150                # contour levels
CMAP  = "viridis"


# ═══════════════════════════════════════════════════════════════════════════════
# Steric map grid extractor (replicates morfeus internals without plotting)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_steric_grid(bv: BuriedVolume, grid: int = GRID):
    """
    Return (x_, y_, z) arrays for the steric map surface.
    Replicates BuriedVolume.plot_steric_map() grid logic without matplotlib.
    """
    atoms          = bv._atoms
    center         = np.array(bv._sphere.center)
    all_coords     = bv._all_coordinates.copy()
    coords         = np.array([a.coordinates for a in atoms])
    r              = bv._sphere.radius

    # Translate to center
    all_coords -= center
    coords     -= center

    # z-axis direction: from z_axis_atoms midpoint → center → amine
    z_ax_coords = all_coords[np.array(bv._z_axis_atoms) - 1]
    point  = np.mean(z_ax_coords, axis=0)
    vector = point / np.linalg.norm(point)

    # Rotate so z-axis aligns with [0, 0, -1]
    coords = rotate_coordinates(coords, vector, np.array([0, 0, -1]))

    # Build grid
    x_ = np.linspace(-r, r, grid)
    y_ = np.linspace(-r, r, grid)

    z = []
    for xy in np.dstack(np.meshgrid(x_, y_)).reshape(-1, 2):
        xp, yp = xy
        if np.linalg.norm(xy) > r:
            z.append(np.nan)
            continue
        z_list = []
        for i, atom in enumerate(atoms):
            xs, ys, zs = coords[i]
            test = atom.radius**2 - (xp - xs)**2 - (yp - ys)**2
            if test >= 0:
                z_atom = math.sqrt(test) + zs
                z_list.append(z_atom)
        if z_list:
            z_max = max(z_list)
            if z_max < 0:
                if np.linalg.norm([xp, yp, z_max]) >= r:
                    z_max = np.nan
        else:
            z_max = np.nan
        z.append(z_max)

    z_arr = np.array(z).reshape(len(x_), len(y_))
    return x_, y_, z_arr


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)

    # Load 2A structures from registry
    registry = load_registry()
    entries_2A = sorted(
        [e for e in registry if e["state"] == "2A"],
        key=lambda e: e["amine"]
    )
    print(f"Found {len(entries_2A)} amines in 2A state\n")

    # ── compute Vbur + steric grids ───────────────────────────────────────────
    results = []
    for entry in entries_2A:
        amine = entry["amine"]
        filepath = entry["filepath"]
        elements, coordinates = read_xyz(filepath)

        # Exclude catalyst atoms so %V_bur reflects the amine only
        excluded = list(entry["cat_atoms"]) + list(entry["extra_atoms"])
        bv = BuriedVolume(
            elements, coordinates, RU_INDEX,
            excluded_atoms=excluded,
            z_axis_atoms=Z_AXIS_ATOMS,
            xz_plane_atoms=XZ_PLANE_ATOMS,
            radius=SPHERE_RADIUS,
            include_hs=INCLUDE_H,
            radii_type=RADII_TYPE,
            radii_scale=RADII_SCALE,
        )
        vbur = bv.fraction_buried_volume * 100
        x_, y_, z_arr = compute_steric_grid(bv, grid=GRID)
        yield_val = MEOH_YIELD.get(amine, None)
        results.append({
            "amine":    amine,
            "vbur":     vbur,
            "yield":    yield_val,
            "x_":       x_,
            "y_":       y_,
            "z_arr":    z_arr,
        })
        marker = f"  yield={yield_val:.1f}%" if yield_val is not None else ""
        print(f"  {amine:>8s}  %V_bur = {vbur:.1f}%{marker}")

    # ── build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(NCOLS * 3.6, NROWS * 4.2 + 1.6))

    # Reserve a narrow row at the bottom for the shared horizontal colorbar
    outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[NROWS * 4.2, 1.6],  # map panels | colorbar
        hspace=0.12,
    )
    map_gs = gridspec.GridSpecFromSubplotSpec(
        NROWS, NCOLS,
        subplot_spec=outer[0],
        hspace=0.38,
        wspace=0.22,
    )
    cbar_ax = fig.add_subplot(outer[1])

    # Shared colour limits: −R to +R  (same as morfeus default)
    r = SPHERE_RADIUS
    clim = (-r, r)

    # Use the first valid z array to get contourf mappable for the colorbar
    mappable = None

    for idx, res in enumerate(results):
        row = idx // NCOLS
        col = idx % NCOLS
        ax = fig.add_subplot(map_gs[row, col])

        x_, y_, z_arr = res["x_"], res["y_"], res["z_arr"]

        cf = ax.contourf(x_, y_, z_arr, LEVELS, cmap=CMAP)
        cf.set_clim(*clim)

        # Sphere boundary circle
        circle = plt.Circle((0, 0), r, fill=False, lw=0.8, color="white")
        ax.add_patch(circle)

        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove individual axes spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Amine label (top)
        yield_str = (f"\n{res['yield']:.1f}%"
                     if res["yield"] is not None else "")
        ax.set_title(
            f"{res['amine']}{yield_str}\n%V_bur = {res['vbur']:.1f}%",
            fontsize=16, pad=3, linespacing=1.3
        )

        if mappable is None:
            mappable = cf

    # Hide any unused panels (28 − 26 = 2)
    total_panels = NROWS * NCOLS
    for idx in range(len(results), total_panels):
        row = idx // NCOLS
        col = idx % NCOLS
        ax = fig.add_subplot(map_gs[row, col])
        ax.set_visible(False)

    # Shared colorbar at the bottom (horizontal)
    if mappable is not None:
        mappable.set_clim(*clim)
        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("z (Å)", fontsize=22, labelpad=10)
        cbar_ax.tick_params(labelsize=18)

    fig.suptitle(
        f"Steric Maps — 2A State  (R = {SPHERE_RADIUS} Å, Bondi×{RADII_SCALE})\n"
        "Label: amine  ·  MeOH yield  ·  %V_bur",
        fontsize=24, y=1.002
    )

    fig.savefig(OUT_PLOT, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {OUT_PLOT}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame([
        {"amine": r["amine"], "Vbur_pct": r["vbur"], "MeOH_yield": r["yield"]}
        for r in results
    ]).sort_values("Vbur_pct", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
