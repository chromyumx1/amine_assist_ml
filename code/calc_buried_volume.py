"""
calc_buried_volume.py — Compute %V_bur, quadrant analysis, and steric maps.

Usage:
    python calc_buried_volume.py                # all states, all amines
    python calc_buried_volume.py --state 2A     # only 2A state
    python calc_buried_volume.py --sensitivity  # run radius sensitivity analysis
    python calc_buried_volume.py --maps         # generate steric map images
    python calc_buried_volume.py --maps --state 2A  # maps for 2A only
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morfeus import BuriedVolume, read_xyz

from config import (
    RU_INDEX, Z_AXIS_ATOMS, XZ_PLANE_ATOMS,
    SPHERE_RADIUS, INCLUDE_H, RADII_TYPE, RADII_SCALE,
    SENSITIVITY_RADII, CSV_DIR, STERIC_MAP_DIR, PLOT_DIR,
    MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS, ensure_dirs,
)
from parse_structures import load_registry


def compute_vbur(entry, radius=SPHERE_RADIUS, include_extra=False):
    """
    Compute buried volume for one structure.

    Parameters
    ----------
    entry : dict
        Structure registry entry.
    radius : float
        Sphere radius in Å.
    include_extra : bool
        If True, include trailing fragment atoms (CO2, OCOH, H2) together
        with amine. If False, exclude them (measure pure amine only).

    Returns
    -------
    dict with results, or None on failure.
    """
    filepath = entry["filepath"]
    label = entry["label"]

    try:
        elements, coordinates = read_xyz(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return None

    # Atoms to exclude = catalyst + optionally trailing fragment
    excluded = list(entry["cat_atoms"])
    if not include_extra and entry["extra_atoms"]:
        excluded += list(entry["extra_atoms"])

    try:
        bv = BuriedVolume(
            elements,
            coordinates,
            RU_INDEX,
            excluded_atoms=excluded,
            radius=radius,
            include_hs=INCLUDE_H,
            radii_type=RADII_TYPE,
            radii_scale=RADII_SCALE,
            z_axis_atoms=Z_AXIS_ATOMS,
            xz_plane_atoms=XZ_PLANE_ATOMS,
        )
    except Exception as e:
        print(f"  [ERROR] BuriedVolume failed for {label}: {e}")
        return None

    result = {
        "label": label,
        "state": entry["state"],
        "amine": entry["amine"],
        "radius": radius,
        "Vbur_pct": bv.percent_buried_volume,
        "Vbur_abs": bv.buried_volume,
        "Vfree": bv.free_volume,
    }

    # Quadrant analysis
    # morfeus quadrant integer keys: 1=NE, 2=NW, 3=SW, 4=SE
    # data lives at bv.quadrants["percent_buried_volume"][int_key]
    _QUADRANT_KEY_MAP = {"NE": 1, "NW": 2, "SW": 3, "SE": 4}
    try:
        bv.octant_analysis()
        pct = bv.quadrants["percent_buried_volume"]
        for q_name, q_key in _QUADRANT_KEY_MAP.items():
            result[f"Q_{q_name}"] = pct[q_key]
    except Exception:
        for q_name in _QUADRANT_KEY_MAP:
            result[f"Q_{q_name}"] = np.nan

    return result, bv


def run_all(states=None, radius=SPHERE_RADIUS, include_extra=False):
    """Compute %V_bur for all structures. Returns DataFrame."""
    registry = load_registry()
    results = []

    for entry in registry:
        if states and entry["state"] not in states:
            continue

        out = compute_vbur(entry, radius=radius, include_extra=include_extra)
        if out is None:
            continue
        result, _ = out
        results.append(result)
        print(f"  {result['label']:>20s}  %V_bur = {result['Vbur_pct']:5.1f}%")

    df = pd.DataFrame(results)
    return df


def run_sensitivity(states=None):
    """Run %V_bur at multiple sphere radii."""
    print("\n=== Sensitivity analysis ===")
    all_dfs = []
    for r in SENSITIVITY_RADII:
        print(f"\nRadius = {r:.1f} Å")
        df = run_all(states=states, radius=r)
        all_dfs.append(df)
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def generate_steric_maps(states=None, include_extra=False):
    """Generate steric map PNG images for all structures."""
    registry = load_registry()

    for entry in registry:
        if states and entry["state"] not in states:
            continue

        out = compute_vbur(entry, include_extra=include_extra)
        if out is None:
            continue
        result, bv = out

        # Create state-specific subdirectory
        state_dir = os.path.join(STERIC_MAP_DIR, entry["state"])
        os.makedirs(state_dir, exist_ok=True)

        outpath = os.path.join(state_dir, f"steric_map_{result['label']}.png")

        try:
            bv.plot_steric_map()          # creates its own fig; plt.show() is no-op (Agg)
            fig = plt.gcf()
            fig.set_size_inches(5, 5)
            fig.axes[0].set_title(f"{result['label']}\n%V_bur = {result['Vbur_pct']:.1f}%",
                                  fontsize=11)
            fig.tight_layout()
            fig.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {outpath}")
        except Exception as e:
            print(f"  [ERROR] Steric map for {result['label']}: {e}")
            plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Buried volume calculations")
    parser.add_argument("--state", nargs="*", default=None,
                        help="Limit to specific states (e.g., --state 2A 3A)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run radius sensitivity analysis")
    parser.add_argument("--maps", action="store_true",
                        help="Generate steric map images")
    parser.add_argument("--include-extra", action="store_true",
                        help="Include trailing fragments (CO2 etc.) with amine")
    args = parser.parse_args()

    ensure_dirs()
    states = args.state

    if args.sensitivity:
        df = run_sensitivity(states=states)
        outpath = os.path.join(CSV_DIR, "Vbur_sensitivity.csv")
        df.to_csv(outpath, index=False)
        print(f"\nSaved: {outpath}")
        return

    if args.maps:
        generate_steric_maps(states=states, include_extra=args.include_extra)
        return

    # Default: compute %V_bur for all
    print("\n=== Buried Volume Calculations ===\n")
    df = run_all(states=states, include_extra=args.include_extra)

    outpath = os.path.join(CSV_DIR, "Vbur_results.csv")
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(df)} results to {outpath}")

    # Also save per-state pivot tables
    if len(df) > 0:
        for state in df["state"].unique():
            sub = df[df["state"] == state].sort_values("amine")
            state_path = os.path.join(CSV_DIR, f"Vbur_{state}.csv")
            sub.to_csv(state_path, index=False)


if __name__ == "__main__":
    main()
