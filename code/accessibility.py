"""
calc_accessibility.py — Ru-centric accessibility analysis for amine-Ru complexes.

This script measures how much of the Ru coordination sphere remains accessible
AFTER amine binding, rather than how much space the amine itself occupies.

Key descriptors computed:
  1. Ru SASA           — solvent-accessible surface area of the Ru atom
                         (how exposed is Ru to solvent/reactant?)
  2. Ru SASA shielded  — decomposed into shielding by amine vs by catalyst
  3. Free %V_bur       — 100 − total buried volume within small sphere near Ru
                         (complement of buried volume = accessible volume)
  4. Hemisphere split  — free volume on AMINE side vs OPPOSITE side of Ru
                         (the opposite-amine hemisphere is where H2 would approach
                          for assistive H1 → TSH2 heterolytic cleavage)
  5. Probe-sphere test — whether a sphere of H2 radius fits near Ru without
                         overlapping any atom (binary + min distance)
  6. Ligand-gap angles — angular gaps between existing Ru ligands
                         (largest gap = preferred incoming ligand trajectory)

All outputs go to output/accessibility/.

Usage:
    python calc_accessibility.py                    # all states
    python calc_accessibility.py --state 2A         # only 2A
    python calc_accessibility.py --probe-h2         # include H2 probe test
    python calc_accessibility.py --viz              # render per-amine 3D plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from morfeus import BuriedVolume, SASA, read_xyz

# ------------------------------------------------------------
# Config (mirrors your existing project layout)
# ------------------------------------------------------------
try:
    from config import (
        RU_INDEX, N_CATALYST_ATOMS, TRAILING_FRAGMENT,
        XYZ_DIR, OUTPUT_DIR, ensure_dirs,
    )
except ImportError:
    # Standalone fallback defaults
    RU_INDEX = 1
    N_CATALYST_ATOMS = 63
    TRAILING_FRAGMENT = {
        "2A": 0, "3A": 0, "TS3A": 0, "H1": 0, "TSH2": 0, "H3": 0,
        "4A": 3, "TS5A": 4, "5A": 5,
    }
    XYZ_DIR = "../xyz_opt"
    OUTPUT_DIR = "../output"
    def ensure_dirs():
        os.makedirs(OUTPUT_DIR, exist_ok=True)

ACCESS_DIR = os.path.join(OUTPUT_DIR, "accessibility")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# Reference atoms (from your Ru-MACHO geometry)
AXIAL_HYDRIDE = 19   # H trans to amine site
P_LEFT = 2
P_RIGHT = 3
N_AMIDO = 5
C_CO = 18

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
# Probe radii (Å)
PROBE_SOLVENT = 1.40    # water/THF-sized probe
PROBE_H2 = 1.20         # approx. H2 van der Waals radius (kinetic diameter ~2.4 Å)
PROBE_SMALL = 0.75      # small probe for fine pocket resolution

# Sphere radii for buried-volume–style accessibility
R_POCKET_SMALL = 3.0    # tight coordination sphere (first-shell atoms only)
R_POCKET_STD = 3.5      # standard SambVca radius
R_POCKET_WIDE = 4.5     # wider sphere including second-shell contacts

# Grid spacing for voxel-based free volume (Å)
GRID_SPACING = 0.15

# H2 approach geometry: angular tolerance for "opposite-amine" hemisphere
HEMISPHERE_AXIS = "amine_N"   # define axis as Ru → amine_N


# ============================================================
# Helpers
# ============================================================

def load_registry():
    """Load structure registry. Falls back to scanning xyz_opt/ if missing."""
    json_path = os.path.join(CSV_DIR, "structure_registry.json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            return json.load(f)
    # fallback: rebuild
    try:
        from parse_structures import load_registry as _lr
        return _lr()
    except Exception:
        raise RuntimeError("Cannot find structure_registry.json. "
                           "Run parse_structures.py first.")


def find_amine_N(elements, coordinates, amine_atom_indices, ru_coord):
    """Find amine N atom (1-indexed) bonded to Ru. Returns (idx, dist)."""
    best_idx, best_dist = None, 1e9
    for idx in amine_atom_indices:
        elem = elements[idx - 1]
        is_N = (elem == 7) or (isinstance(elem, str) and elem.upper() == "N")
        if not is_N:
            continue
        dist = np.linalg.norm(coordinates[idx - 1] - ru_coord)
        if dist < best_dist:
            best_dist, best_idx = dist, idx
    return best_idx, best_dist


def get_vdw_radius(element):
    """Bondi vdW radii in Å. Fallback to 1.7 Å for unknown."""
    if isinstance(element, int):
        z_to_sym = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P",
                    16: "S", 17: "Cl", 44: "Ru"}
        element = z_to_sym.get(element, "C")
    table = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80,
             "S": 1.80, "Cl": 1.75, "Ru": 2.00, "F": 1.47}
    return table.get(str(element).upper() if isinstance(element, str) else element, 1.70)


# ============================================================
# Descriptor 1 — Ru SASA (total and decomposed)
# ============================================================

def compute_ru_sasa(elements, coordinates, cat_atoms, amine_atoms, extra_atoms=None,
                    probe=PROBE_SOLVENT):
    """
    Compute SASA of the Ru atom specifically, with decomposition by shielder.

    Returns dict with:
        Ru_SASA_full     — Ru accessible surface with all atoms present
        Ru_SASA_bare     — Ru SASA without any ligand (hypothetical max)
        Ru_SASA_cat_only — Ru SASA with only catalyst ligands (no amine)
        Ru_shield_amine  — surface area blocked by amine binding
        Ru_shield_cat    — surface area blocked by catalyst ligands
        Ru_access_frac   — Ru_SASA_full / Ru_SASA_bare  (0–1)
    """
    out = {}
    extra_atoms = extra_atoms or []

    # Full complex SASA
    try:
        sasa_full = SASA(elements, coordinates, radii_type="bondi", probe_radius=probe)
        out["Ru_SASA_full"] = sasa_full.atom_areas.get(RU_INDEX, 0.0)
    except Exception as e:
        out["Ru_SASA_full"] = np.nan
        print(f"    [WARN] Full SASA failed: {e}")

    # Bare Ru: isolated atom
    ru_radius = get_vdw_radius("Ru")
    out["Ru_SASA_bare"] = 4 * np.pi * (ru_radius + probe) ** 2

    # Ru + catalyst only (amine and extras removed)
    keep_cat = sorted(set(cat_atoms))
    elem_cat = [elements[i - 1] for i in keep_cat]
    coord_cat = np.array([coordinates[i - 1] for i in keep_cat])
    # Find the Ru index within the reduced list
    ru_in_cat = keep_cat.index(RU_INDEX) + 1
    try:
        sasa_cat = SASA(elem_cat, coord_cat, radii_type="bondi", probe_radius=probe)
        out["Ru_SASA_cat_only"] = sasa_cat.atom_areas.get(ru_in_cat, 0.0)
    except Exception:
        out["Ru_SASA_cat_only"] = np.nan

    # Shielding contributions
    if not np.isnan(out["Ru_SASA_full"]) and not np.isnan(out["Ru_SASA_cat_only"]):
        out["Ru_shield_amine"] = out["Ru_SASA_cat_only"] - out["Ru_SASA_full"]
        out["Ru_shield_cat"] = out["Ru_SASA_bare"] - out["Ru_SASA_cat_only"]
        out["Ru_access_frac"] = out["Ru_SASA_full"] / out["Ru_SASA_bare"]
    else:
        out["Ru_shield_amine"] = np.nan
        out["Ru_shield_cat"] = np.nan
        out["Ru_access_frac"] = np.nan

    return out


# ============================================================
# Descriptor 2 — Free %V_bur (complement of buried volume)
# ============================================================

def compute_free_buried_volume(elements, coordinates, radius=R_POCKET_STD):
    """
    Compute the COMPLEMENT of buried volume: free volume fraction within
    a sphere around Ru, counting ALL atoms (catalyst + amine + extras).

    excluded_atoms is empty so every atom contributes to blocking.
    Returns %free = 100 - %V_bur.
    """
    try:
        bv = BuriedVolume(
            elements, coordinates, RU_INDEX,
            excluded_atoms=[],       # count everything
            radius=radius,
            include_hs=True,
            radii_type="bondi",
            radii_scale=1.17,
        )
        pct_bur = bv.percent_buried_volume * 100 if bv.percent_buried_volume < 2 else bv.percent_buried_volume
        # morfeus returns fraction (0-1); normalize defensively
        if pct_bur > 100:
            pct_bur = bv.fraction_buried_volume * 100
        return {
            f"Vbur_total_R{radius}": pct_bur,
            f"Vfree_total_R{radius}": 100 - pct_bur,
            f"Vfree_abs_R{radius}": bv.free_volume,
        }
    except Exception as e:
        print(f"    [WARN] Free BV (R={radius}) failed: {e}")
        return {
            f"Vbur_total_R{radius}": np.nan,
            f"Vfree_total_R{radius}": np.nan,
            f"Vfree_abs_R{radius}": np.nan,
        }


# ============================================================
# Descriptor 3 — Hemisphere accessibility (directional)
# ============================================================

def compute_hemisphere_free_volume(elements, coordinates, ru_coord,
                                    amine_N_coord, radius=R_POCKET_STD,
                                    grid=GRID_SPACING):
    """
    Voxel-based free-volume calculation split into two hemispheres:
      - AMINE side:    hemisphere containing amine_N
      - OPPOSITE side: hemisphere where H2 can approach for assistive cleavage

    The axis runs Ru → amine_N. A grid point is on the AMINE side if its
    projection onto this axis is positive; opposite otherwise.

    Returns volumes in Å³ and fractions.
    """
    # Axis from Ru toward amine N (unit vector)
    axis = amine_N_coord - ru_coord
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 0.1:
        return {k: np.nan for k in
                ["Vfree_amine_side", "Vfree_opposite_side",
                 "Vfree_opposite_frac", "Vfree_asymmetry"]}
    axis_unit = axis / axis_norm

    # Build grid of points inside sphere of radius R around Ru
    n = int(2 * radius / grid) + 1
    lin = np.linspace(-radius, radius, n)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Keep only points inside sphere
    r_from_ru = np.linalg.norm(pts, axis=1)
    in_sphere = r_from_ru <= radius
    pts = pts[in_sphere]

    # Points are in Ru-local frame; translate to absolute for atom-distance test
    pts_abs = pts + ru_coord

    # For each grid point, is it occupied by any atom's vdW sphere?
    occupied = np.zeros(len(pts), dtype=bool)
    for i, elem in enumerate(elements):
        atom_pos = coordinates[i]
        r_vdw = get_vdw_radius(elem)
        # Skip Ru itself for the "free space" question
        if i + 1 == RU_INDEX:
            continue
        d = np.linalg.norm(pts_abs - atom_pos, axis=1)
        occupied |= (d < r_vdw)

    free_mask = ~occupied

    # Hemisphere assignment: positive dot product with axis_unit → amine side
    proj = pts @ axis_unit
    amine_side = proj > 0
    opposite_side = proj < 0

    voxel_vol = grid ** 3
    V_free_amine = free_mask[amine_side].sum() * voxel_vol
    V_free_opp = free_mask[opposite_side].sum() * voxel_vol
    V_total_amine_side = amine_side.sum() * voxel_vol
    V_total_opp_side = opposite_side.sum() * voxel_vol

    frac_opp = V_free_opp / V_total_opp_side if V_total_opp_side > 0 else np.nan
    frac_amine = V_free_amine / V_total_amine_side if V_total_amine_side > 0 else np.nan

    return {
        "Vfree_amine_side": V_free_amine,
        "Vfree_opposite_side": V_free_opp,
        "Vfree_amine_frac": frac_amine,
        "Vfree_opposite_frac": frac_opp,
        "Vfree_asymmetry": frac_opp - frac_amine,  # positive → assistive direction
    }


# ============================================================
# Descriptor 4 — H2 probe approach test
# ============================================================

def compute_h2_probe_test(elements, coordinates, ru_coord, probe=PROBE_H2,
                           shell_inner=2.0, shell_outer=4.0, n_test=2000):
    """
    Sample points in a spherical shell around Ru (between shell_inner and
    shell_outer) and check whether a sphere of radius `probe` can be placed
    there without overlapping any atom.

    Returns:
        H2_approach_frac      — fraction of shell points that are accessible
        H2_min_clearance      — min distance from Ru at which an accessible
                                 point exists (smaller = H2 can get closer)
        H2_n_accessible       — number of accessible probe positions
    """
    rng = np.random.default_rng(42)
    # Uniform sampling in spherical shell: r ∝ (u)^(1/3) scaled
    u = rng.random(n_test)
    r = (shell_inner ** 3 + u * (shell_outer ** 3 - shell_inner ** 3)) ** (1 / 3)
    theta = np.arccos(1 - 2 * rng.random(n_test))
    phi = 2 * np.pi * rng.random(n_test)
    offsets = np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=1)
    pts = ru_coord + offsets

    # For each probe position, check clearance against all heavy atoms
    accessible = np.ones(n_test, dtype=bool)
    for i, elem in enumerate(elements):
        if i + 1 == RU_INDEX:
            continue
        r_vdw = get_vdw_radius(elem)
        min_dist = r_vdw + probe
        d = np.linalg.norm(pts - coordinates[i], axis=1)
        accessible &= (d >= min_dist)

    n_acc = accessible.sum()
    frac = n_acc / n_test
    if n_acc > 0:
        min_r = r[accessible].min()
    else:
        min_r = np.nan

    return {
        "H2_approach_frac": frac,
        "H2_min_clearance": min_r,
        "H2_n_accessible": int(n_acc),
    }


# ============================================================
# Descriptor 5 — Ligand gap angles
# ============================================================

def compute_ligand_gaps(coordinates, ru_coord, ligand_indices):
    """
    For each known ligand atom directly bonded to Ru, compute the angular
    distribution. Returns the LARGEST angular gap — this is where H2 has
    most room to approach.

    ligand_indices should be 1-indexed atom numbers of ligands bonded to Ru.
    """
    vectors = []
    for idx in ligand_indices:
        v = coordinates[idx - 1] - ru_coord
        v /= np.linalg.norm(v)
        vectors.append(v)
    vectors = np.array(vectors)

    # Compute pairwise angles (degrees)
    angles = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos = np.clip(vectors[i] @ vectors[j], -1, 1)
            angles.append(np.degrees(np.arccos(cos)))

    if not angles:
        return {"ligand_max_gap_deg": np.nan, "ligand_mean_angle_deg": np.nan}

    return {
        "ligand_max_gap_deg": float(max(angles)),
        "ligand_mean_angle_deg": float(np.mean(angles)),
    }


# ============================================================
# Master: compute all accessibility descriptors for one structure
# ============================================================

def analyze_structure(entry, do_h2_probe=True):
    """Compute all accessibility descriptors for one XYZ file."""
    filepath = entry["filepath"]
    label = entry["label"]

    try:
        elements, coordinates = read_xyz(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return None

    ru_coord = coordinates[RU_INDEX - 1]
    cat_atoms = entry["cat_atoms"]
    amine_atoms = entry["amine_atoms"]
    extra_atoms = entry.get("extra_atoms", [])

    result = {
        "label": label,
        "state": entry["state"],
        "amine": entry["amine"],
    }

    # ---- Descriptor 1: Ru SASA ----
    sasa_results = compute_ru_sasa(elements, coordinates, cat_atoms, amine_atoms,
                                    extra_atoms, probe=PROBE_SOLVENT)
    result.update(sasa_results)

    # ---- Descriptor 2: Free %V_bur at multiple radii ----
    for r in [R_POCKET_SMALL, R_POCKET_STD, R_POCKET_WIDE]:
        result.update(compute_free_buried_volume(elements, coordinates, radius=r))

    # ---- Descriptor 3: Hemisphere split ----
    amine_N_idx, ru_n_dist = find_amine_N(elements, coordinates, amine_atoms, ru_coord)
    if amine_N_idx is not None:
        result["amine_N_index"] = amine_N_idx
        result["Ru_N_distance"] = ru_n_dist
        amine_N_coord = coordinates[amine_N_idx - 1]
        hemi = compute_hemisphere_free_volume(elements, coordinates, ru_coord,
                                               amine_N_coord, radius=R_POCKET_STD)
        result.update(hemi)
    else:
        result["amine_N_index"] = np.nan
        result["Ru_N_distance"] = np.nan
        for k in ["Vfree_amine_side", "Vfree_opposite_side",
                  "Vfree_amine_frac", "Vfree_opposite_frac", "Vfree_asymmetry"]:
            result[k] = np.nan

    # ---- Descriptor 4: H2 probe test ----
    if do_h2_probe:
        result.update(compute_h2_probe_test(elements, coordinates, ru_coord))

    # ---- Descriptor 5: Ligand gap angles ----
    ligand_idx = [P_LEFT, P_RIGHT, N_AMIDO, AXIAL_HYDRIDE, C_CO]
    if amine_N_idx is not None:
        ligand_idx.append(amine_N_idx)
    ligand_idx = [i for i in ligand_idx if i <= len(elements)]
    result.update(compute_ligand_gaps(coordinates, ru_coord, ligand_idx))

    return result


# ============================================================
# Plotting
# ============================================================

def _load_yield_map():
    try:
        from config import MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS
        return MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS
    except ImportError:
        return {}, {}, {"high": "#2ca02c", "medium": "#ff7f0e",
                        "low": "#d62728", "unknown": "#7f7f7f"}


def plot_accessibility_vs_yield(df, state="2A"):
    """Key plot: Ru accessibility vs MeOH yield."""
    MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS = _load_yield_map()

    sub = df[df["state"] == state].copy()
    sub["Yield"] = sub["amine"].map(MEOH_YIELD)
    sub["cat"] = sub["amine"].map(YIELD_CATEGORY).fillna("unknown")
    sub = sub.dropna(subset=["Yield"])
    if len(sub) < 3:
        print(f"  [SKIP] Not enough data for state {state}")
        return

    descriptors_to_plot = [
        ("Ru_SASA_full", "Ru SASA (Å²)"),
        ("Ru_access_frac", "Ru accessible fraction"),
        (f"Vfree_total_R{R_POCKET_STD}", f"%V_free (R={R_POCKET_STD} Å)"),
        ("Vfree_opposite_frac", "Free volume fraction (opposite-amine side)"),
        ("Vfree_asymmetry", "Opposite − amine free-vol asymmetry"),
        ("H2_approach_frac", "H₂ probe approach fraction"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, descriptors_to_plot):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        data = sub.dropna(subset=[col])
        colors = [CATEGORY_COLORS.get(c, "#7f7f7f") for c in data["cat"]]
        ax.scatter(data[col], data["Yield"], c=colors, s=60,
                   edgecolors="k", linewidth=0.5, zorder=3)
        for _, row in data.iterrows():
            ax.annotate(row["amine"], (row[col], row["Yield"]),
                        fontsize=6, ha="left",
                        xytext=(3, 3), textcoords="offset points", alpha=0.75)
        # Regression
        if len(data) >= 3:
            try:
                from scipy import stats
                x, y = data[col].values, data["Yield"].values
                slope, intercept, r, p, _ = stats.linregress(x, y)
                xl = np.linspace(x.min(), x.max(), 50)
                ax.plot(xl, slope * xl + intercept, "--", color="gray", alpha=0.5)
                ax.text(0.05, 0.95, f"R² = {r ** 2:.3f}\np = {p:.1e}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.8))
            except Exception:
                pass
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("MeOH yield (%)", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Ru-centric accessibility vs methanol yield — {state} state",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ACCESS_DIR, f"accessibility_vs_yield_{state}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: accessibility_vs_yield_{state}.png")


def plot_hemisphere_comparison(df, state="2A"):
    """Compare amine-side vs opposite-side free volume across amines."""
    MEOH_YIELD, YIELD_CATEGORY, CATEGORY_COLORS = _load_yield_map()

    sub = df[df["state"] == state].copy()
    sub["Yield"] = sub["amine"].map(MEOH_YIELD)
    sub = sub.dropna(subset=["Vfree_opposite_frac", "Vfree_amine_frac"])
    if len(sub) == 0:
        return
    sub = sub.sort_values("Yield", ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.4), 5))
    x = np.arange(len(sub))
    width = 0.4
    ax.bar(x - width / 2, sub["Vfree_amine_frac"], width,
           label="Amine side (blocked)", color="#d62728", edgecolor="k", lw=0.3)
    ax.bar(x + width / 2, sub["Vfree_opposite_frac"], width,
           label="Opposite side (H₂ approach)", color="#2ca02c", edgecolor="k", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["amine"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Free volume fraction", fontsize=11)
    ax.set_title(f"Directional Ru accessibility — {state} (sorted by yield)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ACCESS_DIR, f"hemisphere_comparison_{state}.png"),
                dpi=300)
    plt.close(fig)
    print(f"  Saved: hemisphere_comparison_{state}.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", nargs="*", default=None,
                        help="Limit to specific states (e.g. --state 2A 3A)")
    parser.add_argument("--no-h2-probe", action="store_true",
                        help="Skip H2 probe test (faster)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    ensure_dirs()
    os.makedirs(ACCESS_DIR, exist_ok=True)

    print("=" * 60)
    print("Ru-Centric Accessibility Analysis")
    print("=" * 60)
    print(f"  Probe (solvent):  {PROBE_SOLVENT} Å")
    print(f"  Probe (H2):       {PROBE_H2} Å")
    print(f"  Pocket radius:    {R_POCKET_STD} Å")
    print(f"  Grid spacing:     {GRID_SPACING} Å")
    print()

    registry = load_registry()
    results = []

    for entry in registry:
        if args.state and entry["state"] not in args.state:
            continue
        print(f"  Processing {entry['label']}...")
        res = analyze_structure(entry, do_h2_probe=not args.no_h2_probe)
        if res is None:
            continue
        results.append(res)
        print(f"    Ru SASA: {res.get('Ru_SASA_full', np.nan):.2f} Å²  "
              f"Access frac: {res.get('Ru_access_frac', np.nan):.3f}  "
              f"V_free(opp): {res.get('Vfree_opposite_frac', np.nan):.3f}  "
              f"H₂ approach: {res.get('H2_approach_frac', np.nan):.3f}")

    df = pd.DataFrame(results)
    outpath = os.path.join(CSV_DIR, "accessibility_results.csv")
    os.makedirs(CSV_DIR, exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(df)} results to {outpath}")

    # Plots
    if not args.no_plots and len(df) > 0:
        print("\nGenerating plots...")
        for state in df["state"].unique():
            plot_accessibility_vs_yield(df, state=state)
            plot_hemisphere_comparison(df, state=state)

    # Summary
    print("\n" + "=" * 60)
    print("Top descriptors summary (state=2A)")
    print("=" * 60)
    sub = df[df["state"] == "2A"]
    if len(sub) > 0:
        key_cols = ["Ru_SASA_full", "Ru_access_frac",
                    f"Vfree_total_R{R_POCKET_STD}",
                    "Vfree_opposite_frac", "Vfree_asymmetry",
                    "H2_approach_frac", "ligand_max_gap_deg"]
        for col in key_cols:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals):
                    print(f"  {col:>30s}: "
                          f"mean={vals.mean():7.3f}  "
                          f"std={vals.std():7.3f}  "
                          f"range=[{vals.min():7.3f}, {vals.max():7.3f}]")


if __name__ == "__main__":
    main()
