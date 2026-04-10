"""
calc_descriptors.py — Compute Sterimol, SASA, and cone angle descriptors.

These are computed on the amine fragment within the complex geometry,
giving "bound-state" steric descriptors.

Usage:
    python calc_descriptors.py                  # all states
    python calc_descriptors.py --state 2A 3A    # specific states
"""

import os
import argparse
import numpy as np
import pandas as pd

from morfeus import BuriedVolume, SASA, Sterimol, read_xyz

from config import (
    RU_INDEX, N_CATALYST_ATOMS, INCLUDE_H,
    CSV_DIR, ensure_dirs,
)
from parse_structures import load_registry


def find_amine_N_bonded_to_Ru(elements, coordinates, entry):
    """
    Find the amine nitrogen atom that is directly bonded to Ru.
    This is the first N atom in the amine fragment (atom index > 63)
    that is closest to Ru.
    """
    ru_coord = coordinates[RU_INDEX - 1]  # 0-indexed for numpy

    best_idx = None
    best_dist = 999.0

    for atom_idx in entry["amine_atoms"]:
        i = atom_idx - 1  # 0-indexed
        if elements[i] == 7 or (isinstance(elements[i], str) and elements[i].upper() == "N"):
            # Check if it's nitrogen
            dist = np.linalg.norm(coordinates[i] - ru_coord)
            if dist < best_dist:
                best_dist = dist
                best_idx = atom_idx  # keep 1-indexed

    return best_idx, best_dist


def find_neighbor_of_atom(elements, coordinates, atom_idx_1indexed, exclude_indices=None):
    """
    Find the nearest heavy atom neighbor of a given atom.
    Returns 1-indexed atom index.
    """
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)

    ref_coord = coordinates[atom_idx_1indexed - 1]
    best_idx = None
    best_dist = 999.0

    for i in range(len(elements)):
        idx = i + 1  # 1-indexed
        if idx == atom_idx_1indexed or idx in exclude_indices:
            continue
        # Skip H
        elem = elements[i]
        if elem == 1 or (isinstance(elem, str) and elem.upper() == "H"):
            continue
        dist = np.linalg.norm(coordinates[i] - ref_coord)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


def compute_descriptors(entry):
    """
    Compute SASA and Sterimol descriptors for one structure.
    Returns dict of results.
    """
    filepath = entry["filepath"]
    label = entry["label"]

    try:
        elements, coordinates = read_xyz(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return None

    result = {
        "label": label,
        "state": entry["state"],
        "amine": entry["amine"],
    }

    # ----- SASA on the full complex -----
    try:
        sasa = SASA(elements, coordinates)

        # Total SASA
        result["SASA_total"] = sasa.area

        # SASA of amine fragment only
        amine_sasa = sum(
            sasa.atom_areas.get(idx, 0.0) for idx in entry["amine_atoms"]
        )
        result["SASA_amine"] = amine_sasa

        # SASA of catalyst fragment
        cat_sasa = sum(
            sasa.atom_areas.get(idx, 0.0)
            for idx in entry["cat_atoms"]
        )
        result["SASA_catalyst"] = cat_sasa

        # SASA of amine N bonded to Ru
        amine_N, ru_n_dist = find_amine_N_bonded_to_Ru(elements, coordinates, entry)
        if amine_N is not None:
            result["amine_N_index"] = amine_N
            result["Ru_N_distance"] = ru_n_dist
            result["SASA_amine_N"] = sasa.atom_areas.get(amine_N, 0.0)

            # Ratio: amine N exposure relative to total amine SASA
            if amine_sasa > 0:
                result["SASA_N_ratio"] = result["SASA_amine_N"] / amine_sasa
            else:
                result["SASA_N_ratio"] = np.nan
        else:
            result["amine_N_index"] = np.nan
            result["Ru_N_distance"] = np.nan
            result["SASA_amine_N"] = np.nan
            result["SASA_N_ratio"] = np.nan

    except Exception as e:
        print(f"  [WARN] SASA failed for {label}: {e}")
        for key in ["SASA_total", "SASA_amine", "SASA_catalyst",
                     "amine_N_index", "Ru_N_distance", "SASA_amine_N", "SASA_N_ratio"]:
            result[key] = np.nan

    # ----- Sterimol on amine fragment within complex -----
    # Sterimol needs: atom_1 (donor N), atom_2 (its neighbor away from Ru)
    try:
        amine_N, _ = find_amine_N_bonded_to_Ru(elements, coordinates, entry)
        if amine_N is not None:
            # Find amine N's neighbor WITHIN the amine fragment (not Ru)
            amine_neighbor = find_neighbor_of_atom(
                elements, coordinates, amine_N,
                exclude_indices=set(entry["cat_atoms"])
            )
            if amine_neighbor is not None:
                # Sterimol on the amine fragment only
                sterimol = Sterimol(
                    elements, coordinates,
                    dummy_index=amine_N,
                    attached_index=amine_neighbor,
                )
                result["Sterimol_L"] = sterimol.L_value
                result["Sterimol_B1"] = sterimol.B_1_value
                result["Sterimol_B5"] = sterimol.B_5_value
            else:
                result["Sterimol_L"] = np.nan
                result["Sterimol_B1"] = np.nan
                result["Sterimol_B5"] = np.nan
        else:
            result["Sterimol_L"] = np.nan
            result["Sterimol_B1"] = np.nan
            result["Sterimol_B5"] = np.nan
    except Exception as e:
        print(f"  [WARN] Sterimol failed for {label}: {e}")
        result["Sterimol_L"] = np.nan
        result["Sterimol_B1"] = np.nan
        result["Sterimol_B5"] = np.nan

    return result


def run_all(states=None):
    """Compute descriptors for all structures."""
    registry = load_registry()
    results = []

    for entry in registry:
        if states and entry["state"] not in states:
            continue

        result = compute_descriptors(entry)
        if result is None:
            continue
        results.append(result)
        vbur_str = ""
        sasa_str = f"SASA_amine={result.get('SASA_amine', 0):.1f}"
        sterimol_str = (
            f"L={result.get('Sterimol_L', 0):.2f} "
            f"B1={result.get('Sterimol_B1', 0):.2f} "
            f"B5={result.get('Sterimol_B5', 0):.2f}"
        )
        print(f"  {result['label']:>20s}  {sasa_str}  {sterimol_str}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="SASA & Sterimol descriptors")
    parser.add_argument("--state", nargs="*", default=None)
    args = parser.parse_args()

    ensure_dirs()

    print("\n=== SASA & Sterimol Calculations ===\n")
    df = run_all(states=args.state)

    outpath = os.path.join(CSV_DIR, "descriptors_sasa_sterimol.csv")
    df.to_csv(outpath, index=False)
    print(f"\nSaved {len(df)} results to {outpath}")


if __name__ == "__main__":
    main()
