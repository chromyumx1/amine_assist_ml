"""
parse_structures.py — Scan xyz_opt/ and build a registry of all structures.

Each entry contains:
    state       : str   e.g. "2A", "H1", "TSH2"
    amine       : str   canonical amine name e.g. "EDA", "PIP"
    label       : str   e.g. "2A_EDA"
    filepath    : str   absolute path to .xyz file
    n_atoms     : int   total atoms in structure
    cat_atoms   : list  atom indices belonging to catalyst (1-indexed)
    amine_atoms : list  atom indices belonging to amine (1-indexed)
    extra_atoms : list  atom indices for trailing fragments (CO2, OCOH, H2)
"""

import os
import re
import json
import pandas as pd
from config import (
    XYZ_DIR, CSV_DIR, STATES, N_CATALYST_ATOMS,
    KNOWN_AMINES, AMINE_CANONICAL, TRAILING_FRAGMENT, ensure_dirs,
)


def count_atoms(filepath):
    """Read number of atoms from first line of xyz file."""
    with open(filepath, "r") as f:
        return int(f.readline().strip())


def identify_amine(filename):
    """
    Extract amine name from filename by searching for known abbreviations.
    Case-insensitive matching; returns canonical name or None.
    """
    name_upper = filename.upper()
    # Strip extension and common prefixes
    name_clean = os.path.splitext(filename)[0].upper()

    for abbrev in KNOWN_AMINES:
        # Build pattern: the abbreviation surrounded by non-alpha or boundaries
        # This prevents "EDA" matching inside "DMEDA"
        pattern = r'(?:^|[^A-Z])' + re.escape(abbrev.upper()) + r'(?:[^A-Z]|$)'
        if re.search(pattern, name_clean):
            return AMINE_CANONICAL.get(abbrev, abbrev)

    # Fallback: try simple substring (less strict)
    for abbrev in KNOWN_AMINES:
        if abbrev.upper() in name_clean:
            return AMINE_CANONICAL.get(abbrev, abbrev)

    return None


def build_atom_lists(state, n_atoms):
    """
    Return (cat_atoms, amine_atoms, extra_atoms) as 1-indexed lists.

    cat_atoms   : atoms 1..63 (catalyst)
    amine_atoms : atoms 64..(n_atoms - trailing)
    extra_atoms : trailing small-molecule fragment atoms
    """
    trailing = TRAILING_FRAGMENT.get(state, 0)
    cat_atoms = list(range(1, N_CATALYST_ATOMS + 1))
    last_amine = n_atoms - trailing
    amine_atoms = list(range(N_CATALYST_ATOMS + 1, last_amine + 1))
    extra_atoms = list(range(last_amine + 1, n_atoms + 1))
    return cat_atoms, amine_atoms, extra_atoms


def scan_all():
    """
    Scan xyz_opt/ and return list of structure dicts.
    """
    structures = []
    unmatched = []

    for state in STATES:
        state_dir = os.path.join(XYZ_DIR, state)
        if not os.path.isdir(state_dir):
            print(f"[WARN] Directory not found: {state_dir}")
            continue

        xyz_files = sorted([
            f for f in os.listdir(state_dir)
            if f.lower().endswith(".xyz")
        ])

        for fname in xyz_files:
            filepath = os.path.join(state_dir, fname)
            amine = identify_amine(fname)

            if amine is None:
                unmatched.append((state, fname))
                continue

            n_atoms = count_atoms(filepath)
            cat_atoms, amine_atoms, extra_atoms = build_atom_lists(state, n_atoms)

            structures.append({
                "state": state,
                "amine": amine,
                "label": f"{state}_{amine}",
                "filepath": filepath,
                "filename": fname,
                "n_atoms": n_atoms,
                "n_cat_atoms": len(cat_atoms),
                "n_amine_atoms": len(amine_atoms),
                "n_extra_atoms": len(extra_atoms),
                "cat_atoms": cat_atoms,
                "amine_atoms": amine_atoms,
                "extra_atoms": extra_atoms,
            })

    return structures, unmatched


def build_registry():
    """
    Build and save structure registry.
    Returns DataFrame and prints summary.
    """
    ensure_dirs()
    structures, unmatched = scan_all()

    if unmatched:
        print(f"\n[WARNING] {len(unmatched)} files could not be matched to an amine:")
        for state, fname in unmatched:
            print(f"  {state}/{fname}")
        print("  → Edit KNOWN_AMINES in config.py or rename the files.\n")

    df = pd.DataFrame(structures)

    # Summary
    print(f"Registry built: {len(df)} structures")
    print(f"  States found: {sorted(df['state'].unique())}")
    print(f"  Amines found: {sorted(df['amine'].unique())}")
    print(f"  Amines per state:")
    for state in STATES:
        sub = df[df["state"] == state]
        print(f"    {state:5s}: {len(sub)} amines")

    # Save
    # CSV (without the list columns — those are for internal use)
    df_save = df.drop(columns=["cat_atoms", "amine_atoms", "extra_atoms"])
    csv_path = os.path.join(CSV_DIR, "structure_registry.csv")
    df_save.to_csv(csv_path, index=False)
    print(f"\nRegistry saved to {csv_path}")

    # Also save full registry as JSON for programmatic access
    json_path = os.path.join(CSV_DIR, "structure_registry.json")
    # Convert lists for JSON serialization
    records = []
    for s in structures:
        rec = {k: v for k, v in s.items()}
        records.append(rec)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Full registry (with atom lists) saved to {json_path}")

    return df, structures


def load_registry():
    """Load registry from JSON file."""
    json_path = os.path.join(CSV_DIR, "structure_registry.json")
    if not os.path.exists(json_path):
        print("Registry not found. Building...")
        _, structures = build_registry()
        return structures
    with open(json_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    build_registry()
