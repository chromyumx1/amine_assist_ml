"""
verify_structures.py — Sanity-check atom indexing before running calculations.

Run this FIRST to confirm:
  1. Atom 1 is Ru in every structure
  2. Atoms 2,3 are P
  3. Atom 5 is N (amido)
  4. Atom 18 is C (CO)
  5. Atom 19 is H (hydride)
  6. Amine fragment starts at atom 64
  7. The amine N bonded to Ru is correctly detected

Usage:
    python verify_structures.py             # check all
    python verify_structures.py --state 2A  # check only 2A
    python verify_structures.py --verbose   # print full atom list for first structure
"""

import os
import argparse
import numpy as np
from morfeus import read_xyz

from config import (
    RU_INDEX, P_LEFT, P_RIGHT, N_AMIDO, C_CO, AXIAL_HYDRIDE,
    N_CATALYST_ATOMS, XYZ_DIR, STATES, ensure_dirs,
)
from parse_structures import load_registry


# Expected elements at reference positions (atomic number or symbol)
EXPECTED = {
    RU_INDEX:       ("Ru", 44),
    P_LEFT:         ("P",  15),
    P_RIGHT:        ("P",  15),
    N_AMIDO:        ("N",   7),
    C_CO:           ("C",   6),
    AXIAL_HYDRIDE:  ("H",   1),
}


def elem_matches(elem, expected_sym, expected_num):
    """Check if element matches expected (handle both str and int)."""
    if isinstance(elem, str):
        return elem.upper() == expected_sym.upper()
    return elem == expected_num


def verify_one(entry, verbose=False):
    """Verify atom indexing for one structure. Returns (ok, messages)."""
    filepath = entry["filepath"]
    label = entry["label"]
    messages = []

    try:
        elements, coordinates = read_xyz(filepath)
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    n_atoms = len(elements)
    ok = True

    # Check reference atoms
    for atom_idx, (exp_sym, exp_num) in EXPECTED.items():
        if atom_idx > n_atoms:
            messages.append(f"  Atom {atom_idx} out of range (only {n_atoms} atoms)")
            ok = False
            continue
        actual = elements[atom_idx - 1]
        if not elem_matches(actual, exp_sym, exp_num):
            messages.append(
                f"  Atom {atom_idx}: expected {exp_sym} but got {actual}"
            )
            ok = False

    # Check that amine fragment exists
    if n_atoms <= N_CATALYST_ATOMS:
        messages.append(f"  Only {n_atoms} atoms — no amine fragment!")
        ok = False
    else:
        n_amine = len(entry["amine_atoms"])
        n_extra = len(entry["extra_atoms"])
        messages.append(f"  Amine atoms: {n_amine}, Extra atoms: {n_extra}")

        # Find amine N closest to Ru
        ru_coord = coordinates[RU_INDEX - 1]
        amine_nitrogens = []
        for idx in entry["amine_atoms"]:
            elem = elements[idx - 1]
            if elem_matches(elem, "N", 7):
                dist = np.linalg.norm(coordinates[idx - 1] - ru_coord)
                amine_nitrogens.append((idx, dist))

        if amine_nitrogens:
            amine_nitrogens.sort(key=lambda x: x[1])
            closest_N, closest_dist = amine_nitrogens[0]
            messages.append(f"  Amine N → Ru: atom {closest_N}, d = {closest_dist:.3f} Å")
            if closest_dist > 3.0:
                messages.append(f"  [WARN] Ru–N distance > 3.0 Å — may not be bonded")
        else:
            messages.append(f"  [WARN] No nitrogen found in amine fragment!")

    # Verbose: print all atoms
    if verbose:
        messages.append(f"\n  Full atom list ({n_atoms} atoms):")
        messages.append(f"  {'Idx':>4s} {'Elem':>5s} {'X':>10s} {'Y':>10s} {'Z':>10s}  Fragment")
        for i in range(n_atoms):
            idx = i + 1
            elem = elements[i]
            if isinstance(elem, int):
                from morfeus.utils import convert_elements
                sym = convert_elements([elem], output="symbols")[0]
            else:
                sym = elem
            x, y, z = coordinates[i]
            if idx <= N_CATALYST_ATOMS:
                frag = "CAT"
            elif idx in entry.get("extra_atoms", []):
                frag = "EXTRA"
            else:
                frag = "AMINE"
            messages.append(
                f"  {idx:4d} {sym:>5s} {x:10.4f} {y:10.4f} {z:10.4f}  {frag}"
            )

    return ok, messages


def main():
    parser = argparse.ArgumentParser(description="Verify structure atom indexing")
    parser.add_argument("--state", nargs="*", default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full atom list for first structure per state")
    args = parser.parse_args()

    registry = load_registry()

    total = 0
    passed = 0
    failed = 0
    verbose_done = set()

    for entry in registry:
        if args.state and entry["state"] not in args.state:
            continue

        total += 1
        do_verbose = args.verbose and entry["state"] not in verbose_done

        ok, msgs = verify_one(entry, verbose=do_verbose)

        if do_verbose:
            verbose_done.add(entry["state"])

        status = "✓" if ok else "✗"
        print(f"{status} {entry['label']:>20s}  ({entry['filename']})")
        for m in msgs:
            print(m)

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed > 0:
        print("⚠  Fix failed entries before running calculations!")
        print("   Common issues:")
        print("   - Atom ordering differs between structures")
        print("   - Wrong N_CATALYST_ATOMS count in config.py")
        print("   - File format issues")
    else:
        print("✓  All structures verified. Safe to proceed.")


if __name__ == "__main__":
    main()
