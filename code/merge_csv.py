"""
merge_csv.py — Merge Vbur_results.csv and descriptors_sasa_sterimol.csv
into all_descriptors_merged.csv, which is required by plot_results.py.

Usage:
    python merge_csv.py

Expects these files to already exist in output/csv/:
    Vbur_results.csv              (from calc_buried_volume.py)
    descriptors_sasa_sterimol.csv (from calc_descriptors.py)
"""

import os
import sys
import pandas as pd
from config import CSV_DIR, ensure_dirs


def merge():
    ensure_dirs()

    vbur_path = os.path.join(CSV_DIR, "Vbur_results.csv")
    desc_path = os.path.join(CSV_DIR, "descriptors_sasa_sterimol.csv")
    out_path  = os.path.join(CSV_DIR, "all_descriptors_merged.csv")

    missing = [p for p in (vbur_path, desc_path) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] Missing input file: {p}")
        print("Run calc_buried_volume.py and calc_descriptors.py first.")
        sys.exit(1)

    df_vbur = pd.read_csv(vbur_path)
    df_desc = pd.read_csv(desc_path)

    df_merged = df_vbur.merge(
        df_desc, on=["label", "state", "amine"], how="outer", suffixes=("", "_desc")
    )
    df_merged.to_csv(out_path, index=False)
    print(f"Merged {len(df_merged)} entries → {out_path}")


if __name__ == "__main__":
    merge()
