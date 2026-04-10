"""
run_all.py — Master script to run all morfeus analyses.

Usage:
    python run_all.py              # everything
    python run_all.py --state 2A   # only state 2A
    python run_all.py --skip-maps  # skip steric map generation (faster)
"""

import argparse
import time
from config import ensure_dirs
from parse_structures import build_registry


def main():
    parser = argparse.ArgumentParser(description="Run all morfeus analyses")
    parser.add_argument("--state", nargs="*", default=None,
                        help="Limit to specific states")
    parser.add_argument("--skip-maps", action="store_true",
                        help="Skip steric map image generation")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip radius sensitivity analysis")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip correlation plots")
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()

    # Step 1: Build structure registry
    print("=" * 60)
    print("STEP 1: Building structure registry")
    print("=" * 60)
    build_registry()

    # Step 2: Buried volume
    print("\n" + "=" * 60)
    print("STEP 2: Buried volume calculations")
    print("=" * 60)
    from calc_buried_volume import run_all as vbur_run, run_sensitivity, generate_steric_maps
    import os
    from config import CSV_DIR

    df_vbur = vbur_run(states=args.state)
    df_vbur.to_csv(os.path.join(CSV_DIR, "Vbur_results.csv"), index=False)

    # Step 2b: Steric maps
    if not args.skip_maps:
        print("\n--- Generating steric maps ---")
        generate_steric_maps(states=args.state)

    # Step 2c: Sensitivity
    if not args.skip_sensitivity:
        print("\n--- Sensitivity analysis ---")
        df_sens = run_sensitivity(states=args.state)
        df_sens.to_csv(os.path.join(CSV_DIR, "Vbur_sensitivity.csv"), index=False)

    # Step 3: SASA & Sterimol
    print("\n" + "=" * 60)
    print("STEP 3: SASA & Sterimol descriptors")
    print("=" * 60)
    from calc_descriptors import run_all as desc_run
    df_desc = desc_run(states=args.state)
    df_desc.to_csv(os.path.join(CSV_DIR, "descriptors_sasa_sterimol.csv"), index=False)

    # Step 4: Merge all results
    print("\n" + "=" * 60)
    print("STEP 4: Merging results")
    print("=" * 60)
    df_merged = df_vbur.merge(
        df_desc, on=["label", "state", "amine"], how="outer", suffixes=("", "_desc")
    )
    merged_path = os.path.join(CSV_DIR, "all_descriptors_merged.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"Merged {len(df_merged)} entries → {merged_path}")

    # Step 5: Plots
    if not args.skip_plots:
        print("\n" + "=" * 60)
        print("STEP 5: Generating plots")
        print("=" * 60)
        from plot_results import generate_all_plots
        generate_all_plots(df_merged)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All done in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
