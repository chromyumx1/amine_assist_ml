# Morfeus Post-Analysis Code

## Quick Start

```bash
# 1. Install dependencies
pip install morfeus-ml matplotlib numpy pandas scipy

# 2. Verify your structures first (ALWAYS do this before anything else)
python verify_structures.py --verbose

# 3. Build the structure registry
python parse_structures.py

# 4. Run everything
python run_all.py

# Or run step-by-step:
python calc_buried_volume.py                # %V_bur + quadrants
python calc_buried_volume.py --maps         # steric map images
python calc_buried_volume.py --sensitivity  # radius sensitivity
python calc_descriptors.py                  # SASA + Sterimol
python plot_results.py                      # all plots
```

## Before You Run

### 1. Fill in experimental data
Edit `config.py` → `MEOH_YIELD` dictionary with your actual methanol yields for all 26 amines.

### 2. Verify atom indexing
Run `python verify_structures.py --state 2A --verbose` and confirm:
- Atom 1 = Ru
- Atoms 2, 3 = P
- Atom 5 = N (amido)
- Atom 18 = C (CO carbon)
- Atom 19 = H (axial hydride)
- Atom 64+ = amine fragment

### 3. Check amine name matching
Run `python parse_structures.py` and check for any "unmatched" files. If your filenames use unusual abbreviations, add them to `KNOWN_AMINES` and `AMINE_CANONICAL` in `config.py`.

## Expected Directory Structure

```
project_root/
├── xyz_opt/
│   ├── 2A/    (26 .xyz files)
│   ├── 3A/
│   ├── 4A/    (note: last 3 atoms = CO2)
│   ├── 5A/    (note: last 5 atoms = CO2 + H2)
│   ├── H1/
│   ├── H3/
│   ├── TS3A/
│   ├── TS5A/  (note: last 4 atoms = OCOH)
│   └── TSH2/
├── code/
│   ├── config.py               # All settings (edit this)
│   ├── parse_structures.py     # Scan & register structures
│   ├── verify_structures.py    # Atom indexing checks
│   ├── calc_buried_volume.py   # %V_bur + steric maps
│   ├── calc_descriptors.py     # SASA + Sterimol
│   ├── plot_results.py         # All correlation plots
│   ├── run_all.py              # Master runner
│   └── README.md               # This file
└── output/                     # Created automatically
    ├── csv/                    # Numerical results
    ├── plots/                  # Correlation plots
    └── steric_maps/            # Steric map images per state
```

## Output Files

### CSV
| File | Description |
|---|---|
| `structure_registry.csv` | All detected structures with metadata |
| `Vbur_results.csv` | %V_bur + quadrants for all structures |
| `Vbur_sensitivity.csv` | %V_bur at multiple sphere radii |
| `descriptors_sasa_sterimol.csv` | SASA + Sterimol for all |
| `all_descriptors_merged.csv` | Everything merged by label |

### Plots
| File | Description |
|---|---|
| `Vbur_vs_yield_2A.png` | %V_bur vs MeOH yield (key plot for paper) |
| `Vbur_across_states.png` | How steric footprint changes along pathway |
| `quadrant_analysis_2A.png` | Directional steric profile |
| `SASA_vs_yield_2A.png` | Amine accessibility vs yield |
| `Sterimol_vs_yield_2A.png` | Amine shape parameters vs yield |
| `Vbur_vs_Sterimol_2A.png` | Bound steric vs free amine shape |
| `RuN_dist_vs_yield_2A.png` | Bond distance vs yield |
| `sensitivity_2A.png` | Radius sensitivity check |
| `descriptor_correlation_heatmap.png` | Cross-descriptor correlations |

## Special State Handling

| State | Amine atoms | Extra fragment | Notes |
|---|---|---|---|
| 2A, 3A, TS3A, H1, TSH2, H3 | 64 → end | None | Standard |
| 4A | 64 → (N−3) | Last 3 = CO₂ | Carbamate formation |
| TS5A | 64 → (N−4) | Last 4 = OCOH | Carbamic acid TS |
| 5A | 64 → (N−5) | Last 5 = CO₂+H₂ | Post-dissociation |

Use `--include-extra` flag in `calc_buried_volume.py` to include trailing fragments with the amine (measures carbamate steric effect rather than pure amine).

## Citation
If using %V_bur in publication, cite:
- Falivene et al. *Organometallics* **2016**, 35, 2286–2293
- Morfeus package: https://doi.org/10.5281/zenodo.6114976
