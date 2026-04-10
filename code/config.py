"""
config.py — Central configuration for morfeus post-analysis.

All atom indices are 1-indexed (morfeus convention).
Catalyst fragment: atoms 1–63 (universal across all states).
Amine fragment: atoms 64+ (with state-specific exceptions).

Reference atoms in catalyst:
    1  = Ru (metal center)
    2  = P (phosphorus ligand, left)
    3  = P (phosphorus ligand, right)
    5  = N (amido ligand on PNP backbone)
    18 = C (carbonyl carbon of CO ligand, trans to amido)
    19 = H (axial hydride on Ru)
"""

import os

# ============================================================
# Directory layout
# ============================================================
# Adjust BASE_DIR to wherever your project root is.
# Expected structure:
#   BASE_DIR/
#     xyz_opt/
#       2A/  3A/  4A/  5A/  H1/  H3/  TS3A/  TS5A/  TSH2/
#     code/
#       config.py  (this file)
#       ...
#     output/
#       csv/  plots/  steric_maps/

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XYZ_DIR = os.path.join(BASE_DIR, "xyz_opt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
STERIC_MAP_DIR = os.path.join(OUTPUT_DIR, "steric_maps")

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, CSV_DIR, PLOT_DIR, STERIC_MAP_DIR]:
        os.makedirs(d, exist_ok=True)

# ============================================================
# Atom indexing (1-indexed for morfeus)
# ============================================================
RU_INDEX = 1
N_CATALYST_ATOMS = 63  # atoms 1..63 belong to catalyst

# Reference atoms for orientation
AXIAL_HYDRIDE = 19     # H trans to vacant/amine site
P_LEFT = 2             # phosphorus (left hand)
P_RIGHT = 3            # phosphorus (right hand)
N_AMIDO = 5            # amido N in PNP backbone
C_CO = 18              # C of carbonyl ligand (trans to N_AMIDO)

# For BuriedVolume z-axis: we look from the amine toward Ru.
# z_axis_atoms defines the "bottom" of the sphere.
# Using the CO carbon (atom 18) on the opposite side of amine gives a
# z-axis that points from CO → Ru → amine, so the steric map shows
# the amine's footprint as seen from Ru.
Z_AXIS_ATOMS = [C_CO]

# xz-plane reference: one of the P atoms for reproducible quadrant orientation
XZ_PLANE_ATOMS = [P_LEFT]

# ============================================================
# Buried volume parameters
# ============================================================
SPHERE_RADIUS = 3.5    # Å, standard SambVca value
INCLUDE_H = True       # include H atoms (important for amines)
RADII_TYPE = "bondi"
RADII_SCALE = 1.17     # SambVca default scaling

# Sensitivity radii
SENSITIVITY_RADII = [3.0, 3.5, 4.0, 4.5, 5.0]

# ============================================================
# Reaction states and fragment handling
# ============================================================
# For most states, atoms 64+ are purely amine.
# Some states have extra small-molecule fragments at the END.

STATES = ["2A", "3A", "4A", "5A", "H1", "H3", "TS3A", "TS5A", "TSH2"]

# Number of trailing atoms that are NOT amine in special states
# (counted from the last atom backward)
TRAILING_FRAGMENT = {
    "2A":   0,
    "3A":   0,
    "TS3A": 0,
    "H1":   0,
    "TSH2": 0,
    "H3":   0,
    "4A":   3,    # last 3 atoms = CO2
    "TS5A": 4,    # last 4 atoms = OCOH
    "5A":   5,    # last 5 atoms = CO2 (3) + H2 (2)
}

# ============================================================
# Known amine abbreviations
# ============================================================
# This list is used to identify the amine name from filenames.
# Order matters: longer names first to avoid partial matches
# (e.g., "DMEDA" before "EDA", "NN-DMDPTA" before "DMDPTA").

KNOWN_AMINES = [
    # Long / composite names first (longer before shorter to avoid partial matches)
    "NN-DMDPTA", "NNDMDPTA", "DMDPTA",
    "33'-DADPA", "33-DADPA", "33DADPA", "DADPA",
    "3-DEAPA", "3DEAPA", "DEAPA",
    "DEGB3APE", "DEGB3", "DGB3APE",
    "1,2-BMAE", "12BMAE", "12-BMAE",
    "3MP15D", "3MP",
    "1-APIP", "1APIP", "APIP",
    "N-DMEDA", "NDMEDA",
    "N-BMA", "NBMA", "BMA",
    "NIPBzA", "NIPBZA", "NIPBz",
    "DMEDA", "N,N-DMEDA",
    "22AEE", "2AEE",
    "B2MEA", "B2ME",
    "13DAP", "13DA",
    "14DAB", "14DA",
    "2EAE", "2EA",
    "mTOL", "MTOL",
    "oTOL", "OTOL",
    # Canonical short names (column B) — used directly in xyz filenames
    "DETRA", "METOX",
    "DMDP", "DEGE", "DEPA", "DBUA", "DADP",
    "BEMT", "BENZ", "ISBZ", "MEPI", "OCTA", "BMAE",
    "AMP", "BOX", "EAE", "MTL", "OTL", "ANI",
    # Single-token names
    "DETA",
    "DIPA",
    "TREA",
    "DBzA", "DBZA",
    "INDOL", "INDO",
    "PIP",
    "NMA",
    "THQ",
    "MOR",
    "1MP",
    "DBA",
    "DEA",
    "EDA",
    "OA",
]

# Canonical name mapping: maps any matched variant to a standard label
# Canonical names (values) match column B of dataset_exported.xlsx
AMINE_CANONICAL = {
    "NN-DMDPTA": "DMDP", "NNDMDPTA": "DMDP", "DMDPTA": "DMDP",
    "33'-DADPA": "DADP", "33-DADPA": "DADP", "33DADPA": "DADP", "DADPA": "DADP",
    "3-DEAPA": "DEPA", "3DEAPA": "DEPA", "DEAPA": "DEPA",
    "DEGB3APE": "DEGE", "DEGB3": "DEGE", "DGB3APE": "DEGE",
    "1,2-BMAE": "BMAE", "12BMAE": "BMAE", "12-BMAE": "BMAE", "BMAE": "BMAE",
    "3MP15D": "3MP15D", "3MP": "3MP15D",
    "1-APIP": "AMP", "1APIP": "AMP", "APIP": "AMP",
    "N-DMEDA": "DMEDA", "NDMEDA": "DMEDA",
    "N-BMA": "BEMT", "NBMA": "BEMT", "BMA": "BEMT",
    "NIPBzA": "ISBZ", "NIPBZA": "ISBZ", "NIPBz": "ISBZ",
    "DMEDA": "DMEDA", "N,N-DMEDA": "DMEDA",
    "22AEE": "METOX", "2AEE": "METOX",
    "B2MEA": "BOX", "B2ME": "BOX",
    "13DAP": "13DAP", "13DA": "13DAP",
    "14DAB": "14DAB", "14DA": "14DAB",
    "2EAE": "EAE", "2EA": "EAE",
    "mTOL": "MTL", "MTOL": "MTL",
    "oTOL": "OTL", "OTOL": "OTL",
    "DETA": "DETA",
    "DIPA": "DIPA",
    "TREA": "TREA",
    # Canonical short names map to themselves (used directly in xyz filenames)
    "DMDP": "DMDP", "DEGE": "DEGE", "DEPA": "DEPA", "DBUA": "DBUA", "DADP": "DADP",
    "BEMT": "BEMT", "BENZ": "BENZ", "ISBZ": "ISBZ", "MEPI": "MEPI", "OCTA": "OCTA",
    "AMP": "AMP", "BOX": "BOX", "EAE": "EAE", "MTL": "MTL", "OTL": "OTL",
    "METOX": "METOX", "BMAE": "BMAE",
    "ANI": "ANI", "DETRA": "DETRA",
    "DBzA": "BENZ", "DBZA": "BENZ",
    "INDOL": "INDO", "INDO": "INDO",
    "PIP": "PIP",
    "NMA": "NMA",
    "THQ": "THQ",
    "MOR": "MOR",
    "1MP": "MEPI",
    "DBA": "DBUA",
    "DEA": "DEA",
    "EDA": "EDA",
    "OA": "OCTA",
}

# ============================================================
# Experimental methanol yield data
# ============================================================
# Fill in your actual experimental yields here (%).
# Used for correlation plots.
MEOH_YIELD = {
    # Keys use canonical names (column B of dataset_exported.xlsx); values in %
    "THQ":    84.65,
    "INDO":   84.63,
    "NMA":    77.23,
    "PIP":    60.25,
    "OTL":    38.84,
    "MOR":    33.35,
    "MEPI":   28.07,
    "MTL":    19.40,
    "AMP":    17.09,
    "BEMT":   14.47,
    "DETA":   11.02,
    "DMDP":    7.45,
    "BOX":     6.97,
    "EAE":     3.39,
    "DMEDA":   3.37,
    "BMAE":    3.36,
    "DEPA":    2.78,
    "DADP":    2.01,
    "OCTA":    1.95,
    "DBUA":    1.29,
    "ISBZ":    1.23,
    "DEGE":    0.65,
    "BENZ":    0.40,
    "EDA":     0.18,
    "METOX":   0.00,
    "DEA":     0.00,
}

# Yield-based classification
YIELD_CATEGORY = {}
for amine, y in MEOH_YIELD.items():
    if y >= 50:
        YIELD_CATEGORY[amine] = "high"
    elif y >= 15:
        YIELD_CATEGORY[amine] = "medium"
    else:
        YIELD_CATEGORY[amine] = "low"

# Colors for yield categories
CATEGORY_COLORS = {
    "high":   "#2ca02c",  # green
    "medium": "#ff7f0e",  # orange
    "low":    "#d62728",  # red
    "unknown": "#7f7f7f", # gray
}
