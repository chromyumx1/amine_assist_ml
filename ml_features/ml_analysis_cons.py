"""
ml_analysis.py — ML-based mechanistic feature analysis for amine selectivity.

Updated workflow:
  Phase 1: Data curation (zero-var removal, steric QC, feature engineering)
  Phase 2: Feature importance (LASSO, RF permutation, SHAP) — regression
  Phase 3: Classification (assistive vs inhibitive) — decision tree + RF
  Phase 4: Constructed features (electronic × steric cross-terms)
  Phase 5: Validation and summary plots

Usage:
  python ml_analysis.py
  python ml_analysis.py --electronic-only
  python ml_analysis.py --top 15
  python ml_analysis.py --yield-threshold 15   # cutoff for assistive/inhibitive
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
OUTPUT_DIR = "output/ml"
RANDOM_STATE = 42
TOP_N = 15
YIELD_THRESHOLD = 15.0  # below = inhibitive, above = assistive

STATE_COLORS = {
    "2A": "#1f77b4",  "3A": "#ff7f0e",  "4A": "#2ca02c",
    "5A": "#d62728",  "H1": "#9467bd",  "H3": "#8c564b",
    "TS3A": "#e377c2", "TS5A": "#7f7f7f", "TSH2": "#bcbd22",
    "global": "#17becf", "constructed": "#e6550d",
}

PATHWAY_ORDER = ["2A", "TS3A", "3A", "4A", "TS5A", "5A", "H1", "TSH2", "H3", "global", "constructed"]


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Phase 1: Data loading, curation, feature engineering
# ============================================================

def load_electronic_data(xlsx_path):
    """Load electronic/energetic descriptors."""
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    df = df.dropna(subset=["Yield"])
    df = df.rename(columns={"ABBREV": "amine"})
    non_feature = ["amine", "Amine", "SMILES", "Yield"]
    feature_cols = [c for c in df.columns if c not in non_feature
                    and pd.api.types.is_numeric_dtype(df[c])]
    return df[["amine", "Yield"] + feature_cols].copy()


def load_steric_data(csv_path, xlsx_df):
    """Load steric descriptors, curate, and pivot to wide format."""
    df = pd.read_csv(csv_path)

    # --- Name mapping ---
    manual_map = {
        "INDO": "INDOL", "MTL": "mTOL", "OTL": "oTOL",
        "OCTA": "OA", "MEPI": "1MP", "METOX": "MOR",
        "DADP": "33'-DADPA", "DEPA": "3-DEAPA", "DMDP": "NN-DMDPTA",
        "ISBZ": "NIPBzA", "DEGE": "DEGB3APE", "DBUA": "DBA",
        "BMAE": "1,2-BMAE", "BEMT": "B2MEA", "EAE": "2EAE",
        "BENZ": "DBzA", "AMP": "1-APIP", "BOX": "22AEE",
    }
    electronic_amines = set(xlsx_df["amine"].unique())
    name_map = {}
    for sa in df["amine"].unique():
        if sa in manual_map:
            name_map[sa] = manual_map[sa]
        elif sa in electronic_amines:
            name_map[sa] = sa
        else:
            for ea in electronic_amines:
                if sa.upper().replace("-", "").replace(",", "").replace("'", "") == \
                   ea.upper().replace("-", "").replace(",", "").replace("'", ""):
                    name_map[sa] = ea
                    break

    df["amine_canonical"] = df["amine"].map(name_map)
    unmatched = df[df["amine_canonical"].isna()]["amine"].unique()
    if len(unmatched) > 0:
        print(f"  [WARN] Unmatched steric amines: {list(unmatched)}")
    df = df.dropna(subset=["amine_canonical"])

    # --- Curate: drop uninformative features per state ---
    curated_cols = []
    for col in ["Vbur_pct", "Q_NE", "Q_NW", "Q_SW", "Q_SE",
                "SASA_amine", "SASA_amine_N", "SASA_N_ratio",
                "Ru_N_distance", "Sterimol_L", "Sterimol_B1", "Sterimol_B5"]:
        if col not in df.columns:
            continue
        # Check variance per state; keep only if std > 1% of mean
        for state in df["state"].unique():
            sub = df[df["state"] == state][col].dropna()
            if len(sub) > 0 and sub.std() > 0.01 * abs(sub.mean() + 1e-10):
                curated_cols.append((col, state))

    curated_cols_unique = list(set(c for c, s in curated_cols))
    print(f"  Steric features retained after QC: {len(curated_cols_unique)} types across states")

    # --- Pivot to wide ---
    pivot_frames = []
    for col in curated_cols_unique:
        piv = df.pivot_table(index="amine_canonical", columns="state", values=col)
        piv.columns = [f"{col}_{state}" for state in piv.columns]
        pivot_frames.append(piv)

    if pivot_frames:
        steric_wide = pd.concat(pivot_frames, axis=1)
        steric_wide.index.name = "amine"
        steric_wide = steric_wide.reset_index()
        # Drop columns that are all-zero or zero-variance after pivoting
        for c in steric_wide.columns:
            if c == "amine":
                continue
            if steric_wide[c].std() < 1e-8:
                steric_wide = steric_wide.drop(columns=[c])
        return steric_wide
    return None


def engineer_features(df, feature_cols):
    """
    Create constructed features capturing electronic × steric interplay.
    These test whether the COMBINATION of descriptors explains outliers
    that neither alone can.
    """
    constructed = {}

    # --- Key cross-terms at the 2A bifurcation point ---
    pairs_2A = [
        ("WBI_2A", "Vbur_pct_2A",       "WBI_x_Vbur_2A"),
        ("WBI_2A", "Ru_N_distance_2A",   "WBI_x_RuN_2A"),
        ("E_int_2A", "Vbur_pct_2A",      "Eint_x_Vbur_2A"),
        ("E_DEF_frag2_2A", "Vbur_pct_2A","EDEF_x_Vbur_2A"),
        ("E2_int2A", "Sterimol_L_2A",    "E2_x_L_2A"),
        ("E_CT_2A", "SASA_amine_2A",     "ECT_x_SASA_2A"),
    ]

    # --- Pathway difference features (assistive - inhibitive) ---
    diff_pairs = [
        ("dG_2A", "dG_TS3A",  "barrier_inhibitive"),  # 2A → TS3A
    ]

    for col_a, col_b, name in pairs_2A:
        if col_a in df.columns and col_b in df.columns:
            constructed[name] = df[col_a].values * df[col_b].values

    for col_a, col_b, name in diff_pairs:
        if col_a in df.columns and col_b in df.columns:
            constructed[name] = df[col_b].values - df[col_a].values

    # --- Quadrant asymmetry (directional steric bias) ---
    if "Q_NE_2A" in df.columns and "Q_NW_2A" in df.columns:
        constructed["Q_asymmetry_2A"] = (
            df["Q_NE_2A"].values - df["Q_NW_2A"].values
        )

    n_new = len(constructed)
    if n_new > 0:
        print(f"  Engineered {n_new} constructed features")
        for name, vals in constructed.items():
            df[name] = vals
            feature_cols.append(name)

    return df, feature_cols


def prepare_dataset(xlsx_path, steric_csv_path=None):
    """Load, merge, curate, and engineer full dataset."""
    print("=" * 50)
    print("PHASE 1: Data Preparation")
    print("=" * 50)

    df_elec = load_electronic_data(xlsx_path)
    print(f"  Electronic: {len(df_elec)} amines × {len(df_elec.columns)-2} features")

    if steric_csv_path and os.path.exists(steric_csv_path):
        df_steric = load_steric_data(steric_csv_path, df_elec)
        if df_steric is not None:
            df = df_elec.merge(df_steric, on="amine", how="left")
            print(f"  After merge: {len(df.columns)-2} total features")
        else:
            df = df_elec
    else:
        df = df_elec

    # Drop NaN columns
    feature_cols = [c for c in df.columns if c not in ["amine", "Yield"]]
    nan_cols = [c for c in feature_cols if df[c].isna().any()]
    if nan_cols:
        print(f"  Dropping {len(nan_cols)} features with NaN")
        df = df.drop(columns=nan_cols)
        feature_cols = [c for c in feature_cols if c not in nan_cols]

    # Feature engineering
    df, feature_cols = engineer_features(df, feature_cols)

    print(f"  Final dataset: {len(df)} amines × {len(feature_cols)} features\n")
    return df, feature_cols


# ============================================================
# Phase 1b: Pre-filtering
# ============================================================

def remove_correlated_features(X, feature_names, threshold=0.95):
    """Cluster correlated features, keep highest-variance representative."""
    var = X.var(axis=0)
    nonzero_mask = var > 1e-10
    X_nz = X[:, nonzero_mask]
    nz_indices = np.where(nonzero_mask)[0]

    if X_nz.shape[1] < 2:
        return list(nz_indices), [feature_names[i] for i in nz_indices]

    corr = np.abs(np.corrcoef(X_nz.T))
    corr = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr, 1)

    dist = 1 - corr
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    Z = linkage(condensed, method="complete")
    clusters = fcluster(Z, t=1 - threshold, criterion="distance")

    kept_indices = []
    for cid in np.unique(clusters):
        local = np.where(clusters == cid)[0]
        variances = X_nz[:, local].var(axis=0)
        best = local[np.argmax(variances)]
        kept_indices.append(int(nz_indices[best]))

    kept_indices = sorted(kept_indices)
    kept_names = [feature_names[i] for i in kept_indices]
    print(f"  Correlation filter (r > {threshold}): {len(feature_names)} → {len(kept_names)}")
    return kept_indices, kept_names


# ============================================================
# Phase 2: Regression feature importance
# ============================================================

def run_lasso(X, y, feature_names):
    print("\n--- LASSO Regression ---")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lasso = LassoCV(cv=min(len(y), 10), random_state=RANDOM_STATE,
                    max_iter=10000, n_alphas=100)
    lasso.fit(X_s, y)

    loo = LeaveOneOut()
    y_pred = cross_val_predict(
        LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
        X_s, y, cv=loo
    )

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  LOO-CV: R² = {r2:.3f}, MAE = {mae:.2f}%")

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lasso.coef_,
        "abs_coef": np.abs(lasso.coef_),
    })
    nonzero = coef_df[coef_df["abs_coef"] > 0].sort_values("abs_coef", ascending=False)
    print(f"  Non-zero features: {len(nonzero)} / {len(feature_names)}")
    return nonzero, y_pred, r2


def run_random_forest(X, y, feature_names):
    print("\n--- Random Forest Regressor ---")
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    loo = LeaveOneOut()
    y_pred = cross_val_predict(rf, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  LOO-CV: R² = {r2:.3f}, MAE = {mae:.2f}%")

    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=50,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
        "mdi_importance": rf.feature_importances_,
    }).sort_values("importance_mean", ascending=False)

    return imp_df, y_pred, r2, rf


def run_shap(X, y, feature_names, amine_names, top_n=TOP_N):
    print("\n--- SHAP Analysis ---")
    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed")
        return None, None, None

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=3,
        learning_rate=0.05, random_state=RANDOM_STATE,
    )
    gb.fit(X, y)
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return shap_values, shap_df, gb


# ============================================================
# Phase 3: Classification — assistive vs inhibitive
# ============================================================

def run_classification(X, y_yield, feature_names, amine_names, threshold=YIELD_THRESHOLD):
    """
    Binary classification: assistive (yield >= threshold) vs inhibitive.
    Uses decision tree for interpretability + RF for robustness.
    """
    print("\n" + "=" * 50)
    print("PHASE 3: Classification (assistive vs inhibitive)")
    print("=" * 50)
    print(f"  Threshold: {threshold}% MeOH yield")

    y_class = (y_yield >= threshold).astype(int)  # 1 = assistive, 0 = inhibitive
    n_assist = y_class.sum()
    n_inhib = len(y_class) - n_assist
    print(f"  Assistive: {n_assist}, Inhibitive: {n_inhib}")

    # --- Decision Tree (interpretable, matches paper narrative) ---
    print("\n  --- Decision Tree ---")

    # Find optimal depth via LOO-CV
    best_depth = 2
    best_acc = 0
    for depth in [2, 3, 4, 5]:
        dt = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        loo = LeaveOneOut()
        scores = cross_val_score(dt, X, y_class, cv=loo, scoring="accuracy")
        acc = scores.mean()
        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    dt = DecisionTreeClassifier(
        max_depth=best_depth, min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    # LOO-CV predictions
    loo = LeaveOneOut()
    y_pred_dt = cross_val_predict(dt, X, y_class, cv=loo)
    acc_dt = accuracy_score(y_class, y_pred_dt)
    print(f"  LOO-CV accuracy: {acc_dt:.3f} (depth={best_depth})")

    # Fit on full data for visualization
    dt.fit(X, y_class)

    # Print tree rules
    tree_text = export_text(dt, feature_names=feature_names, max_depth=5)
    print(f"\n  Decision tree rules:\n{tree_text}")

    # Misclassified amines (outliers of interest)
    misclassified = []
    for i in range(len(amine_names)):
        if y_pred_dt[i] != y_class[i]:
            actual = "assistive" if y_class[i] == 1 else "inhibitive"
            predicted = "assistive" if y_pred_dt[i] == 1 else "inhibitive"
            misclassified.append((amine_names[i], actual, predicted, y_yield[i]))

    if misclassified:
        print(f"\n  Misclassified amines ({len(misclassified)}):")
        for name, actual, pred, yld in misclassified:
            print(f"    {name:>12s}: actual={actual}, predicted={pred}, yield={yld:.1f}%")

    # --- RF Classifier (robustness check) ---
    print("\n  --- Random Forest Classifier ---")
    rfc = RandomForestClassifier(
        n_estimators=500, max_depth=4, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    y_pred_rfc = cross_val_predict(rfc, X, y_class, cv=LeaveOneOut())
    acc_rfc = accuracy_score(y_class, y_pred_rfc)
    print(f"  LOO-CV accuracy: {acc_rfc:.3f}")

    rfc.fit(X, y_class)
    perm_cls = permutation_importance(rfc, X, y_class, n_repeats=50,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    cls_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_cls.importances_mean,
        "importance_std": perm_cls.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return dt, y_class, y_pred_dt, acc_dt, cls_imp, acc_rfc, misclassified


# ============================================================
# Phase 4: Feature category analysis
# ============================================================

def analyze_feature_categories(consensus_df, feature_names):
    """
    Break down importance by feature type (electronic vs steric vs constructed)
    and by mechanistic state.
    """
    print("\n" + "=" * 50)
    print("PHASE 4: Feature Category Analysis")
    print("=" * 50)

    steric_keywords = ["Vbur", "Q_NE", "Q_NW", "Q_SW", "Q_SE",
                       "SASA", "Sterimol", "Ru_N_distance"]

    categories = {}
    for feat in consensus_df.index:
        if any(kw in feat for kw in ["_x_", "asymmetry", "barrier_"]):
            categories[feat] = "constructed"
        elif any(kw in feat for kw in steric_keywords):
            categories[feat] = "steric"
        else:
            categories[feat] = "electronic"

    consensus_df["category"] = consensus_df.index.map(lambda f: categories.get(f, "electronic"))
    consensus_df["state"] = consensus_df.index.map(get_state_from_feature)

    # Category breakdown
    cat_imp = consensus_df.groupby("category")["mean_rank"].agg(["sum", "mean", "count"])
    print("\n  Importance by category:")
    print(cat_imp.to_string())

    # State breakdown
    state_imp = consensus_df.groupby("state")["mean_rank"].agg(["sum", "mean", "count"])
    state_imp = state_imp.sort_values("sum", ascending=False)
    print("\n  Importance by state:")
    print(state_imp.to_string())

    return categories


# ============================================================
# Plotting
# ============================================================

def get_state_from_feature(feat):
    for state in ["TS3A", "TS5A", "TSH2", "2A", "3A", "4A", "5A", "H1", "H3"]:
        if feat.endswith(f"_{state}") or f"_{state}_" in feat:
            return state
    if any(kw in feat for kw in ["_x_", "asymmetry", "barrier_"]):
        return "constructed"
    return "global"


def plot_lasso_coefficients(df, top_n=TOP_N):
    df = df.head(top_n).copy()
    if len(df) == 0:
        return
    colors = [STATE_COLORS.get(get_state_from_feature(f), "#333") for f in df["feature"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(range(len(df)), df["coefficient"].values, color=colors, edgecolor="k", lw=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("LASSO coefficient (standardized)")
    ax.set_title("LASSO: Features surviving regularization")
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    _add_state_legend(ax, df["feature"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "lasso_coefficients.png"), dpi=300)
    plt.close(fig)
    print("  Saved: lasso_coefficients.png")


def plot_rf_importance(df, top_n=TOP_N):
    df = df.head(top_n).copy()
    colors = [STATE_COLORS.get(get_state_from_feature(f), "#333") for f in df["feature"]]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(df)), df["importance_mean"].values, xerr=df["importance_std"].values,
            color=colors, edgecolor="k", lw=0.3, capsize=2)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("Permutation importance")
    ax.set_title("Random Forest: Permutation importance")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    _add_state_legend(ax, df["feature"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rf_permutation_importance.png"), dpi=300)
    plt.close(fig)
    print("  Saved: rf_permutation_importance.png")


def plot_state_aggregated_importance(imp_df, suffix=""):
    df = imp_df.copy()
    df["state"] = df["feature"].apply(get_state_from_feature)
    state_imp = df.groupby("state")["importance_mean"].sum()
    ordered = [s for s in PATHWAY_ORDER if s in state_imp.index]
    state_imp = state_imp[ordered]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [STATE_COLORS.get(s, "#333") for s in state_imp.index]
    bars = ax.bar(range(len(state_imp)), state_imp.values, color=colors, edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(state_imp)))
    ax.set_xticklabels(state_imp.index, fontsize=10)
    ax.set_ylabel("Cumulative importance")
    ax.set_title(f"Mechanistic state importance{suffix}")
    ax.grid(axis="y", alpha=0.3)

    # Add inhibitive / assistive labels
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"state_importance{suffix.replace(' ', '_')}.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: state_importance{suffix.replace(' ', '_')}.png")


def plot_prediction_vs_actual(y, y_pred_lasso, y_pred_rf, amine_names, r2_lasso, r2_rf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, r2, title in [
        (axes[0], y_pred_lasso, r2_lasso, "LASSO"),
        (axes[1], y_pred_rf, r2_rf, "Random Forest"),
    ]:
        ax.scatter(y, y_pred, c="#1f77b4", s=50, edgecolors="k", lw=0.5, zorder=3)
        for i, name in enumerate(amine_names):
            ax.annotate(name, (y[i], y_pred[i]), fontsize=6, ha="left",
                        xytext=(3, 3), textcoords="offset points", alpha=0.7)
        lims = [min(y.min(), y_pred.min()) - 5, max(y.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual MeOH yield (%)")
        ax.set_ylabel("Predicted MeOH yield (%)")
        ax.set_title(f"{title} (LOO-CV, R² = {r2:.3f})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"), dpi=300)
    plt.close(fig)
    print("  Saved: prediction_vs_actual.png")


def plot_shap_summary(shap_values, X, feature_names, amine_names, top_n=TOP_N):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    for rank, fi in enumerate(top_idx):
        vals = shap_values[:, fi]
        fv = X[:, fi]
        fmin, fmax = fv.min(), fv.max()
        normed = (fv - fmin) / (fmax - fmin) if fmax > fmin else np.full_like(fv, 0.5)
        jitter = np.random.normal(0, 0.1, size=len(vals))
        ax.scatter(vals, rank + jitter, c=normed, cmap="RdBu_r", s=15, alpha=0.7, edgecolors="none")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("SHAP value (impact on yield)")
    ax.set_title("SHAP: Per-amine feature impact")
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Feature value (low → high)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300)
    plt.close(fig)
    print("  Saved: shap_summary.png")


def plot_decision_tree(dt, feature_names, amine_names, y_class):
    """Visualize the decision tree — the key interpretable figure."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        dt, feature_names=feature_names,
        class_names=["Inhibitive", "Assistive"],
        filled=True, rounded=True, fontsize=8,
        impurity=False, proportion=True, ax=ax,
    )
    ax.set_title("Decision Tree: Assistive vs. Inhibitive Classification", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "decision_tree.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: decision_tree.png")

    # Also save text version
    tree_text = export_text(dt, feature_names=feature_names, max_depth=10)
    with open(os.path.join(OUTPUT_DIR, "decision_tree_rules.txt"), "w") as f:
        f.write(tree_text)


def plot_classification_importance(cls_imp, top_n=TOP_N):
    """Feature importance for the classification task specifically."""
    df = cls_imp.head(top_n).copy()
    colors = [STATE_COLORS.get(get_state_from_feature(f), "#333") for f in df["feature"]]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(df)), df["importance_mean"].values, xerr=df["importance_std"].values,
            color=colors, edgecolor="k", lw=0.3, capsize=2)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("Permutation importance (classification)")
    ax.set_title("What separates assistive from inhibitive amines?")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    _add_state_legend(ax, df["feature"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "classification_importance.png"), dpi=300)
    plt.close(fig)
    print("  Saved: classification_importance.png")


def plot_consensus_ranking(lasso_df, rf_df, shap_df, top_n=TOP_N):
    methods = {}
    if lasso_df is not None and len(lasso_df) > 0:
        r = lasso_df.set_index("feature")["abs_coef"]
        methods["LASSO"] = r / r.max() if r.max() > 0 else r
    if rf_df is not None:
        r = rf_df.set_index("feature")["importance_mean"]
        methods["RF"] = r / r.max() if r.max() > 0 else r
    if shap_df is not None:
        r = shap_df.set_index("feature")["mean_abs_shap"]
        methods["SHAP"] = r / r.max() if r.max() > 0 else r

    all_feats = set()
    for v in methods.values():
        all_feats.update(v.index)

    consensus = pd.DataFrame(index=list(all_feats))
    for name, ranking in methods.items():
        consensus[name] = ranking.reindex(consensus.index).fillna(0)
    consensus["mean_rank"] = consensus.mean(axis=1)
    consensus = consensus.sort_values("mean_rank", ascending=False)

    top = consensus.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    x = np.arange(len(top))
    width = 0.25
    for i, (mn, col) in enumerate(methods.items()):
        ax.barh(x + i * width, top[mn].values, width, label=mn, edgecolor="k", lw=0.3, alpha=0.8)
    ax.set_yticks(x + width)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Normalized importance")
    ax.set_title("Consensus feature ranking (all methods)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "consensus_ranking.png"), dpi=300)
    plt.close(fig)
    print("  Saved: consensus_ranking.png")

    consensus.to_csv(os.path.join(OUTPUT_DIR, "consensus_feature_ranking.csv"))
    return consensus


def _add_state_legend(ax, features):
    states = list(set(get_state_from_feature(f) for f in features))
    handles = [Patch(facecolor=STATE_COLORS.get(s, "#333"), label=s) for s in sorted(states)]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="State")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="dataset_exported.xlsx")
    parser.add_argument("--steric-csv", default="all_descriptors_merged.csv")
    parser.add_argument("--electronic-only", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--yield-threshold", type=float, default=YIELD_THRESHOLD,
                        help="MeOH yield cutoff for assistive/inhibitive (default 15%%)")
    args = parser.parse_args()

    ensure_output()
    top_n = args.top

    # ---- Phase 1: Data ----
    steric_path = None if args.electronic_only else args.steric_csv
    df, feature_cols = prepare_dataset(args.xlsx, steric_path)

    X = df[feature_cols].values.astype(float)
    y = df["Yield"].values.astype(float)
    amine_names = df["amine"].values

    kept_idx, kept_names = remove_correlated_features(X, feature_cols, args.corr_threshold)
    X_f = X[:, kept_idx]

    # Save curated feature list
    pd.DataFrame({"feature": kept_names}).to_csv(
        os.path.join(OUTPUT_DIR, "curated_features.csv"), index=False)

    # ---- Phase 2: Regression importance ----
    print("\n" + "=" * 50)
    print("PHASE 2: Regression Feature Importance")
    print("=" * 50)

    lasso_df, y_pred_lasso, r2_lasso = run_lasso(X_f, y, kept_names)
    if len(lasso_df) > 0:
        print(f"\n  Top LASSO features:")
        for _, row in lasso_df.head(top_n).iterrows():
            print(f"    {row['feature']:>35s}  coef = {row['coefficient']:+.3f}")
        lasso_df.to_csv(os.path.join(OUTPUT_DIR, "lasso_features.csv"), index=False)
        plot_lasso_coefficients(lasso_df, top_n)

    rf_df, y_pred_rf, r2_rf, rf_model = run_random_forest(X_f, y, kept_names)
    print(f"\n  Top RF features:")
    for _, row in rf_df.head(top_n).iterrows():
        print(f"    {row['feature']:>35s}  imp = {row['importance_mean']:.4f}")
    rf_df.to_csv(os.path.join(OUTPUT_DIR, "rf_importance.csv"), index=False)
    plot_rf_importance(rf_df, top_n)
    plot_state_aggregated_importance(rf_df, suffix=" (regression)")

    shap_values, shap_df, gb_model = run_shap(X_f, y, kept_names, amine_names, top_n)
    if shap_df is not None:
        print(f"\n  Top SHAP features:")
        for _, row in shap_df.head(top_n).iterrows():
            print(f"    {row['feature']:>35s}  |SHAP| = {row['mean_abs_shap']:.3f}")
        shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
        plot_shap_summary(shap_values, X_f, kept_names, amine_names, top_n)

    plot_prediction_vs_actual(y, y_pred_lasso, y_pred_rf, amine_names, r2_lasso, r2_rf)

    print("\n--- Consensus (Regression) ---")
    consensus = plot_consensus_ranking(lasso_df, rf_df, shap_df, top_n)

    # ---- Phase 3: Classification ----
    dt, y_class, y_pred_dt, acc_dt, cls_imp, acc_rfc, misclassified = \
        run_classification(X_f, y, kept_names, amine_names, args.yield_threshold)

    plot_decision_tree(dt, kept_names, amine_names, y_class)
    plot_classification_importance(cls_imp, top_n)
    plot_state_aggregated_importance(cls_imp, suffix=" (classification)")
    cls_imp.to_csv(os.path.join(OUTPUT_DIR, "classification_importance.csv"), index=False)

    # ---- Phase 4: Category analysis ----
    analyze_feature_categories(consensus, kept_names)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {X_f.shape[0]} amines × {X_f.shape[1]} features (after filtering)")
    print(f"\nRegression:")
    print(f"  LASSO LOO-CV R² = {r2_lasso:.3f}  ({len(lasso_df)} non-zero features)")
    print(f"  RF    LOO-CV R² = {r2_rf:.3f}")
    print(f"\nClassification (threshold = {args.yield_threshold}% yield):")
    print(f"  Decision Tree LOO-CV acc = {acc_dt:.3f}")
    print(f"  RF Classifier  LOO-CV acc = {acc_rfc:.3f}")
    if misclassified:
        print(f"  Misclassified: {[m[0] for m in misclassified]}")
    print(f"\nTop consensus features:")
    for feat in consensus.head(5).index:
        state = get_state_from_feature(feat)
        print(f"  • {feat} (state: {state})")
    print(f"\nOutputs: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
