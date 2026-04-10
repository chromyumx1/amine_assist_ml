"""
ml_feature_analysis.py — ML-based feature importance for amine selectivity.

Purpose: Identify which descriptors and mechanistic states control methanol yield.
NOT for prediction accuracy — for mechanistic insight.

Methods:
  1. Correlation pre-filtering (remove redundant features)
  2. LASSO regression (sparse feature selection)
  3. Random Forest + permutation importance (LOO-CV)
  4. SHAP analysis (per-amine explanations)

Requirements:
  pip install scikit-learn shap matplotlib pandas numpy openpyxl

Usage:
  python ml_feature_analysis.py
  python ml_feature_analysis.py --electronic-only    # skip steric merge
  python ml_feature_analysis.py --top 20             # show top 20 features
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
OUTPUT_DIR = "output/ml"
RANDOM_STATE = 42
TOP_N = 15  # default number of top features to display

# State colors for grouping features by mechanistic state
STATE_COLORS = {
    "2A": "#1f77b4",
    "3A": "#ff7f0e",
    "4A": "#2ca02c",
    "5A": "#d62728",
    "H1": "#9467bd",
    "H3": "#8c564b",
    "TS3A": "#e377c2",
    "TS5A": "#7f7f7f",
    "TSH2": "#bcbd22",
    "global": "#17becf",
    "steric": "#aec7e8",
}


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Data loading and merging
# ============================================================

def load_electronic_data(xlsx_path):
    """Load electronic/energetic descriptors from the master xlsx."""
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
    df = df.dropna(subset=["Yield"])
    df = df.rename(columns={"ABBREV": "amine"})

    # Separate target and features
    non_feature_cols = ["amine", "Amine", "SMILES", "Yield"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Drop any non-numeric
    numeric_features = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)

    return df[["amine", "Yield"] + numeric_features].copy()


def load_steric_data(csv_path, xlsx_df):
    """
    Load steric descriptors and pivot to wide format (one row per amine).
    Merge with electronic data by matching amine names.
    """
    df = pd.read_csv(csv_path)

    # Build name mapping between steric and electronic datasets
    steric_amines = sorted(df["amine"].unique())
    electronic_amines = sorted(xlsx_df["amine"].unique())

    # Auto-match by uppercase similarity
    name_map = {}
    for sa in steric_amines:
        for ea in electronic_amines:
            if sa.upper().replace("-", "").replace(",", "").replace("'", "") == \
               ea.upper().replace("-", "").replace(",", "").replace("'", ""):
                name_map[sa] = ea
                break

    # Manual fallback matches (edit if needed)
    manual_map = {
        "INDO": "INDOL", "MTL": "mTOL", "OTL": "oTOL",
        "OCTA": "OA", "MEPI": "1MP", "METOX": "MOR",
        "DADP": "33'-DADPA", "DEPA": "3-DEAPA", "DMDP": "NN-DMDPTA",
        "ISBZ": "NIPBzA", "DEGE": "DEGB3APE", "DBUA": "DBA",
        "BMAE": "1,2-BMAE", "BEMT": "B2MEA", "EAE": "2EAE",
        "BENZ": "DBzA", "AMP": "1-APIP", "BOX": "22AEE",
    }
    for k, v in manual_map.items():
        if k not in name_map:
            name_map[k] = v

    df["amine_canonical"] = df["amine"].map(name_map)
    unmatched = df[df["amine_canonical"].isna()]["amine"].unique()
    if len(unmatched) > 0:
        print(f"  [WARN] Unmatched steric amines: {unmatched}")
        print("  Edit manual_map in ml_feature_analysis.py to fix.")

    df = df.dropna(subset=["amine_canonical"])

    # Pivot: state-specific columns
    steric_cols = ["Vbur_pct", "Q_NE", "Q_NW", "Q_SW", "Q_SE",
                   "SASA_amine", "SASA_amine_N", "Ru_N_distance",
                   "Sterimol_L", "Sterimol_B1", "Sterimol_B5"]
    avail_cols = [c for c in steric_cols if c in df.columns]

    pivot_frames = []
    for col in avail_cols:
        piv = df.pivot_table(index="amine_canonical", columns="state", values=col)
        piv.columns = [f"{col}_{state}" for state in piv.columns]
        pivot_frames.append(piv)

    if pivot_frames:
        steric_wide = pd.concat(pivot_frames, axis=1)
        steric_wide.index.name = "amine"
        steric_wide = steric_wide.reset_index()
        return steric_wide
    return None


def prepare_dataset(xlsx_path, steric_csv_path=None):
    """Load, merge, and clean the full dataset."""
    print("Loading electronic data...")
    df_elec = load_electronic_data(xlsx_path)
    print(f"  {len(df_elec)} amines, {len(df_elec.columns)-2} electronic features")

    if steric_csv_path and os.path.exists(steric_csv_path):
        print("Loading steric data...")
        df_steric = load_steric_data(steric_csv_path, df_elec)
        if df_steric is not None:
            df = df_elec.merge(df_steric, on="amine", how="left")
            n_steric = len(df_steric.columns) - 1
            print(f"  Merged {n_steric} steric features")
        else:
            df = df_elec
    else:
        df = df_elec
        print("  No steric data provided; using electronic only.")

    # Drop columns with any NaN (from partial steric coverage)
    feature_cols = [c for c in df.columns if c not in ["amine", "Yield"]]
    nan_cols = [c for c in feature_cols if df[c].isna().any()]
    if nan_cols:
        print(f"  Dropping {len(nan_cols)} features with NaN values")
        df = df.drop(columns=nan_cols)

    feature_cols = [c for c in df.columns if c not in ["amine", "Yield"]]
    print(f"  Final: {len(df)} amines × {len(feature_cols)} features\n")

    return df, feature_cols


# ============================================================
# 2. Feature pre-filtering
# ============================================================

def remove_correlated_features(X, feature_names, threshold=0.95):
    """
    Cluster highly correlated features and keep one per cluster.
    Reduces multicollinearity without arbitrary manual selection.
    """
    # Drop zero-variance columns before correlation (np.corrcoef gives NaN for them)
    var = X.var(axis=0)
    nonzero_mask = var > 0
    X_nz = X[:, nonzero_mask]
    nz_indices = np.where(nonzero_mask)[0]

    corr = np.abs(np.corrcoef(X_nz.T))
    # Replace any residual NaN (e.g. from near-zero variance) with 0 (uncorrelated)
    corr = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr, 1)             # diagonal = perfect self-correlation

    # Hierarchical clustering on correlation distance
    dist = 1 - corr
    dist = np.clip(dist, 0, None)          # ensure non-negative
    dist = (dist + dist.T) / 2            # enforce exact symmetry (float rounding)
    np.fill_diagonal(dist, 0)             # diagonal must be exactly 0
    condensed = squareform(dist)
    Z = linkage(condensed, method="complete")
    clusters = fcluster(Z, t=1 - threshold, criterion="distance")

    # Keep the feature with highest variance in each cluster
    # (indices are into X_nz; map back to original X via nz_indices)
    kept_indices = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        local_indices = np.where(mask)[0]          # indices into X_nz
        variances = X_nz[:, local_indices].var(axis=0)
        best_local = local_indices[np.argmax(variances)]
        kept_indices.append(int(nz_indices[best_local]))  # map to original X

    kept_indices = sorted(kept_indices)
    kept_names = [feature_names[i] for i in kept_indices]

    print(f"  Correlation filter (r > {threshold}): {len(feature_names)} → {len(kept_names)} features")
    return kept_indices, kept_names


# ============================================================
# 3. LASSO — sparse feature selection
# ============================================================

def run_lasso(X, y, feature_names):
    """LASSO with cross-validated alpha selection."""
    print("--- LASSO Regression ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=min(len(y), 10), random_state=RANDOM_STATE,
                    max_iter=10000, n_alphas=100)
    lasso.fit(X_scaled, y)

    # LOO-CV prediction
    loo = LeaveOneOut()
    y_pred = cross_val_predict(
        LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000),
        X_scaled, y, cv=loo
    )

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  LOO-CV: R² = {r2:.3f}, MAE = {mae:.2f}%")
    print(f"  Optimal alpha = {lasso.alpha_:.4f}")

    # Non-zero coefficients
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lasso.coef_,
        "abs_coef": np.abs(lasso.coef_),
    })
    nonzero = coef_df[coef_df["abs_coef"] > 0].sort_values("abs_coef", ascending=False)
    print(f"  Non-zero features: {len(nonzero)} / {len(feature_names)}")

    return nonzero, y_pred, r2


# ============================================================
# 4. Random Forest + Permutation Importance
# ============================================================

def run_random_forest(X, y, feature_names):
    """RF with LOO-CV and permutation importance."""
    print("\n--- Random Forest ---")

    rf = RandomForestRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    # LOO-CV prediction
    loo = LeaveOneOut()
    y_pred = cross_val_predict(rf, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  LOO-CV: R² = {r2:.3f}, MAE = {mae:.2f}%")

    # Fit on full data for feature importance
    rf.fit(X, y)

    # Permutation importance (more reliable than MDI for correlated features)
    perm_imp = permutation_importance(
        rf, X, y, n_repeats=50, random_state=RANDOM_STATE, n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
        "mdi_importance": rf.feature_importances_,
    }).sort_values("importance_mean", ascending=False)

    return imp_df, y_pred, r2, rf


# ============================================================
# 5. SHAP analysis
# ============================================================

def run_shap(X, y, feature_names, amine_names, top_n=TOP_N):
    """SHAP on gradient boosting for per-sample feature explanations."""
    print("\n--- SHAP Analysis ---")

    try:
        import shap
    except ImportError:
        print("  [SKIP] shap not installed. pip install shap")
        return None, None, None

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=3,
        learning_rate=0.05, random_state=RANDOM_STATE,
    )
    gb.fit(X, y)

    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    return shap_values, shap_df, gb


# ============================================================
# 6. Plotting
# ============================================================

def get_state_from_feature(feature_name):
    """Extract mechanistic state from feature name for coloring."""
    for state in ["TS3A", "TS5A", "TSH2", "2A", "3A", "4A", "5A", "H1", "H3"]:
        if feature_name.endswith(f"_{state}") or f"_{state}_" in feature_name:
            return state
    # Check steric features
    for state in ["2A", "3A", "4A", "5A", "H1", "H3", "TS3A", "TS5A", "TSH2"]:
        if feature_name.endswith(f"_{state}"):
            return state
    return "global"


def plot_lasso_coefficients(nonzero_df, top_n=TOP_N):
    """Bar chart of LASSO coefficients."""
    df = nonzero_df.head(top_n).copy()
    if len(df) == 0:
        return

    colors = [STATE_COLORS.get(get_state_from_feature(f), "#333333")
              for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df["coefficient"].values, color=colors, edgecolor="k", lw=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("LASSO coefficient (standardized)", fontsize=10)
    ax.set_title("LASSO: Features surviving regularization", fontsize=12)
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Legend for states
    unique_states = list(set(get_state_from_feature(f) for f in df["feature"]))
    handles = [Patch(facecolor=STATE_COLORS.get(s, "#333"), label=s)
               for s in sorted(unique_states)]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="State")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "lasso_coefficients.png"), dpi=300)
    plt.close(fig)
    print("  Saved: lasso_coefficients.png")


def plot_rf_importance(imp_df, top_n=TOP_N):
    """Permutation importance from Random Forest."""
    df = imp_df.head(top_n).copy()
    colors = [STATE_COLORS.get(get_state_from_feature(f), "#333333")
              for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df["importance_mean"].values, xerr=df["importance_std"].values,
            color=colors, edgecolor="k", lw=0.3, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlabel("Permutation importance (decrease in R²)", fontsize=10)
    ax.set_title("Random Forest: Feature importance (LOO-CV)", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    unique_states = list(set(get_state_from_feature(f) for f in df["feature"]))
    handles = [Patch(facecolor=STATE_COLORS.get(s, "#333"), label=s)
               for s in sorted(unique_states)]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="State")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rf_permutation_importance.png"), dpi=300)
    plt.close(fig)
    print("  Saved: rf_permutation_importance.png")


def plot_state_aggregated_importance(imp_df):
    """
    Aggregate feature importance by mechanistic state.
    KEY PLOT: shows which states along the pathway matter most.
    """
    df = imp_df.copy()
    df["state"] = df["feature"].apply(get_state_from_feature)

    state_imp = df.groupby("state")["importance_mean"].sum().sort_values(ascending=False)

    # Order by reaction pathway
    pathway_order = ["2A", "TS3A", "3A", "4A", "TS5A", "5A", "H1", "TSH2", "H3", "global"]
    ordered = [s for s in pathway_order if s in state_imp.index]
    state_imp = state_imp[ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [STATE_COLORS.get(s, "#333") for s in state_imp.index]
    ax.bar(range(len(state_imp)), state_imp.values, color=colors, edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(state_imp)))
    ax.set_xticklabels(state_imp.index, fontsize=10)
    ax.set_ylabel("Cumulative permutation importance", fontsize=10)
    ax.set_title("Which mechanistic states control methanol yield?", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "state_importance.png"), dpi=300)
    plt.close(fig)
    print("  Saved: state_importance.png")


def plot_prediction_vs_actual(y, y_pred_lasso, y_pred_rf, amine_names, r2_lasso, r2_rf):
    """Predicted vs actual yield for both models."""
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
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual MeOH yield (%)", fontsize=10)
        ax.set_ylabel("Predicted MeOH yield (%)", fontsize=10)
        ax.set_title(f"{title} (LOO-CV, R² = {r2:.3f})", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"), dpi=300)
    plt.close(fig)
    print("  Saved: prediction_vs_actual.png")


def plot_shap_summary(shap_values, X, feature_names, amine_names, top_n=TOP_N):
    """SHAP beeswarm-style plot (manual implementation to avoid shap plotting issues)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

    for rank, feat_idx in enumerate(top_idx):
        vals = shap_values[:, feat_idx]
        feat_vals = X[:, feat_idx]

        # Normalize feature values for color mapping
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            normed = (feat_vals - fmin) / (fmax - fmin)
        else:
            normed = np.full_like(feat_vals, 0.5)

        # Jitter y
        jitter = np.random.normal(0, 0.1, size=len(vals))
        ax.scatter(vals, rank + jitter, c=normed, cmap="RdBu_r",
                   s=15, alpha=0.7, edgecolors="none")

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("SHAP value (impact on yield prediction)", fontsize=10)
    ax.set_title("SHAP: Feature impact per amine", fontsize=12)
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Feature value (low → high)", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300)
    plt.close(fig)
    print("  Saved: shap_summary.png")


def plot_shap_per_amine(shap_values, feature_names, amine_names, top_n=8):
    """
    Waterfall-style: show top SHAP contributors for selected amines.
    Pick 2 high-yield, 2 medium, 2 low-yield examples.
    """
    from config import YIELD_CATEGORY

    # Select representative amines
    representatives = []
    for cat in ["high", "low", "medium"]:
        candidates = [a for a in amine_names if YIELD_CATEGORY.get(a) == cat]
        representatives.extend(candidates[:2])

    if not representatives:
        # fallback: pick highest and lowest yield
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, amine in enumerate(representatives[:6]):
        if idx >= len(axes):
            break
        ax = axes[idx]
        amine_idx = list(amine_names).index(amine)
        sv = shap_values[amine_idx]

        # Top features by absolute SHAP for this amine
        top_feat_idx = np.argsort(np.abs(sv))[-top_n:]
        names = [feature_names[i] for i in top_feat_idx]
        vals = sv[top_feat_idx]

        colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
        ax.barh(range(len(names)), vals, color=colors, edgecolor="k", lw=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        cat = YIELD_CATEGORY.get(amine, "?")
        ax.set_title(f"{amine} ({cat} yield)", fontsize=10)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("SHAP value", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    for idx in range(len(representatives), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("SHAP: What drives each amine's predicted yield?", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "shap_per_amine.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: shap_per_amine.png")


def plot_consensus_ranking(lasso_df, rf_df, shap_df, top_n=TOP_N):
    """
    Consensus: overlay rankings from all 3 methods.
    Features that rank high in ALL methods are the most trustworthy.
    """
    # Normalize each ranking to [0, 1]
    methods = {}
    if lasso_df is not None and len(lasso_df) > 0:
        lasso_rank = lasso_df.set_index("feature")["abs_coef"]
        lasso_rank = lasso_rank / lasso_rank.max()
        methods["LASSO"] = lasso_rank
    if rf_df is not None:
        rf_rank = rf_df.set_index("feature")["importance_mean"]
        rf_rank = rf_rank / rf_rank.max() if rf_rank.max() > 0 else rf_rank
        methods["RF"] = rf_rank
    if shap_df is not None:
        shap_rank = shap_df.set_index("feature")["mean_abs_shap"]
        shap_rank = shap_rank / shap_rank.max() if shap_rank.max() > 0 else shap_rank
        methods["SHAP"] = shap_rank

    # Merge
    all_features = set()
    for v in methods.values():
        all_features.update(v.index)

    consensus = pd.DataFrame(index=list(all_features))
    for name, ranking in methods.items():
        consensus[name] = ranking.reindex(consensus.index).fillna(0)

    consensus["mean_rank"] = consensus.mean(axis=1)
    consensus = consensus.sort_values("mean_rank", ascending=False)
    top = consensus.head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    x = np.arange(len(top))
    width = 0.25

    for i, (method_name, col) in enumerate(methods.items()):
        vals = top[method_name].values
        ax.barh(x + i * width, vals, width, label=method_name,
                edgecolor="k", lw=0.3, alpha=0.8)

    ax.set_yticks(x + width)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Normalized importance", fontsize=10)
    ax.set_title("Consensus: Features ranked high by multiple methods", fontsize=12)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "consensus_ranking.png"), dpi=300)
    plt.close(fig)
    print("  Saved: consensus_ranking.png")

    # Save to CSV
    consensus.to_csv(os.path.join(OUTPUT_DIR, "consensus_feature_ranking.csv"))
    print("  Saved: consensus_feature_ranking.csv")

    return consensus


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="dataset_exported.xlsx")
    parser.add_argument("--steric-csv", default="all_descriptors_merged.csv")
    parser.add_argument("--electronic-only", action="store_true")
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                        help="Correlation threshold for pre-filtering")
    args = parser.parse_args()

    ensure_output()
    top_n = args.top

    # Load data
    steric_path = None if args.electronic_only else args.steric_csv
    df, feature_cols = prepare_dataset(args.xlsx, steric_path)

    X = df[feature_cols].values.astype(float)
    y = df["Yield"].values.astype(float)
    amine_names = df["amine"].values

    print(f"Dataset: {X.shape[0]} amines × {X.shape[1]} features")
    print(f"Yield range: {y.min():.1f} – {y.max():.1f}%\n")

    # Pre-filter correlated features
    kept_idx, kept_names = remove_correlated_features(
        X, feature_cols, threshold=args.corr_threshold
    )
    X_filtered = X[:, kept_idx]
    print()

    # --- LASSO ---
    lasso_df, y_pred_lasso, r2_lasso = run_lasso(X_filtered, y, kept_names)
    if len(lasso_df) > 0:
        print(f"\n  Top LASSO features:")
        for _, row in lasso_df.head(top_n).iterrows():
            print(f"    {row['feature']:>30s}  coef = {row['coefficient']:+.3f}")
        lasso_df.to_csv(os.path.join(OUTPUT_DIR, "lasso_features.csv"), index=False)
        plot_lasso_coefficients(lasso_df, top_n)

    # --- Random Forest ---
    rf_df, y_pred_rf, r2_rf, rf_model = run_random_forest(X_filtered, y, kept_names)
    print(f"\n  Top RF features:")
    for _, row in rf_df.head(top_n).iterrows():
        print(f"    {row['feature']:>30s}  imp = {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    rf_df.to_csv(os.path.join(OUTPUT_DIR, "rf_importance.csv"), index=False)
    plot_rf_importance(rf_df, top_n)
    plot_state_aggregated_importance(rf_df)

    # --- SHAP ---
    shap_result = run_shap(X_filtered, y, kept_names, amine_names, top_n)
    shap_df = None
    shap_values, shap_df, gb_model = shap_result
    if shap_df is not None:
        print(f"\n  Top SHAP features:")
        for _, row in shap_df.head(top_n).iterrows():
            print(f"    {row['feature']:>30s}  |SHAP| = {row['mean_abs_shap']:.3f}")
        shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
        plot_shap_summary(shap_values, X_filtered, kept_names, amine_names, top_n)
        try:
            plot_shap_per_amine(shap_values, kept_names, amine_names)
        except Exception as e:
            print(f"  [WARN] Per-amine SHAP plot failed: {e}")

    # --- Prediction comparison ---
    plot_prediction_vs_actual(y, y_pred_lasso, y_pred_rf, amine_names, r2_lasso, r2_rf)

    # --- Consensus ---
    print("\n--- Consensus Ranking ---")
    consensus = plot_consensus_ranking(lasso_df, rf_df, shap_df, top_n)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"LASSO LOO-CV R² = {r2_lasso:.3f}  ({len(lasso_df)} non-zero features)")
    print(f"RF    LOO-CV R² = {r2_rf:.3f}")
    print(f"\nTop consensus features:")
    for feat in consensus.head(5).index:
        state = get_state_from_feature(feat)
        print(f"  • {feat} (state: {state})")
    print(f"\nAll outputs in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
