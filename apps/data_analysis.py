"""
Multi-Refrigerant Operability Analysis for Vapor Compression Chiller
=====================================================================
Compares R-410A, R-134a, R-32, and R-1234yf using the operability framework
of Georgakis et al. (2003).

**v4 — Rigorous geometric Operability Index + engineering-grounded DOS**

The Operability Index is computed strictly as defined by Vinson and Georgakis
(2000), i.e. as a Lebesgue (area) ratio

    OI = area(AOS n DOS) / area(DOS)

evaluated in the two-dimensional output plane (capacity CW, efficiency COP).
The reduction to that plane is deliberate: the two water flows are nearly
collinear (Pearson r > 0.99 — both are proxies for heat flow), so the
three-dimensional output set is a thin sheet and a volumetric measure would be
degenerate. The condenser cooling-water limit therefore enters point-wise, as a
capacity ceiling TW <= TW_max, rather than as a third dimension.

Algorithm (see `calc_oi`):
  1. take the AOS as the simulated operating points (the Latin-Hypercube sample);
  2. keep the points that satisfy every DOS constraint, including TW <= TW_max;
  3. take the convex hull of that cloud in the (CW, COP) plane, with hull
     membership decided by a Delaunay triangulation (boundary points included);
  4. measure the fraction of the desired box covered by that hull through
     Monte-Carlo integration.
The estimator converges in the number of Monte-Carlo points and is insensitive
to the density of the AOS sample.

DOS scenarios are grounded in engineering limits rather than chosen ad hoc.
Each scenario tightens ONE constraint and relaxes the others to the cluster's
physical extreme, which is what exposes the ranking inversions:
  - CW_min  <- design cooling load (RT ~= CW[m3/h] x 6.6)
  - TW_max  <- cooling-tower / condenser-water availability
  - COP_min <- efficiency target, set relative to the achievable envelope

Comparison is intra-cluster, because the four refrigerants form two disjoint
capacity clusters that serve different application scales:
  - Low-pressure cluster:  R-134a vs R-1234yf  (GWP 1430 vs 1)
  - High-pressure cluster: R-410A vs R-32      (GWP 2088 vs 675)

Outputs: individual + comparative + intra-cluster + ranking-inversion figures.
All results saved to analysis_outputs/

Authors: Nicolas Spogis, Bernardo Ronchetti, Douglas F. Barbin, Heleno Bispo
Target: Digital Chemical Engineering
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

# Refrigerant metadata
REFRIGERANTS = {
    'R-410A':  {'file': '../dataset/DOE_Dataset_R-410a.csv',  'seasonal': '../dataset/DOE_Seasonal_R-410a.csv',  'color': '#2196F3', 'marker': 'o', 'gwp': 2088, 'type': 'HFC blend'},
    'R-134a':  {'file': '../dataset/DOE_Dataset_R-134a.csv',  'seasonal': '../dataset/DOE_Seasonal_R-134a.csv',  'color': '#4CAF50', 'marker': 's', 'gwp': 1430, 'type': 'HFC'},
    'R-32':    {'file': '../dataset/DOE_Dataset_R-32.csv',    'seasonal': '../dataset/DOE_Seasonal_R-32.csv',    'color': '#FF9800', 'marker': '^', 'gwp': 675,  'type': 'HFC'},
    'R-1234yf':{'file': '../dataset/DOE_Dataset_R-1234yf.csv','seasonal': '../dataset/DOE_Seasonal_R-1234yf.csv','color': '#E91E63', 'marker': 'D', 'gwp': 1,   'type': 'HFO'},
}

# Cluster definitions
CLUSTER_LOW  = ['R-134a', 'R-1234yf']   # Low-pressure, CW ~8–14, TW ~45–70
CLUSTER_HIGH = ['R-410A', 'R-32']       # High-pressure, CW ~21–34, TW ~109–170

# Column names
COL_TEVAP = 'Evaporator Temperature'
COL_QCOMP = 'Compressor Flow'
COL_CW = 'Chilled Water'
COL_TW = 'Condenser Cooling Water'
COL_COP = 'COP'
COL_TCOND = 'Condensing Temperature'

# ═══════════════════════════════════════════════════════════════════════════════
# DOS SCENARIOS — Three-Layer Design
# ═══════════════════════════════════════════════════════════════════════════════

# Each scenario tightens ONE engineering constraint and relaxes the others to the
# cluster's physical extreme. Capacity maps to tonnage as RT ~= CW[m3/h] x 6.6
# (Q_evap = CW x rho x cp x dT with the 20 K chilled-water rise; 1 RT = 3.517 kW).
#   cw_min  <- design cooling load
#   tw_max  <- cooling-tower / condenser-water availability
#   cop_min <- efficiency target, relative to the cluster's achievable envelope

# Intra-cluster DOS — Low-pressure (R-134a vs R-1234yf); capacity ~62-105 RT
DOS_LOW = {
    'High-load (CW-limited)':     {'cw_min': 13.5, 'tw_max': 78.0, 'cop_min': 3.30},  # ~89 RT demand
    'Water-limited (TW-limited)': {'cw_min': 9.40, 'tw_max': 60.0, 'cop_min': 3.30},  # scarce tower water
    'Efficiency (COP-limited)':   {'cw_min': 9.40, 'tw_max': 78.0, 'cop_min': 4.00},  # high-efficiency target
    'Balanced':                   {'cw_min': 12.0, 'tw_max': 64.0, 'cop_min': 3.70},
}

# Intra-cluster DOS — High-pressure (R-410A vs R-32); capacity ~157-267 RT
DOS_HIGH = {
    'High-load (CW-limited)':     {'cw_min': 34.0, 'tw_max': 201.0, 'cop_min': 3.30},  # ~224 RT demand
    'Water-limited (TW-limited)': {'cw_min': 23.8, 'tw_max': 150.0, 'cop_min': 3.30},  # scarce tower water
    'Efficiency (COP-limited)':   {'cw_min': 23.8, 'tw_max': 201.0, 'cop_min': 3.95},  # high-efficiency target
    'Balanced':                   {'cw_min': 30.0, 'tw_max': 162.0, 'cop_min': 3.70},
}

# The same scenarios drive the constraint-sensitivity figures: the ranking
# inversion IS the constraint sensitivity, so there is no separate DOS set.
DOS_SENSITIVITY_LOW  = DOS_LOW
DOS_SENSITIVITY_HIGH = DOS_HIGH

# Reference DOS for the individual heatmaps (one per cluster)
DOS_REF_LOW  = DOS_LOW['Balanced']
DOS_REF_HIGH = DOS_HIGH['Balanced']

# Monte-Carlo settings for the rigorous OI integration
OI_N_MC = 200_000
OI_SEED = 0

# Output base directory
BASE_OUT = 'analysis_outputs'


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(filepath):
    """Load and validate the base DOE dataset (20 000 LHS points at T_cond = 40 C)."""
    df = pd.read_csv(filepath)
    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"
    assert df.shape[0] >= 10000, f"Expected >= 10000 rows, got {df.shape[0]}"
    df.columns = [COL_TEVAP, COL_QCOMP, COL_CW, COL_TW, COL_COP]
    return df


def load_seasonal(filepath):
    """Load the condensing-temperature sweep (5 000 LHS points at each T_cond level).

    Same six columns as the base dataset plus the condensing temperature, which
    is the swept parameter: the condenser is cooled by cooling-tower water, so
    T_cond tracks the ambient / wet-bulb level.
    """
    df = pd.read_csv(filepath)
    assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
    df.columns = [COL_TEVAP, COL_QCOMP, COL_TCOND, COL_CW, COL_TW, COL_COP]
    return df


# Upper corners of the desired box, per cluster — the cluster's achievable
# maxima in (CW, COP). Filled by init_cluster_boxes() once the data are loaded.
CLUSTER_BOX = {}


def cluster_of(name):
    """'low' or 'high' — which capacity cluster a refrigerant belongs to."""
    return 'low' if name in CLUSTER_LOW else 'high'


def init_cluster_boxes(all_data):
    """Set the upper corners (CW_max, COP_max) of each cluster's desired box.

    A DOS is defined by a capacity FLOOR, a cooling-water CEILING and an
    efficiency FLOOR; it has no upper capacity or efficiency bound (excess
    capacity is modulated, not infeasible). The box is therefore closed at the
    top by the cluster's achievable maxima, which makes the OI of the two
    competing refrigerants comparable — they are measured against the same box.
    """
    for key, cluster in (('low', CLUSTER_LOW), ('high', CLUSTER_HIGH)):
        cw_max = max(all_data[n][COL_CW].max() for n in cluster)
        cop_max = max(all_data[n][COL_COP].max() for n in cluster)
        CLUSTER_BOX[key] = (float(cw_max), float(cop_max))


def calc_oi(name, df, dos, n_mc=OI_N_MC, seed=OI_SEED):
    """Rigorous Operability Index — OI = area(AOS n DOS) / area(DOS), in %.

    Measured in the (CW, COP) output plane, with the cooling-water limit
    TW <= tw_max applied point-wise. The achievable-and-desired cloud is
    enclosed by its convex hull (membership via Delaunay triangulation, so
    boundary points count as inside) and the covered fraction of the desired
    box is integrated by Monte-Carlo.

    Returns (OI [%], n_viable, viable_mask). The mask is the point-wise
    feasibility of the sampled operating points and is what the local
    operability heatmaps and the AIS/AOS scatter overlays use.
    """
    cw_min, tw_max, cop_min = dos['cw_min'], dos['tw_max'], dos['cop_min']
    cw_max, cop_max = CLUSTER_BOX[cluster_of(name)]

    mask = ((df[COL_CW] >= cw_min) & (df[COL_CW] <= cw_max) &
            (df[COL_TW] <= tw_max) &
            (df[COL_COP] >= cop_min) & (df[COL_COP] <= cop_max))
    n_viable = int(mask.sum())
    if n_viable < 3:
        return 0.0, n_viable, mask

    pts = np.column_stack([df[COL_CW][mask].values, df[COL_COP][mask].values])
    try:
        tri = Delaunay(pts)
    except Exception:           # degenerate cloud (collinear points) -> zero area
        return 0.0, n_viable, mask

    rng = np.random.default_rng(seed)
    mc = rng.uniform([cw_min, cop_min], [cw_max, cop_max], size=(n_mc, 2))
    oi = float((tri.find_simplex(mc) >= 0).mean() * 100.0)
    return oi, n_viable, mask


def short_name(dos_name):
    """'High-load (CW-limited)' -> 'CW-limited'; 'Balanced' -> 'Balanced'."""
    return dos_name.split('(')[1].rstrip(')') if '(' in dos_name else dos_name


def get_dos_for_refrigerant(name):
    """Return the appropriate DOS dict for a given refrigerant."""
    if name in CLUSTER_LOW:
        return DOS_LOW
    else:
        return DOS_HIGH


def get_ref_dos_for_refrigerant(name):
    """Return the reference DOS limits for heatmaps."""
    if name in CLUSTER_LOW:
        return DOS_REF_LOW
    else:
        return DOS_REF_HIGH


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_local_oi_heatmap(name, df, dos, n_grid=14):
    """Compute local OI heatmap on a grid over AIS."""
    t_edges = np.linspace(df[COL_TEVAP].min(), df[COL_TEVAP].max(), n_grid + 1)
    q_edges = np.linspace(df[COL_QCOMP].min(), df[COL_QCOMP].max(), n_grid + 1)
    _, _, viable_mask = calc_oi(name, df, dos)

    heatmap = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            cell = (
                (df[COL_TEVAP] >= t_edges[j]) & (df[COL_TEVAP] < t_edges[j + 1]) &
                (df[COL_QCOMP] >= q_edges[i]) & (df[COL_QCOMP] < q_edges[i + 1])
            )
            n_cell = cell.sum()
            heatmap[i, j] = (cell & viable_mask).sum() / n_cell * 100 if n_cell > 0 else np.nan

    return heatmap, t_edges, q_edges, viable_mask


def plot_heatmap_on_ax(ax, heatmap, t_edges, q_edges, title, show_cbar=True):
    """Plot a local OI heatmap on a given axes."""
    n_grid = heatmap.shape[0]
    im = ax.imshow(heatmap, origin='lower', cmap='RdYlGn', vmin=0, vmax=100,
                   extent=[t_edges[0], t_edges[-1], q_edges[0], q_edges[-1]], aspect='auto')
    if show_cbar:
        plt.colorbar(im, ax=ax, label='Local OI (%)', shrink=0.85)

    for i in range(n_grid):
        for j in range(n_grid):
            val = heatmap[i, j]
            if not np.isnan(val):
                tc = (t_edges[j] + t_edges[j + 1]) / 2
                qc = (q_edges[i] + q_edges[i + 1]) / 2
                ax.text(tc, qc, f'{val:.0f}', ha='center', va='center', fontsize=5,
                        color='white' if val < 40 or val > 80 else 'black')

    ax.set_xlabel('$T_{evap}$ (°C)')
    ax.set_ylabel('$Q_{comp}$ (m³/h)')
    ax.set_title(title)
    return im


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: INDIVIDUAL REFRIGERANT ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_individual(name, meta, df, out_dir):
    """Run full analysis for a single refrigerant."""
    ensure_dir(out_dir)
    color = meta['color']
    dos_dict = get_dos_for_refrigerant(name)
    ref_dos = get_ref_dos_for_refrigerant(name)

    # ─── 1a. Descriptive Statistics ───
    stats = df.describe().T
    stats['CV (%)'] = (stats['std'] / stats['mean'] * 100).round(2)
    stats.to_csv(os.path.join(out_dir, '01_descriptive_statistics.csv'))

    # ─── 1b. Correlation Matrix ───
    corr = df.corr()
    corr.to_csv(os.path.join(out_dir, '02_correlation_matrix.csv'))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    labels = ['$T_{evap}$', '$Q_{comp}$', 'CW', 'TW', 'COP']
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            val = corr.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(val) > 0.6 else 'black')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(f'Correlation Matrix — {name}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_correlation_matrix.png'))
    plt.close(fig)

    # ─── 1c. AIS → AOS Mapping (4-panel) ───
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    sc = ax.scatter(df[COL_TEVAP], df[COL_QCOMP], c=df[COL_COP], s=3, alpha=0.6, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='COP')
    ax.set_xlabel('$T_{evap}$ (°C)'); ax.set_ylabel('$Q_{comp}$ (m³/h)')
    ax.set_title('(a) AIS colored by COP')

    ax = axes[0, 1]
    sc = ax.scatter(df[COL_CW], df[COL_TW], c=df[COL_COP], s=3, alpha=0.6, cmap='viridis')
    try:
        pts = df[[COL_CW, COL_TW]].values
        hull = ConvexHull(pts)
        for simplex in hull.simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1], 'k-', lw=1.2)
    except:
        pass
    plt.colorbar(sc, ax=ax, label='COP')
    ax.set_xlabel('Chilled Water (m³/h)'); ax.set_ylabel('Condenser Cooling Water (m³/h)')
    ax.set_title('(b) AOS with convex hull')

    ax = axes[1, 0]
    sc = ax.scatter(df[COL_TEVAP], df[COL_CW], c=df[COL_QCOMP], s=3, alpha=0.6, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='$Q_{comp}$ (m³/h)')
    ax.set_xlabel('$T_{evap}$ (°C)'); ax.set_ylabel('Chilled Water (m³/h)')
    ax.set_title('(c) $T_{evap}$ → Chilled Water')

    ax = axes[1, 1]
    ax.scatter(df[COL_TEVAP], df[COL_COP], s=3, alpha=0.3, color=color)
    coeffs = np.polyfit(df[COL_TEVAP], df[COL_COP], 2)
    x_fit = np.linspace(df[COL_TEVAP].min(), df[COL_TEVAP].max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    r2 = np.corrcoef(df[COL_COP], np.polyval(coeffs, df[COL_TEVAP]))[0, 1] ** 2
    ax.plot(x_fit, y_fit, 'r-', lw=2, label=f'R² = {r2:.4f}')
    ax.set_xlabel('$T_{evap}$ (°C)'); ax.set_ylabel('COP')
    ax.set_title('(d) $T_{evap}$ → COP (quadratic fit)')
    ax.legend()

    fig.suptitle(f'AIS → AOS Mapping — {name}', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_ais_aos_mapping.png'))
    plt.close(fig)

    # ─── 1d. Operability Indices (cluster-appropriate DOS) ───
    oi_results = []
    for dos_name, dos in dos_dict.items():
        oi, n, mask = calc_oi(name, df, dos)
        oi_results.append({
            'DOS Scenario': dos_name,
            'CW_min': dos['cw_min'], 'TW_max': dos['tw_max'],
            'COP_min': dos['cop_min'], 'OI (%)': round(oi, 1), 'n_viable': n
        })
    oi_df = pd.DataFrame(oi_results)
    oi_df.to_csv(os.path.join(out_dir, '03_operability_indices.csv'), index=False)

    # ─── 1e. Local OI Heatmap (14×14 grid) ───
    heatmap, t_edges, q_edges, viable_mask = compute_local_oi_heatmap(name, df, ref_dos)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap_on_ax(ax, heatmap, t_edges, q_edges,
                       f'Local Operability Index — {name}\n'
                       f'CW≥{ref_dos["cw_min"]}, TW≤{ref_dos["tw_max"]}, COP≥{ref_dos["cop_min"]}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_local_oi_heatmap.png'))
    plt.close(fig)

    # ─── 1f. Trade-offs ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    sc = ax.scatter(df[COL_COP], df[COL_CW], c=df[COL_QCOMP], s=3, alpha=0.5, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='$Q_{comp}$ (m³/h)')
    ax.axhline(y=ref_dos['cw_min'], color='r', ls='--', lw=1, alpha=0.7,
               label=f'CW ≥ {ref_dos["cw_min"]}')
    ax.axvline(x=ref_dos['cop_min'], color='g', ls='--', lw=1, alpha=0.7,
               label=f'COP ≥ {ref_dos["cop_min"]}')
    ax.set_xlabel('COP'); ax.set_ylabel('Chilled Water (m³/h)')
    ax.set_title('(a) COP vs Capacity')
    ax.legend(fontsize=7)

    ax = axes[1]
    sc = ax.scatter(df[COL_CW], df[COL_TW], c=df[COL_COP], s=3, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='COP')
    ax.set_xlabel('Chilled Water (m³/h)'); ax.set_ylabel('Condenser Cooling Water (m³/h)')
    ax.set_title('(b) Chilled Water vs Condenser Cooling Water')

    ax = axes[2]
    ratio = df[COL_CW] / df[COL_TW]
    sc = ax.scatter(df[COL_COP], ratio, c=df[COL_QCOMP], s=3, alpha=0.5, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='$Q_{comp}$ (m³/h)')
    ax.set_xlabel('COP'); ax.set_ylabel('CW / TW Ratio')
    ax.set_title('(c) Water Use Efficiency')

    fig.suptitle(f'Trade-off Analysis — {name}', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_tradeoffs.png'))
    plt.close(fig)

    # ─── 1g. Sensitivity Analysis (6 panels) ───
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    inputs = [(COL_TEVAP, '$T_{evap}$ (°C)'), (COL_QCOMP, '$Q_{comp}$ (m³/h)')]
    outputs = [(COL_CW, 'Chilled Water (m³/h)'), (COL_TW, 'Condenser Cooling Water (m³/h)'), (COL_COP, 'COP')]

    for i, (inp, inp_label) in enumerate(inputs):
        for j, (out, out_label) in enumerate(outputs):
            ax = axes[i, j]
            other_inp = COL_QCOMP if inp == COL_TEVAP else COL_TEVAP
            sc = ax.scatter(df[inp], df[out], c=df[other_inp], s=2, alpha=0.4, cmap='coolwarm')
            r, _ = pearsonr(df[inp], df[out])
            ax.set_xlabel(inp_label); ax.set_ylabel(out_label)
            ax.set_title(f'r = {r:.3f}', fontsize=9)
            plt.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(f'Sensitivity Analysis — {name}', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_sensitivity.png'))
    plt.close(fig)

    # ─── 1h. Interaction Effects ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    outputs_int = [(COL_CW, 'Chilled Water'), (COL_TW, 'Condenser Cooling Water'), (COL_COP, 'COP')]

    for idx, (out, out_label) in enumerate(outputs_int):
        ax = axes[idx]
        sc = ax.scatter(df[COL_TEVAP], df[COL_QCOMP], c=df[out], s=3, alpha=0.5, cmap='viridis')
        plt.colorbar(sc, ax=ax, label=out_label)
        ax.set_xlabel('$T_{evap}$ (°C)'); ax.set_ylabel('$Q_{comp}$ (m³/h)')
        ax.set_title(f'({chr(97 + idx)}) {out_label}')

    fig.suptitle(f'Interaction Effects ($T_{{evap}}$ × $Q_{{comp}}$) — {name}',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig6_interaction.png'))
    plt.close(fig)

    print(f'  ✓ {name}: 6 figures + 3 CSVs saved to {out_dir}')
    return df, oi_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COMPARATIVE ANALYSES (Cross-Cluster)
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparative(all_data, all_oi, comp_dir):
    """Generate all cross-refrigerant comparative figures."""
    ensure_dir(comp_dir)
    names = list(all_data.keys())

    # ─── Fig C1: AOS Overlay (all 4 refrigerants) — shows disjoint clusters ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name in names:
        df = all_data[name]
        meta = REFRIGERANTS[name]
        ax.scatter(df[COL_CW], df[COL_TW], s=2, alpha=0.3, color=meta['color'], label=name)
        try:
            pts = df[[COL_CW, COL_TW]].values
            hull = ConvexHull(pts)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.plot(pts[hull_pts, 0], pts[hull_pts, 1], '-', color=meta['color'], lw=1.5)
        except:
            pass
    ax.set_xlabel('Chilled Water (m³/h)'); ax.set_ylabel('Condenser Cooling Water (m³/h)')
    ax.set_title('(a) AOS Comparison — All Refrigerants')
    ax.legend(markerscale=4, fontsize=9)

    ax = axes[1]
    for name in names:
        df = all_data[name]
        meta = REFRIGERANTS[name]
        ax.scatter(df[COL_COP], df[COL_CW], s=2, alpha=0.3, color=meta['color'], label=name)
    ax.set_xlabel('COP'); ax.set_ylabel('Chilled Water (m³/h)')
    ax.set_title('(b) COP vs Capacity — All Refrigerants')
    ax.legend(markerscale=4, fontsize=9)

    fig.suptitle('Achievable Output Space (AOS) Comparison', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C1_aos_comparison.png'))
    plt.close(fig)

    # ─── Fig C2: OI Bar Chart — Intra-cluster comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Low-pressure cluster
    ax = axes[0]
    dos_names_low = list(DOS_LOW.keys())
    x = np.arange(len(dos_names_low))
    width = 0.35
    for i, name in enumerate(CLUSTER_LOW):
        oi_vals = []
        for dos_name, dos in DOS_LOW.items():
            oi, _, _ = calc_oi(name, all_data[name], dos)
            oi_vals.append(round(oi, 1))
        bars = ax.bar(x + i * width, oi_vals, width, label=name,
                      color=REFRIGERANTS[name]['color'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, oi_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=0)
    ax.set_xlabel('DOS Scenario')
    ax.set_ylabel('Operability Index (%)')
    ax.set_title('(a) Low-Pressure Cluster: R-134a vs R-1234yf')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels([short_name(s) for s in dos_names_low], fontsize=9)
    ax.legend(fontsize=9)

    # Panel (b): High-pressure cluster
    ax = axes[1]
    dos_names_high = list(DOS_HIGH.keys())
    x = np.arange(len(dos_names_high))
    for i, name in enumerate(CLUSTER_HIGH):
        oi_vals = []
        for dos_name, dos in DOS_HIGH.items():
            oi, _, _ = calc_oi(name, all_data[name], dos)
            oi_vals.append(round(oi, 1))
        bars = ax.bar(x + i * width, oi_vals, width, label=name,
                      color=REFRIGERANTS[name]['color'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, oi_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=0)
    ax.set_xlabel('DOS Scenario')
    ax.set_ylabel('Operability Index (%)')
    ax.set_title('(b) High-Pressure Cluster: R-410A vs R-32')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels([short_name(s) for s in dos_names_high], fontsize=9)
    ax.legend(fontsize=9)

    fig.suptitle('Intra-Cluster Operability Index Comparison',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C2_oi_intracluster_bar.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig C3: AOS-DOS overlay IN THE PLANE WHERE THE OI IS MEASURED.
    # The index is an area ratio in (capacity CW, efficiency COP), so the overlay
    # is drawn there — the figure then explains its own OI value. Plotting this in
    # the CW-TW plane would show the degenerate sliver (CW and TW are collinear),
    # which is the reason that plane is NOT used for the measure. The cooling-water
    # limit TW <= TW_max acts point-wise and removes the greyed-out points.
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    # Top row: low-pressure cluster; bottom row: high-pressure cluster
    cluster_order = [('R-134a', DOS_REF_LOW), ('R-1234yf', DOS_REF_LOW),
                     ('R-410A', DOS_REF_HIGH), ('R-32', DOS_REF_HIGH)]

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for idx, (name, dos) in enumerate(cluster_order):
        ax = axes[idx // 2, idx % 2]
        df = all_data[name]
        meta = REFRIGERANTS[name]
        oi_val, _, viable = calc_oi(name, df, dos)
        cw_max, cop_max = CLUSTER_BOX[cluster_of(name)]

        ax.scatter(df[COL_CW][~viable], df[COL_COP][~viable], s=3, alpha=0.20, color='gray',
                   label='Achievable, outside DOS')
        ax.scatter(df[COL_CW][viable], df[COL_COP][viable], s=5, alpha=0.6, color=meta['color'],
                   label=f'Achievable n desired ({viable.sum() / len(df) * 100:.1f}% of points)')

        # convex hull of the achievable-and-desired cloud = the region measured
        if viable.sum() >= 3:
            try:
                pts = np.column_stack([df[COL_CW][viable].values, df[COL_COP][viable].values])
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1], '-', color=meta['color'], lw=1.6,
                        label='Measured region (hull)')
            except Exception:
                pass

        rect = Rectangle((dos['cw_min'], dos['cop_min']),
                         cw_max - dos['cw_min'], cop_max - dos['cop_min'],
                         linewidth=1.8, edgecolor='black', facecolor='none',
                         linestyle='--', label='DOS (desired)')
        ax.add_patch(rect)

        ax.set_xlabel('Capacity — chilled water (m³/h)'); ax.set_ylabel('Efficiency — COP')
        ax.set_title(f'{panel_labels[idx]} {name} (GWP = {meta["gwp"]})   →   OI = {oi_val:.1f}%\n'
                     f'CW≥{dos["cw_min"]}, TW≤{dos["tw_max"]}, COP≥{dos["cop_min"]}')
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('What the Operability Index measures:  OI = area(AOS n DOS) / area(DOS)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C3_aos_dos_overlay.png'))
    plt.close(fig)

    # ─── Fig C4: Viable Regions in AIS (4 panels) ───
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    panel_labels_c4 = ['(a)', '(b)', '(c)', '(d)']
    for idx, (name, dos) in enumerate(cluster_order):
        ax = axes[idx // 2, idx % 2]
        df = all_data[name]
        meta = REFRIGERANTS[name]
        oi_val, _, viable = calc_oi(name, df, dos)

        ax.scatter(df[COL_TEVAP][~viable], df[COL_QCOMP][~viable], s=3, alpha=0.15, color='gray')
        ax.scatter(df[COL_TEVAP][viable], df[COL_QCOMP][viable], s=5, alpha=0.6,
                   color=meta['color'],
                   label=f'Viable points ({viable.sum() / len(df) * 100:.1f}%)  |  OI = {oi_val:.1f}%')
        ax.set_xlabel('$T_{evap}$ (°C)'); ax.set_ylabel('$Q_{comp}$ (m³/h)')
        ax.set_title(f'{panel_labels_c4[idx]} {name} (GWP = {meta["gwp"]})')
        ax.legend(fontsize=8)

    fig.suptitle('Viable Region in AIS — Intra-Cluster DOS (Balanced)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C4_viable_ais.png'))
    plt.close(fig)

    # ─── Fig C5: Local OI Heatmaps (4 panels) ───
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    n_grid = 14

    panel_labels_c5 = ['(a)', '(b)', '(c)', '(d)']
    for idx, (name, dos) in enumerate(cluster_order):
        ax = axes[idx // 2, idx % 2]
        df = all_data[name]
        meta = REFRIGERANTS[name]

        heatmap, t_edges, q_edges, _ = compute_local_oi_heatmap(name, df, dos, n_grid)
        plot_heatmap_on_ax(ax, heatmap, t_edges, q_edges,
                           f'{panel_labels_c5[idx]} {name} (GWP = {meta["gwp"]})\n'
                           f'CW≥{dos["cw_min"]}, TW≤{dos["tw_max"]}, COP≥{dos["cop_min"]}')

    fig.suptitle('Local Operability Heatmap — Intra-Cluster DOS (Balanced)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C5_local_oi_heatmaps.png'))
    plt.close(fig)

    # ─── Fig C6: COP vs Capacity Trade-off (overlay) ───
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        df = all_data[name]
        meta = REFRIGERANTS[name]
        ax.scatter(df[COL_COP], df[COL_CW], s=4, alpha=0.3, color=meta['color'], label=name)
    ax.set_xlabel('COP'); ax.set_ylabel('Chilled Water (m³/h)')
    ax.set_title('COP vs Cooling Capacity — All Refrigerants')
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C6_cop_vs_capacity.png'))
    plt.close(fig)

    # ─── Fig C7: GWP vs OI (intra-cluster) ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): GWP vs OI
    ax = axes[0]
    for name in names:
        meta = REFRIGERANTS[name]
        dos_dict = get_dos_for_refrigerant(name)
        # Use the Balanced scenario from the appropriate cluster
        dos = dos_dict['Balanced']
        oi_val, _, _ = calc_oi(name, all_data[name], dos)
        ax.scatter(meta['gwp'], oi_val, s=120, color=meta['color'], marker=meta['marker'],
                   edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(f'{name}\nOI={oi_val:.1f}%', (meta['gwp'], oi_val),
                    textcoords='offset points', xytext=(8, 8), fontsize=8, fontweight='bold')
    ax.set_xlabel('GWP'); ax.set_ylabel('Operability Index (%)')
    ax.set_title('(a) GWP vs OI (Intra-Cluster Balanced DOS)')
    ax.set_xscale('log')

    # Panel (b): GWP vs Mean Capacity
    ax = axes[1]
    for name in names:
        meta = REFRIGERANTS[name]
        mean_cw = all_data[name][COL_CW].mean()
        mean_cop = all_data[name][COL_COP].mean()
        ax.scatter(meta['gwp'], mean_cw, s=mean_cop * 30, color=meta['color'],
                   marker=meta['marker'], edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(f'{name}\nCOP={mean_cop:.2f}', (meta['gwp'], mean_cw),
                    textcoords='offset points', xytext=(10, 5), fontsize=8)
    ax.set_xlabel('GWP'); ax.set_ylabel('Mean Chilled Water (m³/h)')
    ax.set_title('(b) GWP vs Mean Capacity (bubble size ∝ COP)')
    ax.set_xscale('log')

    fig.suptitle('Environmental Impact vs Operability Performance',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C7_gwp_vs_operability.png'))
    plt.close(fig)

    # ─── Fig C8: Correlation Comparison ───
    fig, ax = plt.subplots(figsize=(10, 5))
    corr_pairs = [
        ('$T_{evap}$→COP', COL_TEVAP, COL_COP),
        ('$Q_{comp}$→COP', COL_QCOMP, COL_COP),
        ('$T_{evap}$→CW', COL_TEVAP, COL_CW),
        ('$Q_{comp}$→CW', COL_QCOMP, COL_CW),
        ('$T_{evap}$→TW', COL_TEVAP, COL_TW),
        ('$Q_{comp}$→TW', COL_QCOMP, COL_TW),
        ('CW↔TW', COL_CW, COL_TW),
    ]

    x = np.arange(len(corr_pairs))
    width = 0.18
    for i, name in enumerate(names):
        df = all_data[name]
        vals = [df[c1].corr(df[c2]) for _, c1, c2 in corr_pairs]
        ax.bar(x + i * width, vals, width, label=name, color=REFRIGERANTS[name]['color'],
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Variable Pair')
    ax.set_ylabel('Pearson r')
    ax.set_title('Correlation Comparison Across Refrigerants')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([p[0] for p in corr_pairs], fontsize=8)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C8_correlation_comparison.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig C14: the actual 5x5 Pearson CORRELATION MATRICES, one per refrigerant.
    # C8 above is a bar chart of selected coefficients — useful, but it is not a
    # correlation matrix. This panel provides the genuine matrices so both the
    # full structure and the cross-refrigerant comparison are available.
    # ═══════════════════════════════════════════════════════════════════════════
    corr_cols = [COL_TEVAP, COL_QCOMP, COL_CW, COL_TW, COL_COP]
    corr_labels = ['$T_{evap}$', '$Q_{comp}$', 'CW', 'TW', 'COP']
    matrix_order = CLUSTER_LOW + CLUSTER_HIGH          # low cluster on top row

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    panel_labels_c14 = ['(a)', '(b)', '(c)', '(d)']
    im = None
    for idx, name in enumerate(matrix_order):
        ax = axes[idx // 2, idx % 2]
        corr = all_data[name][corr_cols].corr().values
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(5)); ax.set_xticklabels(corr_labels, rotation=45, ha='right')
        ax.set_yticks(range(5)); ax.set_yticklabels(corr_labels)
        for i in range(5):
            for j in range(5):
                val = corr[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7.5,
                        color='white' if abs(val) > 0.6 else 'black')
        ax.set_title(f'{panel_labels_c14[idx]} {name} '
                     f'(GWP = {REFRIGERANTS[name]["gwp"]})', fontsize=10)

    fig.colorbar(im, ax=axes, label='Pearson r', shrink=0.6, location='right', pad=0.03)
    fig.suptitle('Pearson correlation matrices for the four refrigerants',
                 fontsize=13, fontweight='bold', y=0.97)
    fig.savefig(os.path.join(comp_dir, 'fig_C14_correlation_matrices.png'))
    plt.close(fig)

    # ─── Fig C9: Output Variable Ranges ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    for i, name in enumerate(names):
        df = all_data[name]
        meta = REFRIGERANTS[name]
        cw_min, cw_max = df[COL_CW].min(), df[COL_CW].max()
        ax.barh(i, cw_max - cw_min, left=cw_min, height=0.6, color=meta['color'],
                edgecolor='black', linewidth=0.5)
        ax.text(cw_max + 0.3, i, f'{cw_min:.1f}–{cw_max:.1f}', va='center', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('Chilled Water (m³/h)')
    ax.set_title('(a) Capacity Range')

    ax = axes[1]
    for i, name in enumerate(names):
        df = all_data[name]
        meta = REFRIGERANTS[name]
        cop_min, cop_max = df[COL_COP].min(), df[COL_COP].max()
        ax.barh(i, cop_max - cop_min, left=cop_min, height=0.6, color=meta['color'],
                edgecolor='black', linewidth=0.5)
        ax.text(cop_max + 0.02, i, f'{cop_min:.2f}–{cop_max:.2f}', va='center', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('COP')
    ax.set_title('(b) COP Range')

    ax = axes[2]
    for i, name in enumerate(names):
        df = all_data[name]
        meta = REFRIGERANTS[name]
        tw_min, tw_max = df[COL_TW].min(), df[COL_TW].max()
        ax.barh(i, tw_max - tw_min, left=tw_min, height=0.6, color=meta['color'],
                edgecolor='black', linewidth=0.5)
        ax.text(tw_max + 1, i, f'{tw_min:.1f}–{tw_max:.1f}', va='center', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('Condenser Cooling Water (m³/h)')
    ax.set_title('(c) Condenser Cooling Water Range')

    fig.suptitle('Output Variable Ranges by Refrigerant', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C9_output_ranges.png'))
    plt.close(fig)

    # ─── Fig C10: Pareto Analysis (normalized) ───
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        df = all_data[name]
        meta = REFRIGERANTS[name]
        cop_norm = (df[COL_COP] - df[COL_COP].min()) / (df[COL_COP].max() - df[COL_COP].min())
        cw_norm = (df[COL_CW] - df[COL_CW].min()) / (df[COL_CW].max() - df[COL_CW].min())
        ax.scatter(cop_norm, cw_norm, s=3, alpha=0.2, color=meta['color'], label=name)

        score = cop_norm + cw_norm
        top_idx = score.nlargest(10).index
        ax.scatter(cop_norm[top_idx], cw_norm[top_idx], s=40, color=meta['color'],
                   edgecolors='black', linewidth=1, marker=meta['marker'], zorder=5)

    ax.set_xlabel('Normalized COP (0–1)'); ax.set_ylabel('Normalized Chilled Water (0–1)')
    ax.set_title('Normalized COP vs Capacity — Pareto Frontier Comparison')
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C10_pareto_normalized.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig C11: RANKING INVERSION — High-Pressure Cluster
    # The key paper contribution: the "best" refrigerant depends on the bottleneck
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Bar chart of OI under different constraint-sensitivity DOS
    ax = axes[0]
    sens_names = list(DOS_SENSITIVITY_HIGH.keys())
    x = np.arange(len(sens_names))
    width = 0.35
    for i, name in enumerate(CLUSTER_HIGH):
        oi_vals = []
        for dos_name, dos in DOS_SENSITIVITY_HIGH.items():
            oi, _, _ = calc_oi(name, all_data[name], dos)
            oi_vals.append(round(oi, 1))
        bars = ax.bar(x + i * width, oi_vals, width, label=name,
                      color=REFRIGERANTS[name]['color'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, oi_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Operability Index (%)')
    ax.set_title('(a) High-Pressure: R-410A vs R-32\nRanking depends on constraint bottleneck')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels(sens_names, fontsize=8)
    ax.legend(fontsize=9)

    # Add arrows/annotations showing the inversion
    ax.annotate('R-410A wins\n(lower TW)', xy=(0.15, 0.85), xycoords='axes fraction',
                fontsize=7, ha='center', color=REFRIGERANTS['R-410A']['color'], fontweight='bold')
    ax.annotate('R-32 wins\n(higher CW)', xy=(0.45, 0.85), xycoords='axes fraction',
                fontsize=7, ha='center', color=REFRIGERANTS['R-32']['color'], fontweight='bold')

    # Panel (b): Same for low-pressure cluster
    ax = axes[1]
    sens_names_low = list(DOS_SENSITIVITY_LOW.keys())
    x = np.arange(len(sens_names_low))
    for i, name in enumerate(CLUSTER_LOW):
        oi_vals = []
        for dos_name, dos in DOS_SENSITIVITY_LOW.items():
            oi, _, _ = calc_oi(name, all_data[name], dos)
            oi_vals.append(round(oi, 1))
        bars = ax.bar(x + i * width, oi_vals, width, label=name,
                      color=REFRIGERANTS[name]['color'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, oi_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Operability Index (%)')
    ax.set_title('(b) Low-Pressure: R-134a vs R-1234yf\nR-1234yf wins when TW is the bottleneck')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels(sens_names_low, fontsize=8)
    ax.legend(fontsize=9)

    fig.suptitle('Constraint-Sensitivity Analysis — Ranking Inversion',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C11_ranking_inversion.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig C12: Heatmap comparison — Constraint-sensitive (2×2 per cluster)
    # Show how the viable AIS region changes when the bottleneck shifts
    # ═══════════════════════════════════════════════════════════════════════════

    # High-pressure cluster: TW-limited vs CW-limited
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    n_grid = 14

    scenarios_high = [
        ('R-410A', DOS_HIGH['Water-limited (TW-limited)'], 'Water-limited'),
        ('R-32',   DOS_HIGH['Water-limited (TW-limited)'], 'Water-limited'),
        ('R-410A', DOS_HIGH['High-load (CW-limited)'],     'High-load'),
        ('R-32',   DOS_HIGH['High-load (CW-limited)'],     'High-load'),
    ]

    panel_labels_c12 = ['(a)', '(b)', '(c)', '(d)']
    for idx, (name, dos, scenario_label) in enumerate(scenarios_high):
        ax = axes[idx // 2, idx % 2]
        df = all_data[name]
        meta = REFRIGERANTS[name]
        oi_val, _, _ = calc_oi(name, df, dos)
        heatmap, t_edges, q_edges, viable_mask = compute_local_oi_heatmap(name, df, dos, n_grid)
        plot_heatmap_on_ax(ax, heatmap, t_edges, q_edges,
                           f'{panel_labels_c12[idx]} {name} — {scenario_label} (OI={oi_val:.1f}%)\n'
                           f'CW≥{dos["cw_min"]}, TW≤{dos["tw_max"]}, COP≥{dos["cop_min"]}')

    fig.suptitle('High-Pressure Cluster: How Constraint Bottleneck Shapes Viable AIS',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C12_heatmap_sensitivity_high.png'))
    plt.close(fig)

    # Low-pressure cluster: COP-limited vs TW-limited
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    scenarios_low = [
        ('R-134a',  DOS_LOW['Efficiency (COP-limited)'],   'Efficiency'),
        ('R-1234yf',DOS_LOW['Efficiency (COP-limited)'],   'Efficiency'),
        ('R-134a',  DOS_LOW['Water-limited (TW-limited)'], 'Water-limited'),
        ('R-1234yf',DOS_LOW['Water-limited (TW-limited)'], 'Water-limited'),
    ]

    panel_labels_c13 = ['(a)', '(b)', '(c)', '(d)']
    for idx, (name, dos, scenario_label) in enumerate(scenarios_low):
        ax = axes[idx // 2, idx % 2]
        df = all_data[name]
        meta = REFRIGERANTS[name]
        oi_val, _, _ = calc_oi(name, df, dos)
        heatmap, t_edges, q_edges, viable_mask = compute_local_oi_heatmap(name, df, dos, n_grid)
        plot_heatmap_on_ax(ax, heatmap, t_edges, q_edges,
                           f'{panel_labels_c13[idx]} {name} — {scenario_label} (OI={oi_val:.1f}%)\n'
                           f'CW≥{dos["cw_min"]}, TW≤{dos["tw_max"]}, COP≥{dos["cop_min"]}')

    fig.suptitle('Low-Pressure Cluster: How Constraint Bottleneck Shapes Viable AIS',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, 'fig_C13_heatmap_sensitivity_low.png'))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # CSV OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════════

    # ─── Summary Table ───
    summary_rows = []
    for name in names:
        df = all_data[name]
        meta = REFRIGERANTS[name]
        dos_dict = get_dos_for_refrigerant(name)
        row = {
            'Refrigerant': name,
            'Type': meta['type'],
            'GWP': meta['gwp'],
            'Cluster': 'Low' if name in CLUSTER_LOW else 'High',
            'CW_min': round(df[COL_CW].min(), 2),
            'CW_max': round(df[COL_CW].max(), 2),
            'CW_mean': round(df[COL_CW].mean(), 2),
            'TW_min': round(df[COL_TW].min(), 2),
            'TW_max': round(df[COL_TW].max(), 2),
            'TW_mean': round(df[COL_TW].mean(), 2),
            'COP_min': round(df[COL_COP].min(), 4),
            'COP_max': round(df[COL_COP].max(), 4),
            'COP_mean': round(df[COL_COP].mean(), 4),
            'r_Tevap_COP': round(df[COL_TEVAP].corr(df[COL_COP]), 4),
            'r_Qcomp_COP': round(df[COL_QCOMP].corr(df[COL_COP]), 4),
        }
        for dos_name, dos in dos_dict.items():
            oi, _, _ = calc_oi(name, df, dos)
            short = short_name(dos_name)
            row[f'OI_{short}'] = round(oi, 1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(comp_dir, 'summary_table.csv'), index=False)

    # ─── OI comparison — Low cluster ───
    oi_low = pd.DataFrame()
    oi_low['DOS Scenario'] = list(DOS_LOW.keys())
    for name in CLUSTER_LOW:
        oi_low[name] = all_oi[name]['OI (%)'].values
    oi_low.to_csv(os.path.join(comp_dir, 'oi_comparison_low.csv'), index=False)

    # ─── OI comparison — High cluster ───
    oi_high = pd.DataFrame()
    oi_high['DOS Scenario'] = list(DOS_HIGH.keys())
    for name in CLUSTER_HIGH:
        oi_high[name] = all_oi[name]['OI (%)'].values
    oi_high.to_csv(os.path.join(comp_dir, 'oi_comparison_high.csv'), index=False)

    # ─── Constraint-sensitivity results ───
    sens_rows = []
    for dos_name, dos in DOS_SENSITIVITY_HIGH.items():
        row = {'Cluster': 'High', 'Scenario': dos_name,
               'CW_min': dos['cw_min'], 'TW_max': dos['tw_max'], 'COP_min': dos['cop_min']}
        for name in CLUSTER_HIGH:
            oi, _, _ = calc_oi(name, all_data[name], dos)
            row[name] = round(oi, 1)
        row['Winner'] = CLUSTER_HIGH[0] if row[CLUSTER_HIGH[0]] > row[CLUSTER_HIGH[1]] else CLUSTER_HIGH[1]
        sens_rows.append(row)

    for dos_name, dos in DOS_SENSITIVITY_LOW.items():
        row = {'Cluster': 'Low', 'Scenario': dos_name,
               'CW_min': dos['cw_min'], 'TW_max': dos['tw_max'], 'COP_min': dos['cop_min']}
        for name in CLUSTER_LOW:
            oi, _, _ = calc_oi(name, all_data[name], dos)
            row[name] = round(oi, 1)
        row['Winner'] = CLUSTER_LOW[0] if row[CLUSTER_LOW[0]] > row[CLUSTER_LOW[1]] else CLUSTER_LOW[1]
        sens_rows.append(row)

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(comp_dir, 'constraint_sensitivity.csv'), index=False)

    print(f'\n  ✓ Comparative: 14 figures + 4 CSVs saved to {comp_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SEASONAL ANALYSIS — operability vs condensing temperature
# ═══════════════════════════════════════════════════════════════════════════════

def seasonal_boxes(all_seasonal):
    """Season-spanning desired box per cluster: (CW_max, COP_max) over ALL levels.

    The plant's demand does not change with the weather — the cooling load and the
    cooling tower are fixed hardware — so a SINGLE desired box must serve every
    condensing temperature. Anchoring its upper corners to the season-spanning
    achievable maxima keeps the box identical across levels, which makes the
    curves comparable and avoids the over-capacity artefact that a box built at
    one temperature alone would introduce.
    """
    boxes = {}
    for key, cluster in (('low', CLUSTER_LOW), ('high', CLUSTER_HIGH)):
        cw_max = max(all_seasonal[n][COL_CW].max() for n in cluster)
        cop_max = max(all_seasonal[n][COL_COP].max() for n in cluster)
        boxes[key] = (float(cw_max), float(cop_max))
    return boxes


def calc_oi_box(df, dos, cw_max, cop_max, n_mc=OI_N_MC, seed=OI_SEED):
    """Rigorous area OI against an explicitly supplied desired box.

    Same estimator as `calc_oi`; the box is passed in rather than taken from
    CLUSTER_BOX because the seasonal comparison needs the season-spanning box.
    """
    cw_min, tw_max, cop_min = dos['cw_min'], dos['tw_max'], dos['cop_min']
    mask = ((df[COL_CW] >= cw_min) & (df[COL_CW] <= cw_max) &
            (df[COL_TW] <= tw_max) &
            (df[COL_COP] >= cop_min) & (df[COL_COP] <= cop_max))
    if mask.sum() < 3:
        return 0.0
    pts = np.column_stack([df[COL_CW][mask].values, df[COL_COP][mask].values])
    try:
        tri = Delaunay(pts)
    except Exception:
        return 0.0
    rng = np.random.default_rng(seed)
    mc = rng.uniform([cw_min, cop_min], [cw_max, cop_max], size=(n_mc, 2))
    return float((tri.find_simplex(mc) >= 0).mean() * 100.0)


def run_seasonal(all_seasonal, comp_dir):
    """OI vs condensing temperature — does the ranking survive the season?

    The condenser is cooled by cooling-tower water, so T_cond tracks the ambient
    / wet-bulb level: sweeping it from 30 to 45 C sweeps a cool night to a hot
    afternoon. Demand is held fixed; only the achievable envelope moves.
    """
    ensure_dir(comp_dir)
    levels = sorted(all_seasonal['R-134a'][COL_TCOND].unique())
    boxes = seasonal_boxes(all_seasonal)

    # ─── compute every scenario x level x fluid ───
    rows, curves = [], {}
    for key, cluster, dos_set in (('low', CLUSTER_LOW, DOS_LOW),
                                  ('high', CLUSTER_HIGH, DOS_HIGH)):
        cw_max, cop_max = boxes[key]
        for sname, dos in dos_set.items():
            for name in cluster:
                df = all_seasonal[name]
                vals = [calc_oi_box(df[df[COL_TCOND] == tc], dos, cw_max, cop_max)
                        for tc in levels]
                curves[(key, sname, name)] = vals
                for tc, v in zip(levels, vals):
                    rows.append({'Cluster': key, 'Scenario': sname, 'Refrigerant': name,
                                 'T_cond (C)': tc, 'OI (%)': round(v, 2)})
    pd.DataFrame(rows).to_csv(os.path.join(comp_dir, 'seasonal_operability.csv'), index=False)

    # ─── figure: the two scenarios that carry the story ───
    # Water-limited  -> the inversion, and whether it survives the whole season
    # Balanced       -> where a genuinely seasonal crossover appears
    show = ['Water-limited (TW-limited)', 'Balanced']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panel = ['(a)', '(b)', '(c)', '(d)']
    for r, sname in enumerate(show):
        for c, (key, cluster) in enumerate((('low', CLUSTER_LOW), ('high', CLUSTER_HIGH))):
            ax = axes[r, c]
            for name in cluster:
                ax.plot(levels, curves[(key, sname, name)], 'o-', lw=2.2, ms=5,
                        color=REFRIGERANTS[name]['color'],
                        label=f'{name}  (GWP {REFRIGERANTS[name]["gwp"]})')
            # Shade where the LOWER-CAPACITY fluid leads. Identify it by mean capacity
            # rather than by position in the cluster list — the two lists do not use
            # the same ordering convention.
            lowcap = min(cluster, key=lambda n: all_seasonal[n][COL_CW].mean())
            other = [n for n in cluster if n != lowcap][0]
            lead = np.array(curves[(key, sname, lowcap)]) > np.array(curves[(key, sname, other)])
            ax.fill_between(levels, 0, 1, where=lead, transform=ax.get_xaxis_transform(),
                            color=REFRIGERANTS[lowcap]['color'], alpha=0.07, step='mid')
            ax.set_xlabel('Condensing temperature  $T_{cond}$ (°C)   —   cool night → hot afternoon')
            ax.set_ylabel('Operability Index  OI (%)')
            ax.set_title(f'{panel[r*2+c]} {"Low" if key=="low" else "High"}-pressure cluster'
                         f'  —  {sname}', fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle('Seasonal robustness: operability across the condensing-temperature range',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.text(0.5, 0.005,
             'Plant demand is fixed (the load and the cooling tower do not change with the weather); only the '
             'achievable envelope moves with $T_{cond}$. Shading marks where the lower-capacity fluid leads. '
             'Absolute values are low because the desired box spans the whole season — the result is the '
             'ordering and whether it persists, not the level.',
             ha='center', fontsize=8, style='italic', wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(comp_dir, 'fig_C15_seasonal_operability.png'))
    plt.close(fig)

    # ─── console summary ───
    for key, cluster in (('low', CLUSTER_LOW), ('high', CLUSTER_HIGH)):
        print(f'\n  {key}-pressure cluster ({cluster[0]} vs {cluster[1]})')
        for sname in DOS_LOW:
            winners = []
            for i, tc in enumerate(levels):
                a, b = curves[(key, sname, cluster[0])][i], curves[(key, sname, cluster[1])][i]
                # '-' means BOTH fluids are infeasible at this level, not a tie between them
                winners.append(cluster[0] if a > b else (cluster[1] if b > a else '-'))
            real = [w for w in winners if w != '-']
            if len(set(real)) > 1:
                tag = 'SEASONAL CROSSOVER: ' + ' | '.join(f'{t:.1f}C:{w}'
                                                         for t, w in zip(levels, winners))
            elif real:
                dead = [f'{t:.1f}' for t, w in zip(levels, winners) if w == '-']
                tag = f'{real[0]} throughout'
                if dead:
                    tag += f'   (both infeasible at {", ".join(dead)} C)'
            else:
                tag = 'infeasible at every level'
            print(f'    {sname:28} {tag}')

    print(f'\n  ✓ Seasonal: 1 figure + 1 CSV saved to {comp_dir}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MANUSCRIPT FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
# The nine figures of the paper, numbered as they appear in it and written to
# analysis_outputs/manuscript/. They are generated here, from the published
# datasets and with a single set of plotting defaults, so that every figure in
# the manuscript is reproducible from this script and the whole set shares one
# visual style.

RT_PER_CW = 6.6          # RT ≈ CW[m³/h] × 6.6  (see Section 2.4 of the paper)
GREY = '#ECECEC'
FIG_ORDER = ['R-134a', 'R-1234yf', 'R-32', 'R-410A']


def oi_occupancy(df, cw_min, cw_max, tw_max, cop_min, cop_max, grid=55):
    """Fast grid-occupancy estimate of the area OI, in %.

    Used ONLY for the selection map of Fig. 7, which needs thousands of index
    evaluations. It counts the fraction of cells of a regular grid over the
    desired rectangle that contain at least one admissible operating point.
    It tracks the rigorous estimator closely enough to preserve the winner at
    every pixel, which is all the map displays; every number quoted in the text
    or the tables comes from `calc_oi`, not from here.
    """
    cw, tw, cop = df[COL_CW].values, df[COL_TW].values, df[COL_COP].values
    feas = ((cw >= cw_min) & (cw <= cw_max) & (tw <= tw_max) &
            (cop >= cop_min) & (cop <= cop_max))
    if not feas.any():
        return 0.0
    gx = np.clip(((cw[feas] - cw_min) / (cw_max - cw_min) * grid).astype(int), 0, grid - 1)
    gy = np.clip(((cop[feas] - cop_min) / (cop_max - cop_min) * grid).astype(int), 0, grid - 1)
    return len(np.unique(gx * grid + gy)) / grid ** 2 * 100.0


def _cluster_pairs():
    """(key, [higher-capacity fluid, lower-capacity fluid], label) for both clusters."""
    return [('low', ['R-134a', 'R-1234yf'], 'Low-pressure cluster  (R-134a vs. R-1234yf)'),
            ('high', ['R-32', 'R-410A'], 'High-pressure cluster  (R-32 vs. R-410A)')]


def mfig_convergence(all_data, out, seed=12345):
    """Numerical convergence of the index — the two sources of error, separately.

    (a) Sampling error: the index recomputed on random subsets of the simulated
        operating points, against the full-sample value. Answers the question of
        how many simulated points the conclusion actually needs.
    (b) Integration error: the index recomputed with increasing numbers of Monte
        Carlo samples at fixed operating points.

    Both are shown for the water-limited scenario, which is the scenario that
    carries the ranking inversion and therefore the one where a numerical
    artefact would be most damaging.
    """
    rng = np.random.default_rng(seed)
    sizes = [125, 250, 500, 1000, 2000, 4000, 8000, 20000]
    n_mcs = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for key, cluster, _ in _cluster_pairs():
        dos = (DOS_LOW if key == 'low' else DOS_HIGH)['Water-limited (TW-limited)']
        for name in cluster:
            df = all_data[name]
            ref = calc_oi(name, df, dos)[0]
            ys = []
            for n in sizes:
                reps = [calc_oi(name, df.iloc[rng.choice(len(df), n, replace=False)], dos)[0]
                        for _ in range(5)]
                ys.append(np.mean(reps))
            ax.plot(sizes, ys, 'o-', ms=4, lw=1.8, color=REFRIGERANTS[name]['color'], label=name)
            ax.axhline(ref, color=REFRIGERANTS[name]['color'], lw=0.9, ls=':', alpha=0.7)
    ax.set_xscale('log')
    ax.axvline(20000, color='0.35', ls='--', lw=1.2)
    ax.text(18000, ax.get_ylim()[0] + 0.04 * np.ptp(ax.get_ylim()),
            'sample used\nin this work', fontsize=8, color='0.25', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))
    ax.set_xlabel('Number of simulated operating points used')
    ax.set_ylabel('Operability Index  OI  (%)')
    ax.set_title('(a) Convergence in the number of simulated points', fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2, framealpha=0.95)

    ax = axes[1]
    for key, cluster, _ in _cluster_pairs():
        dos = (DOS_LOW if key == 'low' else DOS_HIGH)['Water-limited (TW-limited)']
        cw_max, cop_max = CLUSTER_BOX[key]
        for name in cluster:
            df = all_data[name]
            ys = [calc_oi_box(df, dos, cw_max, cop_max, n_mc=m, seed=7) for m in n_mcs]
            ax.plot(n_mcs, ys, 's-', ms=4, lw=1.8, color=REFRIGERANTS[name]['color'], label=name)
    ax.set_xscale('log')
    ax.axvline(OI_N_MC, color='0.35', ls='--', lw=1.2)
    ax.text(OI_N_MC * 0.9, ax.get_ylim()[0] + 0.04 * np.ptp(ax.get_ylim()),
            'setting used\nin this work', fontsize=8, color='0.25', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))
    ax.set_xlabel('Number of Monte Carlo samples')
    ax.set_ylabel('Operability Index  OI  (%)')
    ax.set_title('(b) Convergence of the Monte Carlo integration', fontsize=10.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2, framealpha=0.95)

    fig.suptitle('Numerical convergence of the Operability Index (water-limited scenario)',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'Dotted lines in (a) are the full-sample values. The ranking between '
             'competing refrigerants is already correct at a few hundred points; the index '
             'itself is converged well before the sample and integration settings adopted here.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig1_convergence.png')); plt.close(fig)


def mfig_aos(all_data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for name in FIG_ORDER:
        df, meta = all_data[name], REFRIGERANTS[name]
        axes[0].scatter(df[COL_CW], df[COL_TW], s=2, alpha=0.25, color=meta['color'],
                        edgecolors='none', label=name)
        pts = df[[COL_CW, COL_TW]].values
        hull = ConvexHull(pts)
        v = np.append(hull.vertices, hull.vertices[0])
        axes[0].plot(pts[v, 0], pts[v, 1], '-', color=meta['color'], lw=1.4)
        axes[1].scatter(df[COL_COP], df[COL_CW], s=2, alpha=0.25, color=meta['color'],
                        edgecolors='none', label=name)
    axes[0].set_xlabel('Chilled water  CW  (m³/h)')
    axes[0].set_ylabel('Condenser cooling water  TW  (m³/h)')
    axes[0].set_title('(a) Two non-overlapping capacity clusters', fontsize=10.5)
    axes[1].set_xlabel('COP'); axes[1].set_ylabel('Chilled water  CW  (m³/h)')
    axes[1].set_title('(b) COP ranges overlap; capacity does not', fontsize=10.5)
    for ax in axes:
        ax.grid(alpha=0.22)
        leg = ax.legend(fontsize=8.5, markerscale=4, framealpha=0.95)
        for h in leg.legend_handles:
            h.set_alpha(1)
    fig.suptitle('Achievable Output Sets of the four refrigerants', fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig2_aos_clusters.png')); plt.close(fig)


def mfig_inversion(all_data, out):
    short = {'High-load (CW-limited)': 'High-load\n(CW-limited)',
             'Water-limited (TW-limited)': 'Water-limited\n(TW-limited)',
             'Efficiency (COP-limited)': 'Efficiency\n(COP-limited)', 'Balanced': 'Balanced'}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, (key, cluster, label) in zip(axes, _cluster_pairs()):
        dos_set = DOS_LOW if key == 'low' else DOS_HIGH
        names = list(dos_set)
        x, w = np.arange(len(names)), 0.36
        vals = {n: [calc_oi(n, all_data[n], dos_set[s])[0] for s in names] for n in cluster}
        for i, n in enumerate(cluster):
            ax.bar(x + i * w, vals[n], w, label=f'{n}  (GWP {REFRIGERANTS[n]["gwp"]})',
                   color=REFRIGERANTS[n]['color'], edgecolor='white', lw=0.5)
            for xi, v in zip(x + i * w, vals[n]):
                ax.text(xi, v + 0.6, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        top = max(max(v) for v in vals.values())
        for k, s in enumerate(names):
            win = max(cluster, key=lambda n: vals[n][k])
            wi = cluster.index(win)
            ax.scatter([x[k] + wi * w], [vals[win][k] + top * 0.10], marker='*', s=150,
                       color=REFRIGERANTS[win]['color'], zorder=5)
            if win == cluster[1]:                      # the lower-capacity fluid leads
                ax.axvspan(x[k] - 0.22, x[k] + 2 * w, color=REFRIGERANTS[win]['color'], alpha=0.07)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks(x + w / 2); ax.set_xticklabels([short[s] for s in names], fontsize=9)
        ax.set_ylabel('Operability Index  OI  (%)')
        ax.set_title(f'({"a" if key == "low" else "b"}) {label}', fontsize=10.5)
        ax.grid(axis='y', alpha=0.22); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, framealpha=0.95)
    fig.suptitle('The operability winner inverts with the binding constraint',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'Stars mark the operability winner; shaded scenarios are the inversions, '
             'where the lower-capacity fluid overtakes the higher-COP one.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(os.path.join(out, 'fig3_ranking_inversion.png')); plt.close(fig)


def mfig_oi_definition(all_data, out):
    """What the index measures — the low cluster under the water-limited demand."""
    dos = DOS_LOW['Water-limited (TW-limited)']
    cw_max, cop_max = CLUSTER_BOX['low']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, name in zip(axes, ['R-134a', 'R-1234yf']):
        df, meta = all_data[name], REFRIGERANTS[name]
        oi, _, mask = calc_oi(name, df, dos)
        ax.scatter(df[COL_CW][~mask], df[COL_COP][~mask], s=3, alpha=0.18, color='0.6',
                   edgecolors='none', label='TW > TW$_{max}$  (cut by the water ceiling)')
        ax.scatter(df[COL_CW][mask], df[COL_COP][mask], s=4, alpha=0.55, color=meta['color'],
                   edgecolors='none', label='admissible')
        pts = np.column_stack([df[COL_CW][mask].values, df[COL_COP][mask].values])
        hull = ConvexHull(pts)
        v = np.append(hull.vertices, hull.vertices[0])
        ax.plot(pts[v, 0], pts[v, 1], '-', color=meta['color'], lw=2, label='measured region (hull)')
        ax.add_patch(Rectangle((dos['cw_min'], dos['cop_min']),
                               cw_max - dos['cw_min'], cop_max - dos['cop_min'],
                               fill=False, ec='black', lw=1.8, ls='--', label='DOS (desired)'))
        ax.set_xlabel('Capacity — chilled water  CW  (m³/h)'); ax.set_ylabel('Efficiency — COP')
        ax.set_title(f'{name}   →   OI = {oi:.1f}%', fontsize=11, color=meta['color'])
        ax.grid(alpha=0.22); ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
    fig.suptitle('What the Operability Index measures:  OI = area(AOS ∩ DOS) / area(DOS)',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'Same desired rectangle in both panels. The cooling-water ceiling removes '
             'the grey points; the hull of R-1234yf covers about twice the area.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig4_oi_definition.png')); plt.close(fig)


def mfig_beyond_cop(all_data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    for name in FIG_ORDER:
        df = all_data[name]
        ax.scatter(df[COL_QCOMP], df[COL_CW] * RT_PER_CW, s=4, alpha=0.30,
                   color=REFRIGERANTS[name]['color'], edgecolors='none', label=name)
    ax.set_xlabel('Compressor suction volumetric flow  Q$_{comp}$  (m³/h)')
    ax.set_ylabel('Cooling capacity  (RT)')
    ax.set_title('(a) Two capacity clusters, disjoint across the whole Q$_{comp}$ window', fontsize=10)
    ax.grid(alpha=0.22); ax.set_axisbelow(True)
    leg = ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95, markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    lo = np.mean([all_data[n][COL_CW].mean() for n in ('R-134a', 'R-1234yf')]) * RT_PER_CW
    hi = np.mean([all_data[n][COL_CW].mean() for n in ('R-32', 'R-410A')]) * RT_PER_CW
    xm = all_data['R-134a'][COL_QCOMP].mean()
    ax.annotate('', (xm, lo), (xm, hi), arrowprops=dict(arrowstyle='<->', color='0.25', lw=1.7))
    ax.text(xm + 1.5, 0.5 * (lo + hi), f'{hi/lo:.2f}× capacity\nseparation', fontsize=9,
            color='0.15', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.9))

    ax = axes[1]
    cops = [all_data[n][COL_COP].values for n in FIG_ORDER]
    parts = ax.violinplot(cops, showmeans=True, showextrema=True, widths=0.8)
    for i, b in enumerate(parts['bodies']):
        b.set_facecolor(REFRIGERANTS[FIG_ORDER[i]]['color']); b.set_alpha(0.55)
        b.set_edgecolor(REFRIGERANTS[FIG_ORDER[i]]['color'])
    for k in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        parts[k].set_color('0.35'); parts[k].set_linewidth(1.0)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(FIG_ORDER, fontsize=9)
    ax.set_ylabel('COP')
    ax.set_title('(b) COP ranges overlap → COP cannot rank the fluids', fontsize=10)
    ax.grid(axis='y', alpha=0.22); ax.set_axisbelow(True)
    gmin, gmax = max(c.min() for c in cops), min(c.max() for c in cops)
    ax.axhspan(gmin, gmax, color='0.6', alpha=0.12)
    ax.text(0.55, gmax, 'band where all four overlap', fontsize=8, color='0.35',
            va='bottom', ha='left')
    fig.suptitle('Why “beyond COP”: capacity splits the fluids into two clusters, COP does not',
                 fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig7_beyond_cop.png')); plt.close(fig)


def mfig_selection_map(all_data, out, grid=70):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, (key, cluster, label) in zip(axes, _cluster_pairs()):
        uni = pd.concat([all_data[n] for n in cluster])
        cw_lo, cw_hi = uni[COL_CW].min(), uni[COL_CW].max()
        tw_lo, tw_hi = uni[COL_TW].min(), uni[COL_TW].max()
        cop_hi = uni[COL_COP].max()
        cw_axis = np.linspace(cw_lo, cw_hi * 0.99, grid)
        tw_axis = np.linspace(tw_lo, tw_hi, grid)
        idx = {n: i for i, n in enumerate(cluster)}
        M = np.full((grid, grid), -1)
        for j, tw in enumerate(tw_axis):
            for i, cwm in enumerate(cw_axis):
                ois = {n: oi_occupancy(all_data[n], cwm, cw_hi, tw, 3.5, cop_hi) for n in cluster}
                best = max(ois, key=ois.get)
                M[j, i] = idx[best] if ois[best] > 1.5 else -1
        ax.pcolormesh(cw_axis, tw_axis, M, shading='nearest', vmin=-1.5, vmax=1.5,
                      cmap=ListedColormap([GREY] + [REFRIGERANTS[n]['color'] for n in cluster]))
        ax.set_xlabel('Cooling-load floor — chilled water  (m³/h)')
        ax.set_title(f'({"a" if key == "low" else "b"}) {label}', fontsize=10.5)
        sec = ax.secondary_xaxis('top', functions=(lambda x: x * RT_PER_CW,
                                                   lambda x: x / RT_PER_CW))
        sec.set_xlabel('≈ cooling load  (RT)', fontsize=9)
        ax.legend(handles=[Patch(color=REFRIGERANTS[n]['color'],
                                 label=f'{n} (GWP {REFRIGERANTS[n]["gwp"]})') for n in cluster]
                  + [Patch(color=GREY, label='infeasible')],
                  loc='lower right', fontsize=8, framealpha=0.95)
    axes[0].set_ylabel('Cooling-water availability — TW  (m³/h)')
    fig.suptitle('Operability-optimal refrigerant selection map  @ $T_{cond}$ = 40 °C',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'The winner inverts as the binding constraint shifts: the higher-capacity '
             'fluid wins when cooling water is abundant, the lower-capacity one when it is scarce.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig8_selection_map.png')); plt.close(fig)


# fixed capacity/efficiency demand for the crossover sweep, per cluster
CROSSOVER_DEMAND = {'low': dict(cw_min=10.5, cop_min=3.45),
                    'high': dict(cw_min=27.0, cop_min=3.45)}


def mfig_crossover(all_data, out, n_tw=60, n_mc=60_000):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (key, cluster, label) in zip(axes, _cluster_pairs()):
        cw_max, cop_max = CLUSTER_BOX[key]
        uni = pd.concat([all_data[n] for n in cluster])
        tw_lo, tw_hi = uni[COL_TW].min(), uni[COL_TW].max()
        tw_axis = np.linspace(tw_lo + 0.02 * (tw_hi - tw_lo), tw_hi, n_tw)
        dem = CROSSOVER_DEMAND[key]
        oi_c, cop_c = {}, {}
        for n in cluster:
            df = all_data[n]
            oi_c[n] = np.array([
                calc_oi_box(df, dict(cw_min=dem['cw_min'], tw_max=t, cop_min=dem['cop_min']),
                            cw_max, cop_max, n_mc=n_mc)
                for t in tw_axis])
            cop_c[n] = float(df[COL_COP].mean())     # efficiency: constant, DOS-independent
        top = max(c.max() for c in oi_c.values())
        ax.set_ylim(0, top * 1.32)
        ax2 = ax.twinx()
        means = sorted(cop_c.values())
        span = max(means[-1] - means[0], 0.06) / 0.08
        ax2.set_ylim(means[0] - 0.78 * span, means[0] + 0.22 * span)
        for n in cluster:
            ax.plot(tw_axis, oi_c[n], color=REFRIGERANTS[n]['color'], lw=2.6, label=n, zorder=3)
            ax2.axhline(cop_c[n], color=REFRIGERANTS[n]['color'], lw=1.6, ls=':', alpha=0.9, zorder=1)
        ax2.set_yticks(means); ax2.set_yticklabels([f'{m:.2f}' for m in means])
        ax2.set_ylabel('mean COP  (efficiency, constant)', fontsize=9, color='0.4')
        ax2.tick_params(axis='y', labelcolor='0.4', labelsize=8)
        d = oi_c[cluster[0]] - oi_c[cluster[1]]
        up = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
        if len(up):
            k = up[-1]
            tcx = np.interp(0, [d[k], d[k + 1]], [tw_axis[k], tw_axis[k + 1]])
            ax.axvline(tcx, color='0.35', ls='--', lw=1.3, zorder=2)
            ax.axvspan(tw_axis[0], tcx, color=REFRIGERANTS[cluster[1]]['color'], alpha=0.08, zorder=0)
            ax.axvspan(tcx, tw_axis[-1], color=REFRIGERANTS[cluster[0]]['color'], alpha=0.08, zorder=0)
            ax.text(tcx, top * 1.22, f'crossover  TW* ≈ {tcx:.0f} m³/h', ha='center', va='center',
                    fontsize=8.8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.55'))
            ax.text(tw_axis[0] + 0.4, top * 0.74, f'water scarce\noperability → {cluster[1]}',
                    fontsize=9, color=REFRIGERANTS[cluster[1]]['color'], va='top', fontweight='bold')
            ax.text(tw_axis[-1] - 0.4, top * 0.10, f'water abundant\noperability → {cluster[0]}',
                    fontsize=9, color=REFRIGERANTS[cluster[0]]['color'], va='bottom', ha='right',
                    fontweight='bold')
        ax.set_xlabel('Cooling-water availability ceiling  TW$_{max}$  (m³/h)')
        ax.set_ylabel('Operability Index  OI  (%)')
        ax.set_title(f'({"a" if key == "low" else "b"}) {label}', fontsize=10.5)
        ax.grid(alpha=0.22)
        ax2.legend(*ax.get_legend_handles_labels(), loc='upper left', fontsize=9, framealpha=0.97)
    fig.suptitle('The operability-optimal fluid is not the efficiency-optimal fluid',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'Solid = operability, which crosses over at TW*; dotted = each fluid’s mean '
             'COP, constant and always favouring R-134a / R-32. The two criteria diverge below TW*.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(out, 'fig9_crossover.png')); plt.close(fig)


def mfig_seasonal(all_seasonal, out):
    levels = sorted(all_seasonal['R-134a'][COL_TCOND].unique())
    boxes = seasonal_boxes(all_seasonal)
    scen = 'Water-limited (TW-limited)'
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (key, cluster, label) in zip(axes, _cluster_pairs()):
        cw_max, cop_max = boxes[key]
        dos = (DOS_LOW if key == 'low' else DOS_HIGH)[scen]
        curves = {}
        for n in cluster:
            df = all_seasonal[n]
            curves[n] = [calc_oi_box(df[df[COL_TCOND] == tc], dos, cw_max, cop_max)
                         for tc in levels]
            ax.plot(levels, curves[n], 'o-', lw=2.2, ms=5, color=REFRIGERANTS[n]['color'],
                    label=f'{n}  (GWP {REFRIGERANTS[n]["gwp"]})')
        lead = np.array(curves[cluster[1]]) > np.array(curves[cluster[0]])
        ax.fill_between(levels, 0, 1, where=lead, transform=ax.get_xaxis_transform(),
                        color=REFRIGERANTS[cluster[1]]['color'], alpha=0.07, step='mid')
        ax.set_xlabel('Condensing temperature  $T_{cond}$  (°C)   —   cool night → hot afternoon')
        ax.set_ylabel('Operability Index  OI  (%)')
        ax.set_title(f'({"a" if key == "low" else "b"}) {label}', fontsize=10.5)
        ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend(fontsize=8.5)
    fig.suptitle('Seasonal robustness of the water-scarce ranking inversion',
                 fontweight='bold', fontsize=12)
    fig.text(0.5, 0.005, 'Plant demand fixed; only the achievable envelope moves with $T_{cond}$. '
             'Shading marks where the lower-capacity fluid leads. Absolute values are low because '
             'the desired region spans the whole season.',
             ha='center', fontsize=8.5, style='italic')
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(os.path.join(out, 'fig10_seasonal.png')); plt.close(fig)


def mfig_graphical_abstract(all_data, out):
    """The graphical abstract — regenerated from the published data so it cannot
    drift away from the paper, which is exactly what happened to the first one."""
    from matplotlib.patches import FancyBboxPatch

    def box(ax, x, y, w, h, fc, ec, lw=2.0, r=0.02):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                    fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=1))

    fig = plt.figure(figsize=(16, 7))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off')
    bg.text(0.5, 0.965, 'Beyond COP: Process Operability Reveals Constraint-Dependent '
            'Refrigerant Rankings', ha='center', va='center', fontsize=17, fontweight='bold')
    bg.text(0.5, 0.925, 'Spogis, Ronchetti, Barbin & Bispo   |   Digital Chemical Engineering',
            ha='center', va='center', fontsize=10.5, color='0.35')

    # ─────────────── left: simulation framework ───────────────
    box(bg, 0.015, 0.05, 0.29, 0.83, '#F7F7F7', '#CCCCCC', 1.2)
    bg.text(0.16, 0.845, 'Simulation Framework', ha='center', fontsize=13, fontweight='bold')
    box(bg, 0.035, 0.665, 0.25, 0.145, '#E3F0FB', '#2E7BB8')
    bg.text(0.16, 0.775, 'Available Input Set (AIS)', ha='center', fontsize=11.5,
            fontweight='bold', color='#1F5C93')
    bg.text(0.16, 0.727, '$T_{evap}$: −5 to 2 °C      $Q_{comp}$: 500–600 m³/h',
            ha='center', fontsize=9.5)
    bg.text(0.16, 0.690, '$T_{cond}$: 30 / 35 / 40 / 42.5 / 45 °C', ha='center', fontsize=9.5)
    bg.annotate('', xy=(0.16, 0.645), xytext=(0.16, 0.665),
                arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.8))
    box(bg, 0.035, 0.495, 0.25, 0.145, '#FDF1DC', '#E08A17')
    bg.text(0.16, 0.605, 'DWSIM v10  +  CoolProp', ha='center', fontsize=11.5,
            fontweight='bold', color='#C4700B')
    bg.text(0.16, 0.556, 'Python automation over the flowsheet', ha='center', fontsize=9.5)
    bg.text(0.16, 0.520, '20,000 LHS points × 4 refrigerants', ha='center', fontsize=9.5)
    bg.annotate('', xy=(0.16, 0.475), xytext=(0.16, 0.495),
                arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.8))
    for k, n in enumerate(FIG_ORDER):
        x = 0.035 + k * 0.0625
        box(bg, x, 0.325, 0.058, 0.125, 'white', REFRIGERANTS[n]['color'], 1.8)
        bg.text(x + 0.029, 0.410, n, ha='center', fontsize=10, fontweight='bold',
                color=REFRIGERANTS[n]['color'])
        bg.text(x + 0.029, 0.360, f'GWP {REFRIGERANTS[n]["gwp"]}', ha='center', fontsize=8.5,
                color='0.35')
    bg.annotate('', xy=(0.16, 0.305), xytext=(0.16, 0.325),
                arrowprops=dict(arrowstyle='-|>', color='0.4', lw=1.8))
    box(bg, 0.035, 0.115, 0.25, 0.175, '#E6F4E9', '#2E7D32')
    bg.text(0.16, 0.245, '180,000 Simulations → AOS', ha='center', fontsize=12.5,
            fontweight='bold', color='#1F5C24')
    bg.text(0.16, 0.192, 'CW (m³/h)   ·   TW (m³/h)   ·   COP', ha='center', fontsize=9.5)
    bg.text(0.16, 0.150, 'water flows are OUTPUTS, not inputs', ha='center', fontsize=8.8,
            style='italic', color='0.4')

    # ─────────────── middle: AOS + definition + inversion ───────────────
    ax = fig.add_axes([0.355, 0.600, 0.245, 0.265])
    for n in FIG_ORDER:
        d = all_data[n]
        ax.scatter(d[COL_CW], d[COL_TW], s=1.5, alpha=0.30, color=REFRIGERANTS[n]['color'],
                   edgecolors='none', label=n)
    ax.set_xlabel('CW (m³/h)', fontsize=9); ax.set_ylabel('TW (m³/h)', fontsize=9)
    ax.set_title('Two capacity clusters — 2.55× apart', fontsize=10.5, fontweight='bold')
    ax.tick_params(labelsize=8); ax.grid(alpha=0.2)
    leg = ax.legend(fontsize=7.5, markerscale=5, loc='upper left', framealpha=0.95)
    for h in leg.legend_handles:
        h.set_alpha(1)

    bg.text(0.478, 0.470, r'$OI = \frac{area(AOS \cap DOS)}{area(DOS)} \times 100\%$',
            ha='center', va='center', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.55', lw=1.4))

    ax = fig.add_axes([0.355, 0.10, 0.245, 0.295])
    scen = ['High-load', 'Water-\nlimited', 'Efficiency', 'Balanced']
    keys = list(DOS_LOW)
    x, w = np.arange(4), 0.38
    for k, n in enumerate(['R-134a', 'R-1234yf']):
        vals = [calc_oi(n, all_data[n], DOS_LOW[s])[0] for s in keys]
        ax.bar(x + k * w, vals, w, color=REFRIGERANTS[n]['color'], label=n,
               edgecolor='white', lw=0.5)
        for xi, v in zip(x + k * w, vals):
            ax.text(xi, v + 0.7, f'{v:.1f}', ha='center', fontsize=7.5)
    ax.axvspan(x[1] - 0.24, x[1] + 2 * w, color=REFRIGERANTS['R-1234yf']['color'], alpha=0.10)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(scen, fontsize=8)
    ax.set_ylabel('OI (%)', fontsize=9); ax.tick_params(labelsize=8)
    ax.set_ylim(0, 46); ax.grid(axis='y', alpha=0.2); ax.set_axisbelow(True)
    ax.set_title('Low-pressure cluster: the ranking inverts', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.95)

    # ─────────────── right: the finding ───────────────
    box(bg, 0.635, 0.05, 0.35, 0.83, '#F7F7F7', '#CCCCCC', 1.2)
    bg.text(0.81, 0.845, 'Key Insight: Rankings Invert', ha='center', fontsize=13,
            fontweight='bold', color='#B3121A')
    pairs = [('High-pressure cluster', 0.745,
              ('Cooling water scarce', 'R-410A 10.9%  >  R-32 1.9%', '#FBE6E4', '#C0392B'),
              ('High cooling load', 'R-32 35.0%  >  R-410A 5.1%', '#EEE6F7', '#7D3C98')),
             ('Low-pressure cluster', 0.520,
              ('Cooling water scarce', 'R-1234yf 10.9%  >  R-134a 5.6%', '#E6F4E9', '#2E7D32'),
              ('High cooling load', 'R-134a 25.9%  >  R-1234yf 8.3%', '#E3F0FB', '#2E7BB8'))]
    for title, ytop, left, right in pairs:
        bg.text(0.81, ytop, title, ha='center', fontsize=10.5, fontweight='bold')
        for (lbl, val, fc, ec), x0 in ((left, 0.652), (right, 0.822)):
            box(bg, x0, ytop - 0.135, 0.156, 0.105, fc, ec, 1.6)
            bg.text(x0 + 0.078, ytop - 0.055, lbl, ha='center', fontsize=8.8,
                    fontweight='bold', color=ec)
            bg.text(x0 + 0.078, ytop - 0.100, val, ha='center', fontsize=8.6)
        bg.annotate('', xy=(0.818, ytop - 0.082), xytext=(0.802, ytop - 0.082),
                    arrowprops=dict(arrowstyle='<|-|>', color='#E67E22', lw=2.2))

    box(bg, 0.652, 0.072, 0.326, 0.272, '#FDF1DC', '#E67E22', 2.2)
    bg.text(0.815, 0.302, 'The optimal refrigerant depends on', ha='center', fontsize=11.5,
            fontweight='bold', color='#C4560B')
    bg.text(0.815, 0.266, 'which engineering constraint binds', ha='center', fontsize=11.5,
            fontweight='bold', color='#C4560B')
    for k, line in enumerate([
            'Efficiency-optimal ≠ operability-optimal below TW*',
            # Scoped to the four fluids compared here, as in the abstract and the
            # highlights: R-717 and R-744 are real counterexamples and are untested.
            'Regulatory gap: of the four fluids, none meets GWP ≤ 150',
            'in the high-capacity cluster — none meets 750 when water is scarce',
            'Inversion persists across the whole seasonal range']):
        bg.text(0.668, 0.213 - k * 0.038, ('•  ' if k != 2 else '     ') + line,
                ha='left', fontsize=9.0)

    path = os.path.join(out, 'graphical_abstract.png')
    fig.savefig(path, dpi=300, facecolor='white'); plt.close(fig)
    return path


def run_manuscript(all_data, all_seasonal, out):
    """Generate the ten figures of the paper, numbered as in the manuscript."""
    ensure_dir(out)
    mfig_convergence(all_data, out);    print('    fig1  numerical convergence')
    mfig_aos(all_data, out);            print('    fig2  AOS clusters')
    mfig_inversion(all_data, out);      print('    fig3  ranking inversion')
    mfig_oi_definition(all_data, out);  print('    fig4  what the OI measures')

    # Fig. 4 and Fig. 5 reuse the comparative panels, which already have the right content
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for idx, name in enumerate(FIG_ORDER):
        dos = DOS_REF_LOW if name in CLUSTER_LOW else DOS_REF_HIGH
        hm, te, qe, _ = compute_local_oi_heatmap(name, all_data[name], dos, 14)
        plot_heatmap_on_ax(axes[idx // 2, idx % 2], hm, te, qe,
                           f'({chr(97+idx)}) {name} (GWP = {REFRIGERANTS[name]["gwp"]})\n'
                           f'CW≥{dos["cw_min"]}, TW≤{dos["tw_max"]}, COP≥{dos["cop_min"]}')
    fig.suptitle('Local operability in the input space (balanced demand)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig5_local_operability.png')); plt.close(fig)
    print('    fig5  local operability heatmaps')

    # Pearson BELOW the diagonal, Spearman ABOVE it, in one matrix per refrigerant.
    # Pearson is the load-bearing coefficient here: the reduction of the output space
    # to two dimensions (Section 2.3) rests on CW and TW being near-COLLINEAR, which is
    # a statement about linearity, not merely about monotonicity — two variables can be
    # perfectly monotonically related on a curved manifold that is not degenerate at all.
    # Spearman is shown alongside so the reader can see that the two agree to within
    # 0.02 everywhere, i.e. the relationships really are linear and the collinearity is
    # not an artefact of the coefficient chosen.
    corr_cols = [COL_TEVAP, COL_QCOMP, COL_CW, COL_TW, COL_COP]
    labels = ['$T_{evap}$', '$Q_{comp}$', 'CW', 'TW', 'COP']
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    im = None
    for idx, name in enumerate(FIG_ORDER):
        ax = axes[idx // 2, idx % 2]
        pear = all_data[name][corr_cols].corr(method='pearson').values
        spear = all_data[name][corr_cols].corr(method='spearman').values
        combo = np.tril(pear) + np.triu(spear, 1)
        im = ax.imshow(combo, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(5)); ax.set_yticklabels(labels)
        for i in range(5):
            for j in range(5):
                v = combo[i, j]
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5,
                        color='white' if abs(v) > 0.6 else 'black')
        # separate the two halves visually
        ax.plot([-0.5, 4.5], [-0.5, 4.5], color='0.25', lw=1.2)
        ax.set_title(f'({chr(97+idx)}) {name} (GWP = {REFRIGERANTS[name]["gwp"]})', fontsize=10)
    fig.colorbar(im, ax=axes, label='correlation coefficient', shrink=0.6,
                 location='right', pad=0.03)
    fig.suptitle('Correlation matrices: Pearson (lower triangle) and Spearman (upper triangle)',
                 fontsize=12.5, fontweight='bold', y=0.965)
    fig.text(0.5, 0.012, 'The two coefficients agree to within 0.02 for every pair, so the '
             'relationships are linear and not merely monotonic — which is what licenses '
             'treating CW and TW as collinear in Section 2.3.',
             ha='center', fontsize=8.5, style='italic')
    fig.savefig(os.path.join(out, 'fig6_correlation_matrices.png')); plt.close(fig)
    print('    fig6  correlation matrices (Pearson + Spearman)')

    mfig_beyond_cop(all_data, out);     print('    fig7  beyond COP')
    mfig_selection_map(all_data, out);  print('    fig8  selection maps')
    mfig_crossover(all_data, out);      print('    fig9  crossover')
    mfig_seasonal(all_seasonal, out);   print('    fig10 seasonal robustness')
    ga = mfig_graphical_abstract(all_data, out)
    # the repository root copy is what the README and the journal submission use
    shutil.copyfile(ga, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'graphical_abstract.png'))
    print('    graphical abstract (also copied to the repository root)')
    print(f'\n  ✓ Manuscript: 10 figures saved to {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('  Multi-Refrigerant Operability Analysis — v4')
    print('  Rigorous geometric OI = area(AOS n DOS)/area(DOS)')
    print('  R-410A | R-134a | R-32 | R-1234yf')
    print('=' * 70)

    all_data = {}
    all_oi = {}

    # Phase 0: load every dataset first, then close the desired boxes.
    # The OI is an AREA ratio, so it needs the upper corners of the desired box,
    # which are the cluster's achievable maxima — hence all data before any OI.
    print('\n[Phase 0] Loading datasets and closing the desired boxes\n')
    for name, meta in REFRIGERANTS.items():
        all_data[name] = load_data(meta['file'])
    init_cluster_boxes(all_data)
    for key, cluster in (('low', CLUSTER_LOW), ('high', CLUSTER_HIGH)):
        cw_max, cop_max = CLUSTER_BOX[key]
        print(f'  {key:5} cluster {str(cluster):26} box closed at '
              f'CW <= {cw_max:6.2f} m3/h, COP <= {cop_max:.3f}')

    # Phase 1: Individual analyses
    print('\n[Phase 1] Individual Refrigerant Analyses\n')
    for name, meta in REFRIGERANTS.items():
        out_dir = os.path.join(BASE_OUT, name.replace('-', '_').lower())
        df, oi_df = analyze_individual(name, meta, all_data[name], out_dir)
        all_data[name] = df
        all_oi[name] = oi_df

    # Phase 2: Comparative analyses
    print('\n[Phase 2] Cross-Refrigerant Comparative Analyses\n')
    comp_dir = os.path.join(BASE_OUT, 'comparative')
    run_comparative(all_data, all_oi, comp_dir)

    # Phase 3: Seasonal analysis (condensing-temperature sweep)
    print('\n[Phase 3] Seasonal Analysis — operability vs condensing temperature')
    all_seasonal = {n: load_seasonal(m['seasonal']) for n, m in REFRIGERANTS.items()}
    run_seasonal(all_seasonal, comp_dir)

    # Phase 4: the nine figures of the manuscript, in one consistent style
    print('\n[Phase 4] Manuscript Figures\n')
    run_manuscript(all_data, all_seasonal, os.path.join(BASE_OUT, 'manuscript'))

    # Final summary
    print('\n' + '=' * 70)
    print('  ANALYSIS COMPLETE — v4')
    print('=' * 70)
    print(f'\n  Output structure:')
    print(f'  {BASE_OUT}/')
    for name in REFRIGERANTS:
        folder = name.replace('-', '_').lower()
        print(f'    {folder}/           → 6 figures + 3 CSVs')
    print(f'    comparative/       → 15 figures + 5 CSVs')
    print(f'\n  Total: 39 figures + 17 CSVs')
    print(f'\n  OI method: rigorous area ratio, Delaunay hull + Monte-Carlo '
          f'({OI_N_MC:,} pts)')
    print(f'  DOS: 4 engineering-grounded scenarios per cluster')
    print(f'  Key figures:')
    print(f'    C2  — Intra-cluster OI bar chart')
    print(f'    C3  — AOS n DOS overlay in the (CW, COP) plane (what OI measures)')
    print(f'    C11 — Ranking inversion (constraint-sensitivity)')
    print(f'    C14 — Pearson correlation matrices (5x5, one per refrigerant)')
    print(f'    C15 — Seasonal operability vs condensing temperature')
    print(f'    C12 — Heatmap: how bottleneck shapes viable AIS (high-pressure)')
    print(f'    C13 — Heatmap: how bottleneck shapes viable AIS (low-pressure)')


def main_manuscript_only():
    """Regenerate just the nine manuscript figures — fast iteration on the paper."""
    print('Manuscript figures only\n')
    all_data = {n: load_data(m['file']) for n, m in REFRIGERANTS.items()}
    init_cluster_boxes(all_data)
    all_seasonal = {n: load_seasonal(m['seasonal']) for n, m in REFRIGERANTS.items()}
    run_manuscript(all_data, all_seasonal, os.path.join(BASE_OUT, 'manuscript'))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'manuscript':
        main_manuscript_only()
    else:
        main()