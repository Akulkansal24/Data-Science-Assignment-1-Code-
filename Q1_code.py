"""
CSE558 - Data Science Assignment 1
Random Projections & Sketching: JL for k-means Clustering (Question 1)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans

# Set style for figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = "results"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ==========================================
# 0. DATA PREPARATION
# ==========================================
print("=" * 60)
print("0. DATA PREPARATION")
print("=" * 60)

DATA_CACHE = "kdd_preprocessed.npz"

if os.path.exists(DATA_CACHE):
    print(f"Loading cached preprocessed data from {DATA_CACHE}...")
    cached = np.load(DATA_CACHE)
    D = cached['D']
    y = cached['y']
    feature_names = cached['feature_names'].tolist()
    const_cols = cached['const_cols'].tolist() if 'const_cols' in cached else ['num_outbound_cmds', 'is_host_login']
    n, d = D.shape
    print(f"Loaded D: shape=({n}, {d}), dtype={D.dtype}")
    print(f"Loaded y: shape={y.shape}, attack count={int(y.sum())}")
else:
    print("Fetching KDD Cup 1999 dataset (10% version)...")
    kdd = fetch_kddcup99(subset=None, as_frame=True)
    df = kdd.data.copy()
    print(f"Raw kdd.data shape: {df.shape}")
    
    # (i) One-hot encode the three text features protocol_type, service and flag,
    # and convert the other 38 features to float.
    text_cols = ['protocol_type', 'service', 'flag']
    other_cols = [c for c in df.columns if c not in text_cols]
    
    for c in other_cols:
        df[c] = df[c].astype(np.float32)
        
    df = pd.get_dummies(df, columns=text_cols, dtype=np.float32)
    print(f"Shape after one-hot encoding: {df.shape}")
    
    # (ii) Drop every column that takes the same value on all rows.
    nunique = df.nunique()
    const_cols = nunique[nunique == 1].index.tolist()
    print(f"Dropping constant columns ({len(const_cols)}): {const_cols}")
    df = df.drop(columns=const_cols)
    print(f"Shape after dropping constant columns: {df.shape}")
    
    # (iii) Rescale each column to [0,1] by x -> (x - min)/(max - min)
    mins = df.min()
    maxs = df.max()
    df = (df - mins) / (maxs - mins)
    
    D = df.to_numpy(dtype=np.float32)
    feature_names = df.columns.tolist()
    n, d = D.shape
    
    # Label: y_i = 1 if connection is attack (!= b'normal.'), 0 otherwise
    y = (kdd.target != b'normal.').to_numpy(dtype=np.float32)
    
    print(f"Preprocessed data matrix D in R^{n}x{d}, dtype={D.dtype}")
    print(f"Min: {D.min():.4f}, Max: {D.max():.4f}, Any NaN: {np.isnan(D).any()}")
    print(f"Target y: shape={y.shape}, sum(y)={int(y.sum())} attacks ({y.mean()*100:.2f}%)")
    
    np.savez_compressed(DATA_CACHE, D=D, y=y, feature_names=feature_names, const_cols=const_cols)
    print(f"Saved preprocessed data to {DATA_CACHE}")


def compute_kmeans_cost(X, centers):
    """
    cost(X, B) = sum_{i=1}^n min_{b in B} ||X_i - b||_2^2
    Centers: (k, d), X: (n, d)
    Uses batching to be memory-efficient and fast.
    """
    k = centers.shape[0]
    n = X.shape[0]
    # ||x - b||^2 = ||x||^2 + ||b||^2 - 2 x.b
    X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    centers_sq = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)
    
    chunk_size = 100000
    total_cost = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dist_sq = X_sq[start:end] + centers_sq - 2.0 * np.dot(X[start:end], centers.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        total_cost += np.sum(np.min(dist_sq, axis=1))
    return float(total_cost)


# ==========================================
# 1. JL FOR K-MEANS CLUSTERING
# ==========================================
print("\n" + "=" * 60)
print("1. JL FOR K-MEANS CLUSTERING (k = 15)")
print("=" * 60)

q1_results = {}

# (a) Run k-means on D; call its centres B and its clusters C_D.
print("\n--- 1(a) Running k-means on D ---")
t0 = time.perf_counter()
km_D = KMeans(n_clusters=15, n_init=5, random_state=GLOBAL_SEED)
km_D.fit(D)
t_km_D = time.perf_counter() - t0

B = km_D.cluster_centers_
C_D = km_D.labels_
cost_D_B = compute_kmeans_cost(D, B)
print(f"1(a) cost(D, B): {cost_D_B:.4f} (inertia: {km_D.inertia_:.4f})")
print(f"1(a) Wall-clock time: {t_km_D:.2f} seconds")

q1_results['1a'] = {
    'cost_D_B': cost_D_B,
    'inertia': float(km_D.inertia_),
    'time_D': float(t_km_D)
}

# (b) & (c) & (d) For each m in {5, 10, 15, 20, 25}, 5 runs each
m_list_q1 = [5, 10, 15, 20, 25]
runs_per_m = 5

q1_b_ratios = {m: [] for m in m_list_q1}
q1_b_costs = {m: [] for m in m_list_q1}
q1_c_ratios = {m: [] for m in m_list_q1}
q1_c_costs = {m: [] for m in m_list_q1}
q1_d_times = {m: [] for m in m_list_q1}
q1_d_proj_times = {m: [] for m in m_list_q1}
q1_d_km_times = {m: [] for m in m_list_q1}

print("\n--- 1(b, c, d) Random Projections and k-means on E ---")

# Seed sequence for reproducibility
q1_seeds = [100 + i for i in range(100)]
seed_idx = 0

for m in m_list_q1:
    print(f"\nProcessing m = {m} ({runs_per_m} runs)...")
    for r in range(runs_per_m):
        run_seed = q1_seeds[seed_idx]
        seed_idx += 1
        rng = np.random.default_rng(run_seed)
        
        # Draw A in R^{m x d} with i.i.d. N(0, 1/m)
        A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, d)).astype(np.float32)
        
        # Measure projection time explicitly: E = D A^T in R^{n x m}
        t0_proj = time.perf_counter()
        E = np.dot(D, A.T)
        t_proj = time.perf_counter() - t0_proj
        q1_d_proj_times[m].append(float(t_proj))
        
        # Run k-means on E
        t0_km = time.perf_counter()
        km_E = KMeans(n_clusters=15, n_init=5, random_state=run_seed)
        km_E.fit(E)
        t_km = time.perf_counter() - t0_km
        q1_d_km_times[m].append(float(t_km))
        
        # Total wall-clock time for computing E plus k-means on E
        t_run = t_proj + t_km
        q1_d_times[m].append(float(t_run))
        
        # Form B_hat: average original rows of D in each cluster
        labels_E = km_E.labels_
        B_hat = np.zeros((15, d), dtype=np.float32)
        for j in range(15):
            mask = (labels_E == j)
            if np.any(mask):
                B_hat[j] = np.mean(D[mask], axis=0)
            else:
                B_hat[j] = B[j]  # fallback if empty cluster
                
        # cost(D, B_hat)
        cost_B_hat = compute_kmeans_cost(D, B_hat)
        ratio_B_hat = cost_B_hat / cost_D_B
        q1_b_costs[m].append(float(cost_B_hat))
        q1_b_ratios[m].append(float(ratio_B_hat))
        
        # (c) B_plus: map each centre a in R^m back to R^d as A^+ a
        # A^+ = A^T (A A^T)^{-1}
        centers_E = km_E.cluster_centers_  # (15, m)
        A_pinv = np.linalg.pinv(A).astype(np.float32)  # (d, m)
        B_plus = np.dot(centers_E, A_pinv.T)  # (15, d)
        
        cost_B_plus = compute_kmeans_cost(D, B_plus)
        ratio_B_plus = cost_B_plus / cost_D_B
        q1_c_costs[m].append(float(cost_B_plus))
        q1_c_ratios[m].append(float(ratio_B_plus))
        
        print(f"  m={m:2d}, run {r+1}: total={t_run:.2f}s (proj={t_proj*1000:.1f}ms, km={t_km:.2f}s), ratio(B_hat)={ratio_B_hat:.4f}, cost(B_hat)={cost_B_hat:.1f}, ratio(B_plus)={ratio_B_plus:.4f}")

q1_results['1b'] = {
    'm_list': m_list_q1,
    'ratios': q1_b_ratios,
    'costs': q1_b_costs,
    'summary': {
        m: {
            'mean': float(np.mean(q1_b_ratios[m])),
            'min': float(np.min(q1_b_ratios[m])),
            'max': float(np.max(q1_b_ratios[m])),
            'mean_ratio': float(np.mean(q1_b_ratios[m])),
            'min_ratio': float(np.min(q1_b_ratios[m])),
            'max_ratio': float(np.max(q1_b_ratios[m])),
            'mean_cost': float(np.mean(q1_b_costs[m])),
            'min_cost': float(np.min(q1_b_costs[m])),
            'max_cost': float(np.max(q1_b_costs[m]))
        } for m in m_list_q1
    }
}

q1_results['1c'] = {
    'ratios': q1_c_ratios,
    'costs': q1_c_costs,
    'summary': {
        m: {
            'mean': float(np.mean(q1_c_ratios[m])),
            'min': float(np.min(q1_c_ratios[m])),
            'max': float(np.max(q1_c_ratios[m])),
            'mean_ratio': float(np.mean(q1_c_ratios[m])),
            'min_ratio': float(np.min(q1_c_ratios[m])),
            'max_ratio': float(np.max(q1_c_ratios[m])),
            'mean_cost': float(np.mean(q1_c_costs[m])),
            'min_cost': float(np.min(q1_c_costs[m])),
            'max_cost': float(np.max(q1_c_costs[m]))
        } for m in m_list_q1
    }
}

q1_results['1d'] = {
    'times': q1_d_times,
    'mean_times': {m: float(np.mean(q1_d_times[m])) for m in m_list_q1},
    'median_times': {m: float(np.median(q1_d_times[m])) for m in m_list_q1},
    'min_times': {m: float(np.min(q1_d_times[m])) for m in m_list_q1},
    'max_times': {m: float(np.max(q1_d_times[m])) for m in m_list_q1},
    'proj_times': q1_d_proj_times,
    'mean_proj_times': {m: float(np.mean(q1_d_proj_times[m])) for m in m_list_q1}
}

# (e) Pick 10^4 random pairs (i, l) with exact D_i != D_l
print("\n--- 1(e) Pairwise distance preservation vs Clustering cost preservation ---")
rng_pairs = np.random.default_rng(GLOBAL_SEED)
num_pairs = 10000
selected_pairs_i = []
selected_pairs_l = []

# Generate candidate pairs and filter out identical rows using exact test
batch_size = 20000
while len(selected_pairs_i) < num_pairs:
    idx_i = rng_pairs.integers(0, n, size=batch_size)
    idx_l = rng_pairs.integers(0, n, size=batch_size)
    valid_idx = (idx_i != idx_l)
    idx_i = idx_i[valid_idx]
    idx_l = idx_l[valid_idx]
    
    # Exact check: keep pairs where at least one feature differs
    distinct = np.any(D[idx_i] != D[idx_l], axis=1)
    selected_pairs_i.extend(idx_i[distinct])
    selected_pairs_l.extend(idx_l[distinct])

pairs_i = np.array(selected_pairs_i[:num_pairs], dtype=np.int64)
pairs_l = np.array(selected_pairs_l[:num_pairs], dtype=np.int64)
D_pair_diff = D[pairs_i] - D[pairs_l]
D_dist_sq = np.sum(D_pair_diff**2, axis=1)
print(f"Generated {num_pairs} strictly distinct pairs (D_i != D_l). Min ||D_i - D_l||^2: {D_dist_sq.min():.8e}, Max: {D_dist_sq.max():.6f}")

# Precompute cluster centers and centered matrix for C_D on D
mu_D = np.zeros((15, d), dtype=np.float32)
for j in range(15):
    mask = (C_D == j)
    if np.any(mask):
        mu_D[j] = np.mean(D[mask], axis=0)

# Centered matrix M where row i is D_i - mu_{C_D[i]}(D)
M = D - mu_D[C_D]
cost_D_C_D = float(np.sum(M**2))
print(f"cost_D(C_D): {cost_D_C_D:.4f}")

# Precompute M^T M in R^{d x d} for fast cost_E(C_D) computation
# cost_E(C_D) = ||M A^T||_F^2 = Tr(A M^T M A^T)
MtM = np.dot(M.T, M)  # (116, 116)
tr_MtM = float(np.trace(MtM))
tr_MtM2 = float(np.trace(np.dot(MtM, MtM)))
effective_rank = float((tr_MtM ** 2) / tr_MtM2)
print(f"Tr(MtM) = {tr_MtM:.4f}, Tr((MtM)^2) = {tr_MtM2:.4f}, Effective rank r = {effective_rank:.4f}")

q1_e_pairwise_ratios = {}
q1_e_clustering_ratios = {}

q1_e_seeds = [500 + i for i in range(500)]
e_seed_idx = 0

for m in m_list_q1:
    # (i) for one draw of A, compute ||E_i - E_l||_2^2 / ||D_i - D_l||_2^2
    rng_A_single = np.random.default_rng(q1_e_seeds[e_seed_idx])
    e_seed_idx += 1
    A_single = rng_A_single.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, d)).astype(np.float32)
    
    # E_i - E_l = (D_i - D_l) A^T
    E_pair_diff = np.dot(D_pair_diff, A_single.T)  # (10000, m)
    E_dist_sq = np.sum(E_pair_diff**2, axis=1)
    pair_ratio = (E_dist_sq / D_dist_sq).astype(float)
    q1_e_pairwise_ratios[m] = pair_ratio.tolist()
    
    # (ii) for 20 independent draws of A, compute cost_E(C_D)/cost_D(C_D)
    cluster_ratios = []
    for _ in range(20):
        rng_A_c = np.random.default_rng(q1_e_seeds[e_seed_idx])
        e_seed_idx += 1
        A_c = rng_A_c.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, d)).astype(np.float32)
        # cost_E(C_D) = Tr(A_c MtM A_c^T)
        cost_E_C_D = float(np.trace(np.dot(np.dot(A_c, MtM), A_c.T)))
        cluster_ratios.append(float(cost_E_C_D / cost_D_C_D))
        
    q1_e_clustering_ratios[m] = cluster_ratios
    theo_rel_std = float(np.sqrt(2.0 / (m * effective_rank)))
    theo_single_pair_std = float(np.sqrt(2.0 / m))
    print(f"m={m:2d}: Pairwise mean={np.mean(pair_ratio):.4f}, std={np.std(pair_ratio):.4f} | Cluster mean={np.mean(cluster_ratios):.4f}, std={np.std(cluster_ratios):.4f} (theo cluster std: {theo_rel_std:.4f}, single-pair std: {theo_single_pair_std:.4f})")

q1_results['1e'] = {
    'pairwise_summary': {
        m: {
            'mean': float(np.mean(q1_e_pairwise_ratios[m])),
            'std': float(np.std(q1_e_pairwise_ratios[m])),
            'min': float(np.min(q1_e_pairwise_ratios[m])),
            'max': float(np.max(q1_e_pairwise_ratios[m])),
            'p95': float(np.percentile(q1_e_pairwise_ratios[m], 95))
        }
        for m in m_list_q1
    },
    'clustering_summary': {
        m: {
            'mean': float(np.mean(q1_e_clustering_ratios[m])),
            'std': float(np.std(q1_e_clustering_ratios[m])),
            'min': float(np.min(q1_e_clustering_ratios[m])),
            'max': float(np.max(q1_e_clustering_ratios[m]))
        }
        for m in m_list_q1
    },
    'theory': {
        'tr_MtM': tr_MtM,
        'tr_MtM2': tr_MtM2,
        'effective_rank': effective_rank,
        'theoretical_rel_std': {
            m: float(np.sqrt(2.0 / (m * effective_rank))) for m in m_list_q1
        },
        'single_pair_rel_std': {
            m: float(np.sqrt(2.0 / m)) for m in m_list_q1
        }
    }
}

# ==========================================
# 2. GENERATE AND SAVE COLOURED PUBLICATION FIGURES
# ==========================================
print("\n" + "=" * 60)
print("2. GENERATING COLOURED PUBLICATION FIGURES")
print("=" * 60)

x_indices = np.arange(len(m_list_q1))

b_hat_means = np.array([q1_results['1b']['summary'][m]['mean'] for m in m_list_q1])
b_hat_mins = np.array([q1_results['1b']['summary'][m]['min'] for m in m_list_q1])
b_hat_maxs = np.array([q1_results['1b']['summary'][m]['max'] for m in m_list_q1])
yerr_b_hat = [b_hat_means - b_hat_mins, b_hat_maxs - b_hat_means]

b_plus_means = np.array([q1_results['1c']['summary'][m]['mean'] for m in m_list_q1])
b_plus_mins = np.array([q1_results['1c']['summary'][m]['min'] for m in m_list_q1])
b_plus_maxs = np.array([q1_results['1c']['summary'][m]['max'] for m in m_list_q1])
yerr_b_plus = [b_plus_means - b_plus_mins, b_plus_maxs - b_plus_means]

mean_times = [q1_results['1d']['mean_times'][m] for m in m_list_q1]
median_times = [q1_results['1d']['median_times'][m] for m in m_list_q1]

# -------------------------------------------------------------
# Figure 1: 2-Panel Plot (Panel A: B_hat detailed scale, Panel B: B_plus scale)
# Solves visibility of B_hat and avoids legend occlusion
# -------------------------------------------------------------
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Panel (a): B_hat detailed
bar_w = 0.50
ax1a.axhline(1.0, color='#333333', linestyle='--', linewidth=1.5, label='Baseline Full $D$ (Ratio = 1.0)')
bars_a = ax1a.bar(x_indices, b_hat_means, bar_w, yerr=yerr_b_hat,
                  capsize=4, ecolor='black', color='#1f77b4', edgecolor='#0e4d7a', linewidth=1.2,
                  label=r'$\hat{B}$ (Row-averaged centroids)')

for idx, (m_val, bm) in enumerate(zip(m_list_q1, b_hat_means)):
    ax1a.annotate(f"{bm:.4f}", (idx, bm), textcoords="offset points", xytext=(0, 10),
                  ha='center', fontsize=9, fontweight='bold', color='#0e4d7a')

ax1a.set_title(r'(a) $\hat{B}$ Cost Ratio: $\mathrm{cost}(D, \hat{B}) / \mathrm{cost}(D, B)$', fontsize=11, fontweight='bold', pad=10)
ax1a.set_xlabel('Projected Target Dimension $m$', fontsize=10, fontweight='bold')
ax1a.set_ylabel(r'Cost Ratio to Baseline ($1.0$ = Exact)', fontsize=10, fontweight='bold')
ax1a.set_xticks(x_indices)
ax1a.set_xticklabels([f'$m={m}$' for m in m_list_q1], fontsize=9.5)
ax1a.set_ylim(0.92, 1.30)
ax1a.legend(loc='upper right', frameon=True, fontsize=8.5)
ax1a.grid(True, linestyle=':', alpha=0.6)

# Panel (b): B_plus (Pseudo-inverse)
bars_b = ax1b.bar(x_indices, b_plus_means, bar_w, yerr=yerr_b_plus,
                  capsize=4, ecolor='black', color='#d62728', edgecolor='#8c1112', linewidth=1.2,
                  label=r'$B^+$ ($A^+ a$ pseudo-inverse)')

for idx, (m_val, bpm) in enumerate(zip(m_list_q1, b_plus_means)):
    ax1b.annotate(f"{bpm:.1f}×", (idx, bpm), textcoords="offset points", xytext=(0, 10),
                  ha='center', fontsize=9, fontweight='bold', color='#8c1112')

ax1b.set_title(r'(b) $B^+$ Cost Ratio: $\mathrm{cost}(D, B^+) / \mathrm{cost}(D, B)$', fontsize=11, fontweight='bold', pad=10)
ax1b.set_xlabel('Projected Target Dimension $m$', fontsize=10, fontweight='bold')
ax1b.set_ylabel(r'Cost Ratio to Baseline ($60\times–80\times$ Error)', fontsize=10, fontweight='bold')
ax1b.set_xticks(x_indices)
ax1b.set_xticklabels([f'$m={m}$' for m in m_list_q1], fontsize=9.5)
ax1b.set_ylim(0, 92)
ax1b.legend(loc='upper right', frameon=True, fontsize=8.5)
ax1b.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig_q1_bc_kmeans_ratios.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"Saved Figure 1 (Coloured 2-Panel) to {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Q1(d) Wall-clock Execution Time vs Dimension m
# Shows mean line, median line, and individual run scatter
# -------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9.5, 4.5), dpi=300)

ax2.axhline(t_km_D, color='#d62728', linestyle='--', linewidth=2.0,
            label=f'Full $k$-Means on $D$ ($d=116$, {t_km_D:.1f} s)')

# Individual run scatter to show the m=5 outlier transparently
for m_val in m_list_q1:
    runs = q1_d_times[m_val]
    ax2.scatter([m_val] * len(runs), runs, color='#1f77b4', alpha=0.55, s=40, zorder=3)

ax2.plot(m_list_q1, mean_times, color='#2ca02c', linewidth=2.2, marker='o', markersize=8,
         label=r'Mean Sketched Time ($DA^T$ + $k$-Means on $E$)', zorder=4)
ax2.plot(m_list_q1, median_times, color='#ff7f0e', linewidth=2.0, linestyle='-.', marker='s', markersize=7,
         label=r'Median Sketched Time (Typical Run)', zorder=4)

# Annotate mean and median values
for m_val, tm, tmed in zip(m_list_q1, mean_times, median_times):
    if m_val == 5:
        ax2.annotate(f"mean: {tm:.2f}s\n(med: {tmed:.2f}s)", (m_val, tm), textcoords="offset points", xytext=(15, 10),
                     ha='left', fontsize=8.5, fontweight='bold', color='#2ca02c',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2ca02c', alpha=0.85))
    else:
        ax2.annotate(f"{tm:.2f}s", (m_val, tm), textcoords="offset points", xytext=(0, 9),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2ca02c')

mean_p_all = np.mean([q1_results['1d']['mean_proj_times'][m] for m in m_list_q1]) * 1000
ax2.annotate(f'Measured $DA^T$ projection time: {mean_p_all:.1f} ms\nLloyd $k$-means iterations dominate runtime',
             xy=(0.03, 0.45), xycoords='axes fraction', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef7ee', edgecolor='#2ca02c', alpha=0.9))

ax2.set_title('Question 1(d): Wall-Clock Execution Time vs. Target Dimension $m$', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Projected Target Dimension $m$', fontsize=11, fontweight='bold')
ax2.set_ylabel('Wall-Clock Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_xticks(m_list_q1)
ax2.set_ylim(0, max(t_km_D, max(mean_times)) * 1.25)
ax2.legend(loc='upper right', frameon=True, fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

fig2_path = os.path.join(FIGURES_DIR, "fig_q1_d_execution_times.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"Saved Figure 2 (Coloured) to {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Q1(e) Concentration of Measure (Pairwise vs. Clustering)
# Boxplots showing light outlier points to be fully transparent
# -------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 4.6), dpi=300)
indices = np.arange(len(m_list_q1))
box_width = 0.28
offset = 0.18

ax3.axhline(1.0, color='#333333', linestyle='--', linewidth=1.5, label='Expectation = 1.0')

for i, m in enumerate(m_list_q1):
    bp1 = ax3.boxplot(q1_e_pairwise_ratios[m], positions=[indices[i] - offset], widths=box_width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='.', markersize=2, alpha=0.15, markeredgecolor='#1f77b4', markerfacecolor='#1f77b4'),
                      medianprops=dict(color='black', linewidth=1.8),
                      boxprops=dict(facecolor='#a6cee3', edgecolor='#1f77b4', linewidth=1.2),
                      whiskerprops=dict(color='#1f77b4', linewidth=1.2),
                      capprops=dict(color='#1f77b4', linewidth=1.2))
    
    bp2 = ax3.boxplot(q1_e_clustering_ratios[m], positions=[indices[i] + offset], widths=box_width,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=3, alpha=0.6, markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e'),
                      medianprops=dict(color='black', linewidth=1.8),
                      boxprops=dict(facecolor='#fdbf6f', edgecolor='#ff7f0e', linewidth=1.2),
                      whiskerprops=dict(color='#ff7f0e', linewidth=1.2),
                      capprops=dict(color='#ff7f0e', linewidth=1.2))

patch_pairwise = mpatches.Patch(facecolor='#a6cee3', edgecolor='#1f77b4',
                                label=r'(i) $10^4$ Pairwise Distances $\|E_i - E_l\|^2 / \|D_i - D_l\|^2$ (1 draw of $A$)')
patch_clustering = mpatches.Patch(facecolor='#fdbf6f', edgecolor='#ff7f0e',
                                  label=r'(ii) 20 Clustering Costs $\mathrm{cost}_E(\mathcal{C}_D) / \mathrm{cost}_D(\mathcal{C}_D)$ (20 draws of $A$)')
line_exp = plt.Line2D([0], [0], color='#333333', linestyle='--', linewidth=1.5, label='Expectation = 1.0')

ax3.set_title('Question 1(e): Concentration of Measure — Pairwise Distance vs. Aggregate Clustering Cost',
              fontsize=12, fontweight='bold', pad=12)
ax3.set_xlabel('Projected Target Dimension $m$', fontsize=11, fontweight='bold')
ax3.set_ylabel('Ratio to Ambient Space Metric', fontsize=11, fontweight='bold')
ax3.set_xticks(indices)
ax3.set_xticklabels([f'$m = {m}$' for m in m_list_q1])
ax3.set_ylim(bottom=0.0, top=3.5)
ax3.legend(handles=[patch_pairwise, patch_clustering, line_exp], loc='upper right', frameon=True, fontsize=8.8)
ax3.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

fig3_path = os.path.join(FIGURES_DIR, "fig_q1_e_boxplots.png")
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)
print(f"Saved Figure 3 (Coloured) to {fig3_path}")

# ==========================================
# 3. SAVE METRICS JSON
# ==========================================
print("\n" + "=" * 60)
print("3. SAVING METRICS JSON")
print("=" * 60)

def convert_to_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    return obj

serializable_q1_results = convert_to_serializable(q1_results)

all_json_path = os.path.join(OUTPUT_DIR, "all_results.json")
with open(all_json_path, "w") as f:
    json.dump(serializable_q1_results, f, indent=2)
print(f"Saved fresh metrics to {all_json_path}")

print("\nAll figures and results generated and saved successfully for Q1 into results/")
