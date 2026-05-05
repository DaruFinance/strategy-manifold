"""Histogram of robust-cluster sizes — visualises the 'fragmented edge' result."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifold import (feature_metrics, robust_labels, fit_umap,
                      connectivity_score)
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
ASSETS_10 = ["ETH_30m_28W", "BTC_30m_27W", "LTC_30m_27W", "TRX_30m_25W",
             "XRP_30M_25W_new", "LINK_30M_23W_new", "ZEC_30m_22W",
             "DOGE_30m_21W", "BCH_30m_20W", "AVAX_30m_17W"]

print("[fig_cluster_sizes] loading metrics...")
X, meta = feature_metrics("/mnt/d/strategies_parquet/strategies",
                          assets=ASSETS_10)
labels = robust_labels("/mnt/d/strategies_parquet/strategies", meta).to_numpy(bool)
print(f"X.shape={X.shape}, robust_rate={labels.mean()*100:.2f}%")
if X.shape[0] > 100_000:
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], 100_000, replace=False)
    X = X[idx]; meta = meta.iloc[idx].reset_index(drop=True); labels = labels[idx]
print("[fig_cluster_sizes] fitting UMAP...")
emb = fit_umap(X)
print("[fig_cluster_sizes] computing k-NN connectivity...")
G = kneighbors_graph(emb, n_neighbors=15, mode="connectivity",
                     include_self=False)
robust_idx = np.where(labels)[0]
sub = G[robust_idx][:, robust_idx]
n_cc, comp_id = connected_components(sub, directed=False)
sizes = np.bincount(comp_id)
print(f"n_components={n_cc}, sizes range {sizes.min()}-{sizes.max()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(sizes, bins=range(1, sizes.max() + 2), color="#cc4c4c",
             edgecolor="black", linewidth=0.4)
axes[0].set_yscale("log")
axes[0].set_xscale("log")
axes[0].set_xlabel("Component size (n robust strategies)")
axes[0].set_ylabel("Number of components (log)")
axes[0].set_title(f"Robust-cluster size distribution\n"
                  f"({n_cc:,} clusters across {labels.sum():,} robust strategies)")
axes[0].grid(True, alpha=0.3, which="both")

cumulative = np.sort(sizes)[::-1].cumsum() / sizes.sum() * 100
axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative,
             color="#cc4c4c", linewidth=1.5)
axes[1].set_xscale("log")
axes[1].set_xlabel("Component rank (largest first)")
axes[1].set_ylabel("Cumulative share of robust strategies (%)")
axes[1].set_title("Cumulative share covered by top-K clusters")
axes[1].axhline(50, color="#888", linestyle="--", linewidth=0.6, label="50%")
axes[1].axhline(80, color="#888", linestyle=":", linewidth=0.6, label="80%")
axes[1].legend()
axes[1].grid(True, alpha=0.3, which="both")

fig.suptitle("Edge as a constellation: robust strategies do not consolidate",
             fontsize=12)
fig.tight_layout()
out = ROOT / "figures" / "fig_cluster_sizes.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
