"""Embed the strategy population into a low-dimensional manifold and ask
whether 'robust' strategies cluster.

Three feature spaces are supported (compose with --feature):

    metrics    — flatten per-window [Sharpe, PF, MaxDD] vectors per strategy.
                 64 OOS dimensions for a 6-window asset (5 test_tags x 3 metrics
                 plus pads). Captures temporal performance fingerprint.
    params     — one-hot of (primary, transform, sl bucket) plus a confluence
                 string hash. Captures the design space, independent of returns.
    pnl        — time-aligned daily PnL vector per strategy (from the
                 trades/ Parquet); requires the trades.bin ETL to have run.

Embeddings: PCA (deterministic baseline), UMAP (default), and t-SNE for
sanity. A k-NN connectivity test on the embedding labels each strategy as
'robust' (passes the funnel) or not, and reports whether robust strategies
form one connected component, isolated islands, or are uniformly mixed in.

Usage:
    python scripts/manifold.py                      # synthetic demo
    python scripts/manifold.py --from-data --feature metrics --asset ALGO_30m_6W_1MetaW
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_FIG = Path(__file__).resolve().parent.parent / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_FIG.parent / "manifold.json"

TEST_TAGS = ["raw", "ENT", "FEE", "SLI", "ENT+IND"]


# -------------------------------------------------------------------
# Feature builders
# -------------------------------------------------------------------

def feature_metrics(parquet_root: str, asset: str | None = None,
                    sample: str = "OOS", min_trades: int = 20,
                    sharpe_clip: float = 5.0,
                    last_k_windows: int = 6,
                    assets: list[str] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Build per-strategy feature vectors from per-window
    [sharpe, pf, max_dd] columns. To keep cross-asset comparison meaningful,
    we use a *fixed* last-K windows per strategy (default 6). Strategies
    with fewer than K windows are dropped.
    """
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    flt = ds.field("sample") == sample
    if assets:
        flt = flt & ds.field("asset").isin(assets)
    elif asset:  # treat None and "" as "all assets"
        flt = flt & (ds.field("asset") == asset)
    df = (d.to_table(filter=flt,
                     columns=["asset", "family", "strategy_name", "primary",
                              "window_idx", "test_tag", "n_trades",
                              "sharpe", "pf", "max_dd"])
          .to_pandas())
    df = df[df["n_trades"].fillna(0) >= min_trades].copy()
    df["sharpe"] = df["sharpe"].clip(lower=-sharpe_clip, upper=sharpe_clip)
    df = df[df["test_tag"].isin(TEST_TAGS)]
    # Per (asset, strategy_name): keep only the last K window indices.
    grp = df.groupby(["asset", "strategy_name"])["window_idx"]
    df["max_w"] = grp.transform("max")
    df["w_from_end"] = df["max_w"] - df["window_idx"]
    df = df[df["w_from_end"] < last_k_windows].copy()
    # Re-key by relative window position ("end-0", "end-1", ...) so
    # different-WFO-depth assets share columns.
    df["wkey"] = "wEnd" + df["w_from_end"].astype(str)
    df["key"] = df["wkey"] + "_" + df["test_tag"].astype(str)
    pivot = df.pivot_table(index=["asset", "family", "strategy_name", "primary"],
                           columns="key",
                           values=["sharpe", "pf", "max_dd"],
                           aggfunc="first")
    pivot.columns = [f"{m}_{k}" for m, k in pivot.columns]
    pivot = pivot.reset_index()
    feat_cols = [c for c in pivot.columns
                 if c not in ("asset", "family", "strategy_name", "primary")]
    # Drop strategies with too many missing values
    keep = pivot[feat_cols].isna().sum(axis=1) < 0.4 * len(feat_cols)
    pivot = pivot[keep].reset_index(drop=True)
    pivot[feat_cols] = pivot[feat_cols].fillna(pivot[feat_cols].median())
    X = pivot[feat_cols].to_numpy(dtype=np.float64)
    # Standardise each column
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    X = (X - mu) / sd
    return X, pivot[["asset", "family", "strategy_name", "primary"]]


def feature_params(parquet_root: str, asset: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Pure parameter-loadings feature space: one-hot of (primary, transform,
    sl-bucket) + length of confluence string. Independent of returns."""
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    flt = (ds.field("sample") == "OOS") & (ds.field("test_tag") == "raw") \
          & (ds.field("window_idx") == 1)
    if asset:
        flt = flt & (ds.field("asset") == asset)
    df = (d.to_table(filter=flt,
                     columns=["asset", "family", "strategy_name", "primary",
                              "transform", "confluence", "sl"])
          .to_pandas()
          .drop_duplicates(["asset", "strategy_name"]))
    one_hot_primary = pd.get_dummies(df["primary"].astype(str), prefix="prim")
    one_hot_transform = pd.get_dummies(df["transform"].astype(str), prefix="tr")
    sl_buckets = pd.cut(df["sl"].fillna(-1), bins=[-2, 0, 1, 1.5, 2, 5],
                        labels=["sl0", "sl1", "sl1.5", "sl2", "sl2plus"])
    one_hot_sl = pd.get_dummies(sl_buckets, prefix="sl")
    confl_len = df["confluence"].fillna("").str.len().to_frame("confl_len")
    confl_n_tokens = (df["confluence"].fillna("").str.count("_") + 1).to_frame("confl_tokens")
    X = pd.concat([one_hot_primary, one_hot_transform, one_hot_sl,
                   confl_len, confl_n_tokens], axis=1).to_numpy(dtype=np.float64)
    return X, df[["asset", "family", "strategy_name", "primary"]]


# -------------------------------------------------------------------
# Robustness label
# -------------------------------------------------------------------

def robust_labels(parquet_root: str, meta: pd.DataFrame,
                  min_trades: int = 20) -> pd.Series:
    """For each strategy in meta, label whether it 'passes the funnel':
    raw OOS Sharpe>0 in last 2 windows AND ENT+OOS Sharpe>0 in last 2 windows.
    """
    import pyarrow.dataset as ds
    d = ds.dataset(parquet_root, format="parquet", partitioning="hive")
    asset_set = sorted(meta["asset"].unique())
    flt = (ds.field("sample") == "OOS") & (ds.field("asset").isin(asset_set))
    df = (d.to_table(filter=flt,
                     columns=["asset", "strategy_name", "window_idx",
                              "test_tag", "n_trades", "sharpe"])
          .to_pandas())
    df = df[df["n_trades"].fillna(0) >= min_trades]
    df = df[df["test_tag"].isin(["raw", "ENT"])]
    # last 2 windows per strategy
    grp = df.groupby(["asset", "strategy_name"])["window_idx"]
    df = df.assign(max_w=grp.transform("max"))
    df = df[(df["max_w"] - df["window_idx"]) < 2]
    pos = df.assign(_pos=df["sharpe"] > 0)
    rate = (pos.groupby(["asset", "strategy_name", "test_tag"])["_pos"]
            .all().unstack("test_tag")
            .fillna(False))
    if "ENT" in rate.columns:
        passes = rate["raw"] & rate["ENT"]
    else:
        passes = rate["raw"]
    passes.name = "passes_funnel"
    out = (meta.merge(passes.reset_index(), on=["asset", "strategy_name"], how="left")
           ["passes_funnel"].fillna(False))
    return out


# -------------------------------------------------------------------
# Embedding + connectivity
# -------------------------------------------------------------------

def fit_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1,
             n_components: int = 2, seed: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=seed,
                        metric="euclidean")
    return reducer.fit_transform(X)


def fit_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=42).fit_transform(X)


def connectivity_score(emb: np.ndarray, labels: np.ndarray, k: int = 15) -> dict:
    """Build a k-NN graph on the embedding, label each edge as
    same-class/cross-class for 'robust' vs 'fragile', and report:
        modularity_proxy: fraction of edges that stay within a class
        n_components_robust: connected components among robust nodes
    """
    from sklearn.neighbors import kneighbors_graph
    G = kneighbors_graph(emb, n_neighbors=k, mode="connectivity",
                         include_self=False)
    rows, cols = G.nonzero()
    same = labels[rows] == labels[cols]
    modularity_proxy = float(same.mean())

    # Connected components among robust nodes only
    robust_idx = np.where(labels)[0]
    if len(robust_idx) > 1:
        sub = G[robust_idx][:, robust_idx]
        from scipy.sparse.csgraph import connected_components
        n_cc, _ = connected_components(sub, directed=False)
    else:
        n_cc = 0
    return {
        "modularity_proxy": modularity_proxy,
        "n_components_robust": int(n_cc),
        "n_robust": int(labels.sum()),
        "n_fragile": int((~labels).sum()),
    }


def plot_embedding(emb: np.ndarray, meta: pd.DataFrame, labels: np.ndarray,
                   title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(emb[~labels, 0], emb[~labels, 1], s=3, alpha=0.25,
               c="#999999", label=f"fragile (n={(~labels).sum():,})")
    ax.scatter(emb[labels, 0], emb[labels, 1], s=4, alpha=0.6,
               c="#cc4c4c", label=f"robust (n={labels.sum():,})")
    ax.set_title(f"{title} — robust vs fragile")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    fams = meta["family"].astype(str)
    palette = plt.colormaps["tab10"]
    fam_counts = fams.value_counts()
    # cap at top-10 categories to keep the legend bounded
    top = list(fam_counts.head(10).index)
    fam_codes_full = fams.where(fams.isin(top), other="other").to_numpy()
    uniq = top + (["other"] if (fam_counts.index.difference(top).size) else [])
    for i, f in enumerate(uniq):
        m = fam_codes_full == f
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5,
                   c=[palette(i % 10)], label=f"{f} (n={m.sum():,})")
    ax.set_title(f"{title} — indicator family")
    ax.legend(loc="upper right", fontsize=7, markerscale=2,
              framealpha=0.85, ncol=1)
    ax.set_xticks([]); ax.set_yticks([])

    fig.set_size_inches(13, 5.5, forward=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Synthetic demo
# -------------------------------------------------------------------

def synthetic_demo(n: int = 3000, seed: int = 42) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """3 Gaussian blobs in 12-D: robust cluster (1k), fragile cluster (1k),
    noise cluster (1k). The robust-cluster mean is well-separated.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([
        np.zeros(12),
        np.concatenate([np.full(6, 2.0), np.zeros(6)]),
        np.concatenate([np.zeros(6), np.full(6, -2.0)]),
    ])
    sizes = [1000, 1000, 1000]
    classes = []
    Xs = []
    for c_idx, (mu, n_c) in enumerate(zip(centers, sizes)):
        Xs.append(rng.normal(loc=mu, scale=1.0, size=(n_c, 12)))
        classes += [c_idx] * n_c
    X = np.vstack(Xs)
    classes = np.array(classes)
    # 'robust' = class 1
    is_robust = classes == 1
    fams = ["EMA", "RSI", "ATR"]
    meta = pd.DataFrame({
        "asset": "SYNTH",
        "family": [fams[c] for c in classes],
        "strategy_name": [f"s{i:05d}" for i in range(n)],
        "primary": [fams[c] for c in classes],
    })
    return X, meta, is_robust


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_one(X: np.ndarray, meta: pd.DataFrame, labels: np.ndarray,
            label_tag: str, do_umap: bool = True) -> dict:
    out: dict = {"n": int(X.shape[0]), "d": int(X.shape[1])}
    print(f"[{label_tag}] X.shape = {X.shape}")
    t0 = time.time()
    pca = fit_pca(X)
    out["pca_var_pc1"] = float(np.var(pca[:, 0]))
    out["pca_var_pc2"] = float(np.var(pca[:, 1]))
    print(f"[{label_tag}] PCA done in {time.time()-t0:.1f}s")
    plot_embedding(pca, meta, labels, f"{label_tag} — PCA",
                   OUT_FIG / f"fig_pca_{label_tag}.png")
    if do_umap:
        t1 = time.time()
        umap_emb = fit_umap(X)
        print(f"[{label_tag}] UMAP done in {time.time()-t1:.1f}s")
        plot_embedding(umap_emb, meta, labels, f"{label_tag} — UMAP",
                       OUT_FIG / f"fig_umap_{label_tag}.png")
        out["connectivity_umap"] = connectivity_score(umap_emb, labels)
    out["connectivity_pca"] = connectivity_score(pca, labels)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-root", default="/mnt/d/strategies_parquet/strategies")
    ap.add_argument("--asset", default="",
                    help="single asset to filter to (default: all assets)")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="restrict to multiple assets (overrides --asset)")
    ap.add_argument("--feature", choices=["metrics", "params"], default="metrics")
    ap.add_argument("--no-umap", action="store_true",
                    help="skip UMAP (PCA only — useful for n>200k)")
    ap.add_argument("--subsample", type=int, default=None,
                    help="random subsample n strategies before fitting")
    ap.add_argument("--synthetic", action="store_true",
                    help="(rare) three-blob synthetic demo; only for "
                         "testing the analysis machinery.")
    args = ap.parse_args()

    summary: dict = {"mode": "synthetic" if args.synthetic else "data",
                     "feature": args.feature}
    if not args.synthetic:
        if args.feature == "metrics":
            X, meta = feature_metrics(args.parquet_root, asset=args.asset,
                                      assets=args.assets)
        else:
            X, meta = feature_params(args.parquet_root, asset=args.asset)
        labels = robust_labels(args.parquet_root, meta).to_numpy(dtype=bool)
        if args.subsample and X.shape[0] > args.subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(X.shape[0], size=args.subsample, replace=False)
            X = X[idx]
            meta = meta.iloc[idx].reset_index(drop=True)
            labels = labels[idx]
        summary["asset"] = args.asset or "ALL"
        summary["n_loaded"] = int(X.shape[0])
        summary["robust_rate"] = float(labels.mean())
        tag = f"{args.feature}_{args.asset or 'ALL'}"
        out = run_one(X, meta, labels, label_tag=tag,
                      do_umap=not args.no_umap)
    else:
        X, meta, labels = synthetic_demo()
        summary["n_loaded"] = int(X.shape[0])
        summary["robust_rate"] = float(labels.mean())
        out = run_one(X, meta, labels, label_tag="synthetic")
    summary.update(out)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nresults summary -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
