# strategy-manifold

**Geometry of strategy space.**

> Companion repository to the M-series of reference implementations on
> [daniel-v-gatto.com](https://daniel-v-gatto.com). Inputs come from the
> [`quant-research-framework-rs`](https://github.com/DaruFinance/quant-research-framework-rs)
> walk-forward backtester, processed through
> [`strategy-generalization-analysis`](https://github.com/DaruFinance/strategy-generalization-analysis).

## What this is

A combinatorial strategy generator (8 indicator families × 12 transforms
× 25 confluence filters × 2 stop-loss variants ≈ 19,806 specs per asset)
produces a population of walk-forward backtests. Each strategy is a
point in a high-dimensional feature space — its per-window Sharpe / PF /
MaxDD vector, or its (primary, transform, confluence, sl) parameter
loadings.

Embed the population into 2 dimensions via PCA and UMAP, then ask:

1. Do **robust** strategies (those passing the OOS+ENT funnel — i.e.
   profitable in the last two windows under both raw signal and entry
   drift) cluster in the embedding, or are they uniformly mixed in?
2. If they cluster — is the robust region one connected component, or
   fragmented across many small islands?

The first answers whether *edge has structure* in feature space. The
second answers whether edge is a single mode that gradient ascent would
find, or a constellation of isolated optima reachable only by luck.

## Reproduce

```bash
git clone https://github.com/DaruFinance/strategy-manifold
cd strategy-manifold
pip install -e .
python scripts/manifold.py                    # default: full real corpus
```

The default reads the strategies/ Parquet substrate at
`/mnt/d/strategies_parquet/strategies` and embeds *all* assets (last-K=6
windows per strategy, so 6W and 27W assets share a 90-D feature space).
For per-asset runs use `--asset ASSET_DIR`. Pass `--subsample N` to fit on
a stratified subsample (recommended above ~100k strategies for UMAP).

A `--synthetic` flag runs the three-blob deterministic demo (only used by
the smoke test).

## Problem statement

Given `N` strategies and a feature map `φ : strategy → ℝᵈ`, build a 2D
embedding `Φ` and a robustness label `r(s) ∈ {0, 1}`. Two summary
statistics for the embedding:

- **modularity proxy** = fraction of k-NN edges (k=15) where both endpoints
  share the same `r` label. Above the chance baseline `r̄² + (1−r̄)²` ⇒
  robust strategies cluster.
- **n_components_robust** = number of connected components in the k-NN
  subgraph induced by the robust subset. n=1 ⇒ single connected region;
  n>>1 ⇒ fragmented.

## Feature spaces

Three are implemented, each captures a different facet of "what makes
strategies similar":

| Flag | Description | Dimensionality |
|------|-------------|----------------|
| `--feature metrics` | Per-window `[Sharpe, PF, MaxDD]` flattened across the 5 robustness perturbations and the W windows. | `15 · W` (e.g. 90 for 6W) |
| `--feature params` | One-hot of `(primary, transform, sl-bucket)` plus confluence string length. | ~30–80 |
| `--feature pnl` *(planned)* | Time-aligned daily PnL vectors from `trades/` Parquet, joined to OHLCV timestamps. Native return-time-series view. | depends on horizon |

## Headline result

On the **10 deepest-WFO assets** (ETH, BTC, LTC, TRX, XRP, LINK, ZEC,
DOGE, BCH, AVAX), 100,000-strategy stratified subsample, last-K=6
windows feature, n_trades ≥ 20 gate, 6.87% robust):

| | UMAP | PCA |
|---|---|---|
| modularity proxy | 0.903 | 0.911 |
| n_components_robust | **1,229** | 938 |
| n_robust | 6,869 | 6,869 |
| n_fragile | 93,131 | 93,131 |

Random-baseline modularity is `r̄² + (1−r̄)² ≈ 0.872`. Robust strategies
do cluster (UMAP lift ≈ 0.03 over chance) but the robust region is
**fragmented into ~1,200 islands** — average of ~5.6 robust strategies
per component. Edge is a constellation of isolated optima, not a single
connected mode that gradient ascent could find.

The pattern is robust to scale and asset selection:

| corpus | n_strats | n_robust | components | strats/island |
|---|---:|---:|---:|---:|
| ALGO single asset | 15,000 | 1,765 | 213 | 8.3 |
| 30-asset (random 50k) | 50,000 | 3,924 | 641 | 6.1 |
| 10-deepest (100k) | 100,000 | 6,869 | 1,229 | 5.6 |

Scaling up keeps the per-island count in the 5–8 range — robust strategies
are not a single connected manifold at any scale we've checked.

## Usage

```bash
# Cross-asset (default):
python scripts/manifold.py --subsample 50000

# Single asset:
python scripts/manifold.py --asset ALGO_30m_6W_1MetaW

# Parameter-loadings feature space instead of metrics:
python scripts/manifold.py --feature params
```

`--subsample` is recommended for `n > 50k` because UMAP at scale costs
quadratic memory; the script fits on the subsample and the next release
will project the rest via `transform()`.

## References

- McInnes, L., Healy, J. & Melville, J. (2018). *UMAP: Uniform Manifold
  Approximation and Projection.*
- See also the companion repos
  [`strategy-tda`](https://github.com/DaruFinance/strategy-tda) (persistent
  homology of the same population) and
  [`strategy-rmt`](https://github.com/DaruFinance/strategy-rmt) (eigenspectrum
  of the strategy correlation matrix).

## License

MIT © Daniel Vieira Gatto.
