# neurocopula

**Learn dependence structure with normalizing flows, instead of assuming it.**

[![Tests](https://github.com/athammad/neurocopula/actions/workflows/ci.yml/badge.svg)](https://github.com/athammad/neurocopula/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neurocopula.svg)](https://pypi.org/project/neurocopula/)
[![Python](https://img.shields.io/pypi/pyversions/neurocopula.svg)](https://pypi.org/project/neurocopula/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Any joint distribution splits into its marginals and its copula:

$$p(x_1, \dots, x_d) = c\big(F_1(x_1), \dots, F_d(x_d)\big) \cdot \prod_j f_j(x_j)$$

Classical copula models choose a parametric family for $c$ — Gaussian, Student-t, Clayton —
and inherit whatever tail behaviour that family imposes. `neurocopula` learns $c$ with a
Neural Spline Flow ([zuko](https://github.com/probabilists/zuko)), so its shape is inferred
from the data rather than assumed, while the marginals $F_j$ are still reproduced **exactly**
by their empirical distribution functions.

---

## Why this matters: the tail dependence problem

A Gaussian copula forces the tail dependence coefficient to **zero**, no matter how high the
correlation. Fit one to data whose extremes genuinely move together and it will tell you the
joint crash you are worried about is far rarer than it is.

```python
from neurocopula import NeuroCopula, datasets

# Clayton copula: crashes together, rallies independently.
df = datasets.make_clayton(n=4000, theta=2.0, seed=0)
model = NeuroCopula(copula_mode=True).fit(df)

model.tail_dependence("x1", "x2", q=0.02, tail="lower")   # 0.731  (truth: 0.707)
model.tail_dependence("x1", "x2", q=0.02, tail="upper")   # 0.059  (truth: 0.0)
```

The model recovers the **asymmetry** — strong dependence going down, essentially none going
up. No elliptical copula can represent that, because Gaussian and Student-t force the two
tails to be equal by construction.

The same fact, restated as something you can act on:

```python
crash, _ = model.conditional_probability("x2", "<", -2.0, given={"x1": -3.0})  # 0.636
calm,  _ = model.conditional_probability("x2", "<", -2.0, given={"x1":  0.0})  # 0.002
```

A 260x difference in the probability that the second asset breaks, depending entirely on
what the first one just did.

## Install

```bash
pip install neurocopula                 # core
pip install "neurocopula[vine]"         # + pyvinecopulib, for the comparison baseline
pip install "neurocopula[gridded]"      # + xarray/netCDF4, for the spatial layer
pip install "neurocopula[regions]"      # + geopandas, for polygon selection
pip install "neurocopula[all]"          # everything, plus notebooks and dev tools
```

Python 3.10+. The core depends on `torch`, `zuko`, `numpy`, `pandas`, `scipy`, `matplotlib`.

## Quick start

```python
import pandas as pd
from neurocopula import NeuroCopula

df = pd.read_csv("returns.csv")          # continuous columns, oldest row first
model = NeuroCopula(copula_mode=True).fit(df)

# 1. How likely is it that these all break at once?
model.joint_exceedance_probability({"SPX": ("<", -0.03), "NDX": ("<", -0.03)})

# 2. Do they actually crash together, or does that just look true on average?
model.tail_dependence("SPX", "NDX", q=0.02, tail="lower")

# 3. Given one has moved, what happens to the other?
model.conditional_probability("NDX", "<", -0.02, given={"SPX": -0.05})
model.conditional_quantiles("NDX", given={"SPX": -0.05})       # conditional VaR

# 4. Synthetic scenarios, with the marginals reproduced exactly.
model.sample(10_000)

# 5. Held-out likelihood, for model selection.
model.score(test_df, copula=True)
```

## Core ideas

**Copula mode (the default).** Each column is mapped through its empirical CDF and the probit
before the flow sees it, so every column is marginally $N(0,1)$ and the flow's whole capacity
goes into dependence. On the way out, the empirical quantile function reproduces the observed
marginals exactly — fat tails, skew and all — without the flow having to learn any of it. Set
`copula_mode=False` for a general joint density estimator instead.

**Honest fitting.** `fit` trains on a train/validation split to find where validation NLL stops
improving, then refits on **100%** of the rows at that epoch count. The split diagnoses
overfitting without permanently costing you data. The default split is *chronological*, which
for time series tests whether the fit generalises forward rather than letting a shuffle hide a
regime shift.

**Uncertainty.** `bayesian=True` makes the flow's weights distributions, so any query can be
reported with a credible interval:

```python
model = NeuroCopula(bayesian=True).fit(df)
model.uncertainty_report(lambda m: m.tail_dependence("x1", "x2", q=0.05))
# {'mean': 0.68, 'std': 0.05, 'q05': 0.60, 'q95': 0.75, 'samples': array([...])}
```

Worth doing before quoting any tail-dependence number from a few thousand rows. If the
interval spans half the unit interval, the point estimate was never worth its decimal places.

**Speed.** `sampling_mode="coupling"` samples every dimension in parallel instead of one at a
time. Use it whenever you draw large Monte Carlo samples, which every query method does
internally.

## Plots

`neurocopula.plotting` provides fourteen figures sharing one visual system. The important one
is the tail dependence curve, $\lambda(q)$ against $q$:

```python
from neurocopula import plotting as ncplot

ncplot.plot_model_comparison(df, {"NeuroCopula": model, "Vine": vine},
                             "x1", "x2", theoretical=0.707)
```

Read it by **shape**. A curve sliding toward zero as $q$ shrinks means asymptotic independence
— the pair decouples in the extreme, which is what a Gaussian copula always predicts. A curve
flattening onto a positive plateau means genuine tail dependence. A model that matches at
$q = 0.25$ but not at $q = 0.01$ will misprice exactly the events you built it for.

| Question | Functions |
|---|---|
| Did it train properly? | `plot_training_curves`, `plot_diagnostics` |
| Is the dependence right? | `plot_copula_scatter`, `plot_pairwise_grid`, `plot_dependence_heatmap`, `plot_tau_error_heatmap` |
| Are the tails right? | `plot_tail_dependence_curve`, `plot_tail_concentration`, `plot_exceedance_comparison` |
| Are the marginals intact? | `plot_marginal_comparison` |
| What does a conditional look like? | `plot_conditional_distribution` |
| How uncertain is it? | `plot_uncertainty` |
| How do the models compare? | `plot_benchmark_bars`, `plot_model_comparison` |

## Gridded data: one model, any region

`NeuroCopula` models dependence between variables at a single site. For data on a
grid there is a second dependence to account for — between locations — and ignoring
it quietly breaks anything aggregated over an area.

The arithmetic is worth seeing. Suppose an event has probability 0.01 at each of
1,000 locations. Under independence, the chance it happens *somewhere* is
`1 - 0.99^1000 ≈ 1`. But locations are not independent: when things go wrong in one
place they usually go wrong next door, so events pile onto the same rows rather than
spreading across different ones. Independence also divides the standard deviation of
an areal mean by `√n` — a factor of ~30 here — which makes region-wide extremes
impossible by construction.

`SpatialProbabilisticLayer` fixes that by fitting two layers:

```python
from neurocopula import SpatialProbabilisticLayer

layer = SpatialProbabilisticLayer(
    variables=["a", "b", "c"],
    marginal_specs={"a": dict(zero_inflated=True, zero_threshold=0.1, tail="gpd")},
).fit(data, coords)          # data: (n_rows, n_locations, n_vars);  coords: (n_locations, 2)

cells = layer.select(bbox=(-8, -6, 105, 115))
layer.sample(cells, n=10_000)        # (10000, n_cells, n_vars) joint draws
```

**Layer 1** is a per-location copula: marginals, then a flow over the variables.
**Layer 2** is a Student-t spatial process linking locations, so the joint
distribution over *any* subset is available after fitting — the region is chosen at
query time, not baked in. It is deliberately Student-t and not Gaussian: a Gaussian
random field has exactly **zero tail dependence**, so locations become independent in
the extremes however correlated they look otherwise. If joint extremes are the
subject, that assumption answers the question before the data is consulted.

Layer 2 operates on the flow's *latent* space rather than on observed values. A flow
already maps data to independent latents, so correlating those and inverting the flow
inherits the spatial dependence while preserving each location's own cross-variable
copula. The other order destroys one of them.

### Asking the right question

These are different questions with different answers, and only the first is
independent of spatial dependence:

```python
conditions = {"a": (">", 50), "b": (">", 35)}

layer.exceedance_probability(conditions, cells, mode="pointwise")  # per location
layer.exceedance_probability(conditions, cells, mode="mean")       # at a typical point
layer.exceedance_probability(conditions, cells, mode="any")        # anywhere in the region
layer.exceedance_probability(conditions, cells, mode="fraction")   # how much of it
layer.posterior_exceedance(conditions, cells, mode="mean")         # with a credible interval
```

`sample` is the general case — it returns coherent joint draws, so any aggregation you
can write is a valid query.

### Named regions

```python
from neurocopula import RegionSet

layer.attach_regions(RegionSet.natural_earth())
cells = layer.select(region="Indonesia", max_cells=2000)
```

`RegionSet.from_file` reads any polygon file geopandas supports, so administrative
boundaries and hydrological catchments work the same way. Identifiers need not be
strings — HydroBASINS numbers its polygons, and integer ids are used as such.

`max_cells` matters. Joint sampling factorizes an n×n matrix — O(n³) time, O(n²)
memory:

| locations | memory | 5,000 draws |
|---|---|---|
| 500 | 0.002 GB | 1.4 s |
| 2,000 | 0.03 GB | 7.7 s |
| 5,000 | 0.20 GB | 23 s |
| 15,000 | 1.8 GB | minutes |

For an *areal* statistic a random subset of the region is an unbiased estimate of the
whole, so thinning costs accuracy that is small next to the Monte Carlo error already
present.

### Stratified fitting

A copula assumes its rows are draws from **one** distribution. Pooling rows from
different regimes estimates a mixture of them instead, and the "dependence" you get
back is partly just the difference between the regimes.

```python
from neurocopula import SeasonalLayers

layers = SeasonalLayers.fit(data, coords, dates, variables=[...],
                            by="season", period=(1991, 2020))
layers.exceedance_probability(conditions, cells, when="DJF", mode="any")
layers.seasonal_cycle(conditions, cells)     # the cycle a pooled fit averages away
```

This is the stationarisation step — the job an ARMA-GARCH filter does in the classic
vine-copula workflow — and it keeps everything in physical units, so no conversion
wraps the queries.

### Marginals with atoms and heavy tails

Plenty of real variables are exactly zero much of the time and heavy-tailed the rest —
rainfall, insurance claims, river discharge, trade volumes. A plain empirical CDF maps
every zero to the same pseudo-observation, so they collapse onto one spike and the
flow spends its capacity modelling a delta function.

```python
from neurocopula import SemiParametricMarginal

m = SemiParametricMarginal(zero_inflated=True, zero_threshold=0.1, tail="gpd").fit(x)
```

Zeros get the randomized probability integral transform, which spreads them over the
interval they actually occupy. The transform is therefore stochastic by design — that
is correct for a discrete-continuous mixture, not a bug. A generalised Pareto upper
tail lets the model extrapolate past the largest observed value, which a bare
empirical CDF can never do.

**One caveat worth knowing.** That randomization destroys exactly the co-occurrence
information Layer 2 needs, because two neighbouring locations that are both zero on
the same row get independent random values. On synthetic data with a known 300 km
range, the naive estimate collapsed to 125 km and reported *no* tail dependence at
all. The layer detects an atom and switches to estimating correlation from the
censoring pattern by tetrachoric inversion, and degrees of freedom from the upper tail
— recovering 289 km.

### Export

```python
from neurocopula.io import save_layer
save_layer(layer, "mylayer", precompute={"compound": conditions})
# mylayer.nc  — grid, spatial parameters, probability fields; opens in any GIS
# mylayer.pt  — the fitted model, for questions you did not precompute
```

## Benchmarks

Reproduce with `python benchmarks/run_benchmark.py`, or step through
[`benchmarks/benchmark_suite.ipynb`](benchmarks/benchmark_suite.ipynb).

The setup is deliberately unflattering to the flow. Both models use the **same** empirical
marginal transform, so only the copula differs. Both are scored on **held-out** copula
log-likelihood, with the train/test split taken before either model is built. Tail
coefficients are estimated from Monte Carlo samples for both, rather than reading the vine's
analytic parameter, which would compare two different quantities. Four of the seven cases are
generated from families that are literally in the vine's candidate set.

Numbers are **paired per-seed** differences over 3 seeds — both models saw the same rows at
each seed, which removes dataset-to-dataset variation and isolates the models.

### Held-out copula log-likelihood (higher is better)

| Case | NeuroCopula | Best vine | Difference | Seeds won |
|---|---|---|---|---|
| `gaussian_rho0.7` | 0.3359 | **0.3416** | −0.0058 | 0/3 |
| `clayton_theta2` | 0.4088 | **0.4265** | −0.0177 | 0/3 |
| `gumbel_theta2` | 0.3681 | **0.3719** | −0.0038 | 0/3 |
| `t_copula_df4` | 0.3464 | **0.3557** | −0.0093 | 0/3 |
| `mixed_regime` | **0.1787** | 0.1502 | **+0.0284** | 3/3 |
| `asset_panel_5d` | 0.7479 | **0.8557** | **−0.1078** | 0/3 |
| `mixed_regime_5d` | **0.8469** | 0.7311 | **+0.1157** | 3/3 |

Every case lands the same way on all three seeds — these differences are consistent, not noise.

### What the results actually say

**When a parametric family fits, use it.** On the four cases generated *from* a family in the
vine's set, the vine wins every seed. The margins are small (0.004–0.018 nats) but real. This
is not surprising and it is not a flaw: the vine is fitting 1–2 parameters where the flow
fits ~29,000, so when the parametric assumption is true, the vine is simply more efficient.
**A flow is not a free upgrade.**

**When no family fits, the flow wins clearly.** On the two-regime mixture — deliberately in no
standard family — the flow wins every seed, in 2-D and 5-D. The log-likelihood gap understates
it; the tail behaviour is where the difference bites:

| `mixed_regime` | λ_L (q=0.02) |
|---|---|
| **Data** | **0.65** |
| NeuroCopula | 0.59 |
| Best vine | 0.29 |

The vine underestimates the probability of a joint crash by roughly half. The flow gets close.

**Dimension is not the deciding factor — family match is.** The two 5-D cases isolate this. On
`asset_panel_5d` (elliptical, so the vine has the right family) the vine wins by a wide margin,
0.108. On `mixed_regime_5d` (same dimension, no matching family) the flow wins by 0.116. The
`asset_panel_5d` loss is not undertraining: raising epochs 5x and widening the network moves
it by less than 0.02. With 2,250 training rows and ~48,000 parameters, the flow is simply
sample-starved where the vine's 20 parameters are not.

**Recovering known tail dependence.** Where truth is known in closed form:

| Case | Truth (λ_L / λ_U) | NeuroCopula | Vine (parametric) | Vine (elliptical) |
|---|---|---|---|---|
| `clayton_theta2` | 0.707 / 0.0 | 0.724 / 0.047 | 0.711 / 0.060 | 0.400 / 0.411 |
| `gumbel_theta2` | 0.0 / 0.586 | 0.191 / 0.610 | 0.189 / 0.591 | 0.383 / 0.394 |

The last column is the cautionary one. Restricted to elliptical pair copulas, a vine
*cannot* represent tail asymmetry — it reports λ_L ≈ λ_U on data where the truth is 0.707
versus 0.0. That failure is structural, and no amount of data fixes it. The flow and the
full-parametric vine both get it right.

**Cost.** The flow fits in 7–110 s against the vine's 0.2–2 s, a 30–100x difference on these
problem sizes, plus ~29,000 parameters against 1–20.

### Choosing between them

| Use a vine copula when | Use `neurocopula` when |
|---|---|
| A standard family plausibly fits | The dependence is asymmetric, multi-modal, or regime-switching |
| Data is limited relative to dimension | You have enough rows to support a flexible model |
| You need interpretable pair-by-pair structure | You care most about joint-tail accuracy |
| Fitting must be fast | Fit cost is amortised over many queries |

Both are one import away, so fit both and compare with
[`benchmark.paired_comparison`](src/neurocopula/benchmark.py). That is the point of shipping
the baseline.

## Examples

Runnable notebooks in [`examples/`](examples/):

| Notebook | Covers |
|---|---|
| [01 Quickstart](examples/01_quickstart.ipynb) | Fit, diagnose, and query in one pass |
| [02 Tail dependence](examples/02_tail_dependence.ipynb) | Why Gaussian copulas are dangerous, and how to see it |
| [03 Conditional queries](examples/03_conditional_queries.ipynb) | Stress testing and conditional risk |
| [04 Bayesian uncertainty](examples/04_bayesian_uncertainty.ipynb) | Credible intervals on any query |
| [05 Multivariate panel](examples/05_multivariate_panel.ipynb) | Scaling past two dimensions |
| [06 Vine comparison](examples/06_vine_comparison.ipynb) | Head-to-head against vine copulas |
| [Benchmark suite](benchmarks/benchmark_suite.ipynb) | The full benchmark, reproduced |

## API reference

### `NeuroCopula`

| Method | Purpose |
|---|---|
| `fit(df, ...)` | Fit, with an overfitting diagnostic and a full-data refit |
| `sample(n)` / `sample_uniform(n)` | Draw rows in original units / on the copula scale |
| `log_prob(df)` / `copula_log_prob(df)` | Log-density / log copula density |
| `score(df, copula=)` / `aic(df)` / `bic(df)` | Mean log-likelihood, AIC, BIC |
| `tail_dependence(...)` / `tail_dependence_curve(...)` | λ at one q, or across a range |
| `joint_exceedance_probability(...)` | P(all conditions hold) |
| `conditional_distribution/probability/quantiles(...)` | Conditional queries |
| `uncertainty_report(fn)` | Credible interval on any query (`bayesian=True`) |
| `save(path)` / `load(path)` | Persistence |

### Modules

- **`datasets`** — generators with closed-form tail dependence and Kendall's tau: Clayton,
  Gumbel, Gaussian, Student-t, two-regime mixtures, a 5-column asset panel.
- **`metrics`** — tail dependence and curves, Kendall/Spearman matrices, tail asymmetry,
  energy distance, MMD, KS statistics, errors against known truth.
- **`marginals`** — the empirical-copula and standardization transforms.
- **`vine`** — `VineCopula`, the `pyvinecopulib` baseline wearing the same API.
- **`benchmark`** — the reproducible suite, `paired_comparison`, summaries.
- **`plotting`** / **`theme`** — the fourteen figures and the shared visual system.
- **`spatial`** — `SpatialDependence`: Student-t random fields, correlation estimation,
  joint sampling over any subset of locations.
- **`semiparametric`** — marginals with an atom and/or a generalised Pareto tail.
- **`layer`** / **`seasonal`** / **`regions`** / **`io`** — the gridded stack: the
  probabilistic layer, stratified fitting, polygon-based selection, NetCDF export.

## Known limitations

- **The empirical CDF cannot extrapolate.** Beyond the most extreme observed value of a
  column, `inverse` returns that observed extreme. For events more extreme than anything in
  your sample, splice on an explicit tail model (e.g. a generalised Pareto fit).
- **`log_prob` in copula mode is an estimate.** Turning a copula density into a density in
  original units needs marginal densities, which are estimated by a per-column KDE.
  `copula_log_prob` is exact and is the right number for model comparison.
- **Marginals are fitted on the full frame** passed to `fit`, before the split, so the
  reported validation NLL is not a fully clean out-of-sample number. Hold out a test set
  *before* calling `fit` when you need an untainted estimate — `benchmark` does exactly that.
- **Conditioning is importance-weighted**, not exact. Always check the reported effective
  sample size before trusting a conditional estimate.
- **Small data favours parametric models**, as the benchmark above shows plainly.

## Development

```bash
git clone https://github.com/athammad/neurocopula.git
cd neurocopula
make install-dev
make test          # full suite
make test-fast     # skip the slow model-fitting tests
make lint
make notebooks     # execute every notebook end to end
make benchmark
```

## Citation

```bibtex
@software{hammad_neurocopula,
  author  = {Hammad, Ahmed},
  title   = {neurocopula: Neural copulas with normalizing flows},
  url     = {https://github.com/athammad/neurocopula},
  version = {0.1.0},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

Built on [zuko](https://github.com/probabilists/zuko) (normalizing flows) and
[pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) (vine copulas).
