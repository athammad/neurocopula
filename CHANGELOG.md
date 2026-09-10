# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-10

First release.

### Added

**Core estimator**
- `NeuroCopula`: a Neural Spline Flow (`zuko`) fitted by maximum likelihood to a DataFrame's
  joint distribution.
- `copula_mode=True` (default) Gaussianizes each column through its empirical CDF, so the flow
  learns dependence only and marginals are reproduced exactly.
- Two-stage `fit`: a train/validation split to diagnose overfitting and choose the epoch count,
  then a refit on 100% of the rows, so the split costs no data.
- Chronological splits with an optional `embargo`, for time-series data.
- `log_prob`, `copula_log_prob`, `score`, `aic` for density evaluation and model selection.
- `sample`, `sample_uniform`, `pseudo_observations` for generation on either scale.
- `tail_dependence`, `tail_dependence_curve`, `joint_exceedance_probability`,
  `conditional_distribution`, `conditional_probability`, `conditional_quantiles`.
- `bayesian=True` plus `uncertainty_report` for credible intervals on any query.
- `sampling_mode="coupling"` for parallel sampling; `device` selection; minibatch training via
  `batch_size`; `save` / `load`.

**Supporting modules**
- `marginals`: `EmpiricalCopulaTransform` and `StandardizeTransform`.
- `datasets`: Clayton, Gumbel, Gaussian, Student-t, two-regime mixtures and a 5-column asset
  panel, each with closed-form tail dependence and Kendall's tau.
- `metrics`: tail dependence and curves, Kendall/Spearman matrices, tail asymmetry, energy
  distance, MMD, per-column KS, and errors against known truth.
- `vine`: `VineCopula`, a `pyvinecopulib` baseline exposing the same API.
- `benchmark`: a seven-case suite, multi-seed runner, `paired_comparison`, and summary tables.
- `plotting` and `theme`: fourteen figures sharing one accessible visual system.

**Documentation**
- Six example notebooks and a reproducible benchmark notebook.
- A README reporting measured benchmark results, including the cases where the flow loses.

### Notes
- `pyvinecopulib` is optional (`pip install "neurocopula[vine]"`); the library imports and runs
  without it.
- Known limitations are listed in the README, chiefly: the empirical CDF cannot extrapolate
  beyond observed extremes, `log_prob` in copula mode depends on a KDE marginal estimate, and
  small datasets relative to dimension favour parametric models.

[0.1.0]: https://github.com/athammad/neurocopula/releases/tag/v0.1.0
