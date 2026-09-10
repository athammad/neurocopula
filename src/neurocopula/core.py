"""
The `NeuroCopula` estimator: a normalizing flow fitted to the dependence
structure of a DataFrame, with the probabilistic queries that make it
useful as a copula.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zuko
import zuko.bayesian
from scipy.stats import norm

from .marginals import EmpiricalCopulaTransform, StandardizeTransform

__all__ = ["NeuroCopula"]

_OPS: dict[str, Callable] = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


class NeuroCopula:
    """
    A neural copula: a Neural Spline Flow (`zuko.flows.NSF`) fitted by
    maximum likelihood to the joint distribution of a DataFrame.

    In the default `copula_mode=True`, each column is mapped through its own
    empirical CDF and the probit before the flow ever sees it, so the flow
    learns only the DEPENDENCE structure and the marginals are reproduced
    exactly on the way back out. That is the same factorisation a vine copula
    uses -- the difference is that the dependence structure here is one
    flexible d-dimensional flow rather than a hand-assembled tree of
    parametric bivariate pieces, so it is not restricted to any particular
    family's tail behaviour and does not need a tree structure to be
    selected.

    Typical use::

        from neurocopula import NeuroCopula
        model = NeuroCopula(copula_mode=True).fit(returns_df)
        model.tail_dependence("SPX", "VIX", q=0.05, tail="lower")
        model.joint_exceedance_probability({"SPX": ("<", -0.03), "NDX": ("<", -0.03)})
        model.sample(10_000)

    Parameters
    ----------
    transforms:
        Number of invertible transform layers stacked in the flow. More
        layers means more capacity to represent complex dependence, at the
        cost of parameters and overfitting risk. 3-6 is a sensible range
        for a handful of variables.
    hidden_features:
        Width of each hidden layer of the small network parameterizing every
        transform. ``(128, 128)`` is two hidden layers of 128 units.
    bins:
        Number of spline knots per transform. More bins resolve sharper
        features in the density; 8 (the zuko default) is usually plenty.
    seed:
        Seed for torch/numpy, for reproducible fits.
    sampling_mode:
        ``"autoregressive"`` (the default NSF: most expressive, but samples
        one dimension at a time, which is slow and memory-hungry for the
        large Monte Carlo draws the query methods use) or ``"coupling"``
        (``passes=2``, samples every dimension in parallel -- much cheaper).
        Choose ``"coupling"`` if you will draw large samples often.
    copula_mode:
        If True (default), fit the dependence structure only, with marginals
        handled exactly by their empirical CDF/quantile function. If False,
        the flow models the full joint on plainly standardized data,
        marginals included -- a general density estimator rather than a
        copula.
    marginal_density:
        Only used when ``copula_mode=True``. ``"kde"`` (default) also fits a
        per-column Gaussian KDE so `log_prob` can report a density in
        original units; ``"none"`` skips it (cheaper; `copula_log_prob`
        still works, `log_prob` raises).
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``. None picks CUDA
        when available, otherwise CPU.
    bayesian:
        If True, wrap the flow in `zuko.bayesian.BayesianModel` so its
        WEIGHTS are distributions rather than point estimates. That gives
        epistemic uncertainty -- how much to trust the fit given your sample
        size -- on top of the spread the density itself describes, so any
        query can be reported with a credible interval. See
        `uncertainty_report`. Costs roughly 2x parameters and needs several
        posterior draws per estimate.
    bayes_init_logvar:
        Initial log-variance of each weight posterior. More negative starts
        tighter (closer to a point estimate); training can widen it.
    bayes_kl_weight:
        Multiplier on the KL-to-prior term in the training loss. Larger
        pulls the posteriors toward the prior (tighter, more regularized
        bands); smaller lets them fit the data more freely.
    bayes_n_eval_samples:
        Posterior weight draws averaged when computing NLL during Bayesian
        training. More draws means less noisy curves, at more compute.
    """

    def __init__(
        self,
        transforms: int = 4,
        hidden_features: Sequence[int] = (128, 128),
        bins: int = 8,
        seed: int = 0,
        sampling_mode: str = "autoregressive",
        copula_mode: bool = True,
        marginal_density: str = "kde",
        device: str | None = None,
        bayesian: bool = False,
        bayes_init_logvar: float = -9.0,
        bayes_kl_weight: float = 1e-6,
        bayes_n_eval_samples: int = 3,
    ) -> None:
        if sampling_mode not in ("autoregressive", "coupling"):
            raise ValueError("sampling_mode must be 'autoregressive' or 'coupling'")
        if transforms < 1:
            raise ValueError("transforms must be >= 1")

        self.transforms = transforms
        self.hidden_features = tuple(hidden_features)
        self.bins = bins
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.copula_mode = copula_mode
        self.marginal_density = marginal_density
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.bayesian = bayesian
        self.bayes_init_logvar = bayes_init_logvar
        self.bayes_kl_weight = bayes_kl_weight
        self.bayes_n_eval_samples = bayes_n_eval_samples

        # Set by fit().
        self.flow = None            # concrete usable flow (a posterior draw if bayesian)
        self.bayes = None           # the BayesianModel wrapper, only if bayesian=True
        self.columns: list[str] | None = None
        self.marginals_ = None
        self.history: dict[str, list[float]] = {"train": [], "val": []}
        self.best_epoch: int | None = None
        self.fit_report_: dict | None = None

    # -----------------------------------------------------------------
    # construction
    # -----------------------------------------------------------------
    def _build_base_flow(self, d: int):
        """A fresh, untrained NSF for `d`-dimensional data."""
        kwargs = dict(
            features=d,
            context=0,
            transforms=self.transforms,
            hidden_features=self.hidden_features,
            bins=self.bins,
        )
        if self.sampling_mode == "coupling":
            # passes=2 turns the masked autoregressive transform into a
            # coupling layer, which samples all dimensions in one pass.
            kwargs["passes"] = 2
        return zuko.flows.NSF(**kwargs)

    def _make_model(self, d: int):
        """A fresh, untrained model ready for an optimizer."""
        base = self._build_base_flow(d)
        if self.bayesian:
            base = zuko.bayesian.BayesianModel(base, init_logvar=self.bayes_init_logvar)
        return base.to(self.device)

    def _make_transform(self):
        if self.copula_mode:
            return EmpiricalCopulaTransform(marginal_density=self.marginal_density)
        return StandardizeTransform()

    # -----------------------------------------------------------------
    # training
    # -----------------------------------------------------------------
    def _nll(self, model, X: torch.Tensor) -> torch.Tensor:
        """Differentiable mean NLL (plus KL term when Bayesian)."""
        if self.bayesian:
            kl = model.kl_divergence()
            with model.reparameterize() as m:
                return -m().log_prob(X).mean() + self.bayes_kl_weight * kl
        return -model().log_prob(X).mean()

    @torch.no_grad()
    def _eval_nll(self, model, X: torch.Tensor) -> float:
        """
        Mean NLL of `model` on `X`, evaluation only. For a Bayesian model
        this averages over `bayes_n_eval_samples` posterior weight draws --
        a cheap Monte Carlo estimate of the posterior predictive NLL, so the
        number stays comparable across epochs instead of being dominated by
        which weights happened to be drawn.
        """
        if self.bayesian:
            vals = []
            for _ in range(self.bayes_n_eval_samples):
                with model.reparameterize() as m:
                    vals.append((-m().log_prob(X).mean()).item())
            return float(np.mean(vals))
        return float((-model().log_prob(X).mean()).item())

    def _train_loop(
        self, model, X_train, X_val, epochs, patience, lr, weight_decay,
        batch_size, verbose, label="",
    ):
        """
        The MLE training loop shared by the validation run and the full-data
        refit.

        When `batch_size` is None every epoch is one full-batch gradient
        step -- cheap and stable for the few-thousand-row datasets this
        library targets. Set a `batch_size` for larger data, and each epoch
        becomes a shuffled pass of minibatch steps instead.

        With `X_val`, the best-validation-NLL state is checkpointed and
        training stops after `patience` epochs without improvement. With
        `X_val=None` (the full-data refit, where nothing is held out) the
        best-training-NLL state is kept instead.

        Returns: (model, history, best_epoch), with `model` restored to its
        best checkpoint.
        """
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val, best_state, best_epoch, patience_ctr = np.inf, None, 0, 0
        history: dict[str, list[float]] = {"train": [], "val": []}
        n = len(X_train)
        report_every = max(epochs // 10, 1)

        for epoch in range(epochs):
            model.train()
            if batch_size is None or batch_size >= n:
                opt.zero_grad()
                loss = self._nll(model, X_train)
                loss.backward()
                opt.step()
                train_nll = float(loss.item())
            else:
                perm = torch.randperm(n, device=X_train.device)
                losses = []
                for start in range(0, n, batch_size):
                    idx = perm[start:start + batch_size]
                    opt.zero_grad()
                    loss = self._nll(model, X_train[idx])
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                train_nll = float(np.mean(losses))

            model.eval()
            val_nll = self._eval_nll(model, X_val) if X_val is not None else train_nll
            history["train"].append(train_nll)
            history["val"].append(val_nll)

            if not np.isfinite(val_nll):
                warnings.warn(
                    f"Non-finite NLL at epoch {epoch}; stopping training early. "
                    "Try a lower learning rate, fewer transforms, or copula_mode=True.",
                    RuntimeWarning, stacklevel=2,
                )
                break

            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch, patience_ctr = epoch, 0
            else:
                patience_ctr += 1

            if verbose and epoch % report_every == 0:
                print(f"{label}epoch {epoch:4d}  train_nll={train_nll:8.3f}  val_nll={val_nll:8.3f}")

            if patience is not None and patience_ctr >= patience:
                if verbose:
                    print(f"{label}early stop at epoch {epoch} (best epoch {best_epoch})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, history, best_epoch

    def fit(
        self,
        df: pd.DataFrame,
        val_frac: float = 0.2,
        epochs: int = 1000,
        patience: int | None = 40,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        verbose: bool = True,
        refit_on_full_data: bool = True,
        split_method: str = "chronological",
        embargo: int = 0,
    ) -> NeuroCopula:
        """
        Fit the copula to a DataFrame.

        The procedure is deliberately two-stage:

        1. Fit the marginal transform, split into train/validation, and
           train on the train part while tracking validation NLL. This
           diagnoses overfitting and finds the right number of epochs
           (`best_epoch`).
        2. If `refit_on_full_data`, throw that model away and train a fresh
           one on 100% of the rows for exactly ``best_epoch + 1`` epochs.
           The split's only job was choosing the epoch count, so the
           deployed model is not permanently missing the held-out rows.

        Note that the marginal transform is fitted on the full DataFrame
        before the split. For the marginals that is intentional -- the
        empirical CDF is what defines the copula scale, and refitting it per
        split would make the two runs incomparable -- but it does mean the
        validation NLL is not a fully clean out-of-sample number. When you
        need an untainted estimate (a benchmark, a paper), hold out a test
        set *before* calling `fit` and score it with `log_prob` /
        `copula_log_prob`; `neurocopula.benchmark` does exactly that.

        Parameters
        ----------
        df:
            DataFrame of continuous columns. Row ORDER matters when
            ``split_method="chronological"`` -- pass it sorted oldest-first
            for time-series data.
        val_frac:
            Fraction of rows held out for validation (at least 20 rows).
        epochs:
            Maximum epochs for the validation run.
        patience:
            Epochs without validation improvement before stopping early, or
            None to always run all `epochs`. The full-data refit never
            early-stops -- there is no held-out set left to check against.
        lr:
            Adam learning rate.
        weight_decay:
            Adam weight decay (L2). A small value (1e-5 to 1e-4) is a cheap
            extra guard against overfitting on small datasets.
        batch_size:
            None (default) trains full-batch, one gradient step per epoch.
            Set an integer for minibatch training on larger datasets.
        verbose:
            Print split info, progress, and a fit summary with an
            overfitting flag.
        refit_on_full_data:
            See the procedure above. False leaves the model trained on the
            train split only.
        split_method:
            ``"chronological"`` (default) validates on the most recent
            `val_frac` of rows in the order given -- the right choice for
            time series, because it tests whether the fitted dependence
            generalises FORWARD in time and will not hide regime shifts the
            way a shuffled split does. ``"random"`` is a classic shuffled
            split, appropriate only when rows are genuinely exchangeable.
        embargo:
            Rows dropped at the train/validation boundary (chronological
            splits only), to stop short-horizon autocorrelation leaking
            across it. 0 is fine for daily-or-slower data.

        Returns: self.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit() expects a pandas DataFrame")
        if split_method not in ("chronological", "random"):
            raise ValueError("split_method must be 'chronological' or 'random'")
        if not 0 < val_frac < 1:
            raise ValueError("val_frac must be strictly between 0 and 1")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.columns = list(df.columns)
        data = self._validate_frame(df)
        n, d = data.shape

        n_val = min(max(int(n * val_frac), 20), n - 1)
        if n <= 40:
            warnings.warn(
                f"Only {n} rows: a flow has many parameters and this fit is very "
                "likely to overfit. Treat the results as illustrative.",
                RuntimeWarning, stacklevel=2,
            )

        self.marginals_ = self._make_transform().fit(data)
        data_std = self.marginals_.forward(data)

        if split_method == "chronological":
            val_idx = np.arange(n - n_val, n)
            train_idx = np.arange(0, max(n - n_val - embargo, 1))
        else:
            idx = np.random.permutation(n)
            val_idx, train_idx = idx[:n_val], idx[n_val:]

        if verbose:
            extra = f", embargo={embargo}" if split_method == "chronological" and embargo else ""
            print(
                f"split={split_method} (train n={len(train_idx)}, val n={len(val_idx)}{extra})  "
                f"copula_mode={self.copula_mode}  bayesian={self.bayesian}  "
                f"sampling_mode={self.sampling_mode}  device={self.device}"
            )

        X_train = self._to_tensor(data_std[train_idx])
        X_val = self._to_tensor(data_std[val_idx])
        X_full = self._to_tensor(data_std)

        model = self._make_model(d)
        model, self.history, self.best_epoch = self._train_loop(
            model, X_train, X_val, epochs, patience, lr, weight_decay,
            batch_size, verbose, label="[val split] " if verbose else "",
        )
        self._install(model)

        final_train = self._eval_nll(model, X_train)
        final_val = self._eval_nll(model, X_val)
        gap = final_val - final_train
        self.fit_report_ = {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": d,
            "best_epoch": self.best_epoch,
            "final_train_nll": final_train,
            "final_val_nll": final_val,
            "overfit_gap": gap,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "refit_on_full_data": bool(refit_on_full_data),
        }

        if verbose:
            print("\n--- fit summary (from the train/val split) ---")
            for k, v in self.fit_report_.items():
                print(f"  {k}: {v}")
            flag = ("looks OK" if gap < 0.5 else
                    "POSSIBLE OVERFITTING -- consider more data, fewer transforms, "
                    "narrower hidden_features, or weight_decay")
            print(f"  overfitting check (val - train = {gap:.3f}): {flag}")

        if refit_on_full_data:
            if verbose:
                print(f"\nrefitting on 100% of the data for {self.best_epoch + 1} epochs...")
            torch.manual_seed(self.seed)
            full_model = self._make_model(d)
            full_model, full_hist, _ = self._train_loop(
                full_model, X_full, None, self.best_epoch + 1,
                patience=None, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, verbose=False,
            )
            self._install(full_model)
            self.fit_report_["full_data_final_nll"] = full_hist["train"][-1]
            if verbose:
                print(f"done. final full-data train NLL: {full_hist['train'][-1]:.3f}")

        return self

    def _install(self, model) -> None:
        """Store a trained model, unwrapping a posterior draw if Bayesian."""
        if self.bayesian:
            self.bayes = model
            self.flow = model.sample_model()
        else:
            self.bayes = None
            self.flow = model

    # -----------------------------------------------------------------
    # density evaluation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def log_prob(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Log-density of each row, in ORIGINAL units.

        With ``copula_mode=False`` this is exact. With ``copula_mode=True``
        it combines the exact copula log-density with KDE-estimated marginal
        log-densities (see `marginals.EmpiricalCopulaTransform`), so it is
        an estimate -- fine for comparing models fitted the same way, but
        `copula_log_prob` is the exact quantity and the one to use when
        comparing against a vine copula.

        Returns: array of shape (n,).
        """
        self._check_fitted()
        data = self._validate_frame(df, fitted=True)
        z = self.marginals_.forward(data)
        lp = self.flow().log_prob(self._to_tensor(z)).cpu().numpy()
        return lp + self.marginals_.log_det_jacobian(data)

    @torch.no_grad()
    def copula_log_prob(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Log copula density ``log c(u_1, ..., u_d)`` of each row, where
        ``u_j = F_j(x_j)`` are the pseudo-observations.

        This is the dependence structure alone, with every marginal effect
        divided out. It is the like-for-like number against a vine copula's
        ``Vinecop.pdf``, and it needs no marginal density estimate, so it is
        exact. Requires ``copula_mode=True``.

        Returns: array of shape (n,).
        """
        self._check_fitted()
        if not self.copula_mode:
            raise RuntimeError(
                "copula_log_prob requires copula_mode=True. With copula_mode=False "
                "the flow models the full joint, so its density is not a copula "
                "density; use log_prob() instead."
            )
        data = self._validate_frame(df, fitted=True)
        z = self.marginals_.forward(data)
        lp = self.flow().log_prob(self._to_tensor(z)).cpu().numpy()
        # c(u) = p_Z(z) / prod_j phi(z_j), since u_j = Phi(z_j).
        return lp - norm.logpdf(z).sum(axis=1)

    def score(self, df: pd.DataFrame | np.ndarray, copula: bool = False) -> float:
        """
        Mean log-likelihood of `df` -- higher is better. Set ``copula=True``
        to score the dependence structure alone via `copula_log_prob`.
        """
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        return float(np.mean(lp))

    def aic(self, df: pd.DataFrame | np.ndarray, copula: bool = False) -> float:
        """
        Akaike information criterion, ``2k - 2 * loglik``, with `k` the
        number of flow parameters. Lower is better.

        A caveat worth stating: a flow has thousands of parameters and is
        fitted with early stopping, so `k` badly overstates its effective
        degrees of freedom and AIC penalises it far more harshly than it
        penalises a vine copula with a handful of parameters. Held-out
        log-likelihood is the honest comparison; AIC is here because it is
        conventional, not because it is fair.
        """
        self._check_fitted()
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        k = sum(p.numel() for p in self.flow.parameters())
        return float(2 * k - 2 * lp.sum())

    def bic(self, df: pd.DataFrame | np.ndarray, copula: bool = False) -> float:
        """
        Bayesian information criterion, ``k * log(n) - 2 * loglik``. Lower is
        better.

        The same caveat as `aic` applies, only more so: BIC penalises each
        parameter by ``log(n)`` rather than 2, so against a vine copula with a
        handful of parameters a flow with thousands is essentially guaranteed
        to lose regardless of fit quality. It is provided for parity with
        `VineCopula.bic`; for an honest comparison use held-out `score`.
        """
        self._check_fitted()
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        k = sum(p.numel() for p in self.flow.parameters())
        return float(k * np.log(len(lp)) - 2 * lp.sum())

    # -----------------------------------------------------------------
    # sampling
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _sample_std_batched(self, n: int, batch_size: int = 10_000) -> np.ndarray:
        """
        Draw `n` samples in the flow's own training space, in batches.

        Batching matters: zuko's spline sampler builds intermediates larger
        than the output suggests, especially in autoregressive mode, so
        drawing 500k rows in one call can exhaust memory where the same
        draw in 10k chunks is fine.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        chunks, remaining = [], n
        dist = self.flow()
        while remaining > 0:
            b = min(batch_size, remaining)
            chunks.append(dist.sample((b,)).cpu().numpy())
            remaining -= b
        return np.concatenate(chunks, axis=0).astype(np.float64)

    def sample(self, n: int = 1000, batch_size: int = 10_000) -> pd.DataFrame:
        """
        Draw `n` synthetic rows from the fitted joint, in original units.

        Returns: DataFrame with the same columns as the one passed to `fit`.
        """
        self._check_fitted()
        s = self.marginals_.inverse(self._sample_std_batched(n, batch_size))
        return pd.DataFrame(s, columns=self.columns)

    def sample_uniform(self, n: int = 1000, batch_size: int = 10_000) -> pd.DataFrame:
        """
        Draw `n` rows on the COPULA scale: pseudo-observations in (0, 1)^d,
        marginals uniform by construction.

        This is what to plot when you want to see the dependence structure
        with marginal shape stripped out, and what to hand to any code that
        expects copula-scale data (`pyvinecopulib` included). Requires
        ``copula_mode=True``.
        """
        self._check_fitted()
        if not self.copula_mode:
            raise RuntimeError("sample_uniform requires copula_mode=True.")
        z = self._sample_std_batched(n, batch_size)
        return pd.DataFrame(norm.cdf(z), columns=self.columns)

    def pseudo_observations(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Map `df` (or, by default, the data the model was fitted on) to
        copula-scale pseudo-observations using the fitted empirical CDFs.
        Requires ``copula_mode=True``.
        """
        self._check_fitted()
        if not self.copula_mode:
            raise RuntimeError("pseudo_observations requires copula_mode=True.")
        if df is None:
            data = self.marginals_._sorted
        else:
            data = self._validate_frame(df, fitted=True)
        return pd.DataFrame(self.marginals_.to_uniform(data), columns=self.columns)

    # -----------------------------------------------------------------
    # probabilistic queries
    # -----------------------------------------------------------------
    def joint_exceedance_probability(
        self, conditions: Mapping[str, tuple[str, float]], n_mc: int = 50_000
    ) -> float:
        """
        Monte Carlo estimate of the probability that ALL conditions hold at
        once -- "how likely is it that these all blow up together":

            P(asset_1 < -0.02 AND asset_2 < -0.015)

        conditions:
            ``{column: (op, value)}`` with `op` in ``<``, ``<=``, ``>``,
            ``>=``, values in original units.
        n_mc:
            Monte Carlo draws. For a rare event, make sure this is large
            enough that more than a handful of draws satisfy the conditions
            -- otherwise the estimate is mostly noise. `n_mc=50_000` resolves
            probabilities down to roughly 1e-3 usefully.

        Returns: the estimated probability.
        """
        self._check_fitted()
        samp = self.sample(n_mc)
        mask = np.ones(len(samp), dtype=bool)
        for col, (op, val) in conditions.items():
            self._check_column(col)
            mask &= self._apply_op(samp[col].values, op, val)
        return float(mask.mean())

    def tail_dependence(
        self, col1: str, col2: str, q: float = 0.05, tail: str = "lower",
        n_mc: int = 50_000,
    ) -> float:
        """
        Tail dependence coefficient between two columns, estimated from the
        fitted model's own samples:

            lambda_L(q) = P(U2 < q | U1 < q)          (tail="lower")
            lambda_U(q) = P(U2 > 1-q | U1 > 1-q)      (tail="upper")

        This is the quantity that decides whether two things crash together.
        A Gaussian copula forces it to zero as ``q -> 0`` no matter how high
        the correlation; a flow is free to keep it positive, which is the
        main reason to reach for one.

        q:
            Tail probability. Smaller probes further into the tail but needs
            a larger `n_mc` to stay stable -- the estimate uses only the
            ``q * n_mc`` draws in that tail.
        tail: ``"lower"`` or ``"upper"``.

        Returns: the coefficient, in [0, 1].
        """
        self._check_fitted()
        self._check_column(col1)
        self._check_column(col2)
        if not 0 < q < 0.5:
            raise ValueError("q must be in (0, 0.5)")
        if tail not in ("lower", "upper"):
            raise ValueError("tail must be 'lower' or 'upper'")

        samp = self.sample(n_mc)
        a, b = samp[col1].values, samp[col2].values
        if tail == "lower":
            t1, t2 = np.quantile(a, q), np.quantile(b, q)
            cond, joint = a < t1, (a < t1) & (b < t2)
        else:
            t1, t2 = np.quantile(a, 1 - q), np.quantile(b, 1 - q)
            cond, joint = a > t1, (a > t1) & (b > t2)
        denom = cond.sum()
        if denom == 0:
            return float("nan")
        return float(joint.sum() / denom)

    def tail_dependence_curve(
        self, col1: str, col2: str, quantiles: Sequence[float] | None = None,
        tail: str = "lower", n_mc: int = 100_000,
    ) -> pd.DataFrame:
        """
        `tail_dependence` evaluated across a range of `q`, from one shared
        Monte Carlo sample.

        The curve is far more informative than a single number: a Gaussian
        copula's decays to zero as q shrinks, while a copula with genuine
        tail dependence flattens out at a positive level. Plot it with
        `neurocopula.plotting.plot_tail_dependence_curve`.

        Returns: DataFrame with columns ``q`` and ``lambda``.
        """
        self._check_fitted()
        self._check_column(col1)
        self._check_column(col2)
        if quantiles is None:
            quantiles = np.geomspace(0.005, 0.25, 20)

        samp = self.sample(n_mc)
        a, b = samp[col1].values, samp[col2].values
        rows = []
        for q in quantiles:
            if tail == "lower":
                cond = a < np.quantile(a, q)
                joint = cond & (b < np.quantile(b, q))
            else:
                cond = a > np.quantile(a, 1 - q)
                joint = cond & (b > np.quantile(b, 1 - q))
            denom = cond.sum()
            rows.append({"q": float(q),
                         "lambda": float(joint.sum() / denom) if denom else np.nan})
        return pd.DataFrame(rows)

    def conditional_distribution(
        self, target: str, given: Mapping[str, float], tol: float = 0.35,
        n_mc: int = 60_000,
    ) -> pd.DataFrame:
        """
        Approximate ``p(target | given)`` by importance-weighted resampling.

        The flow is fitted unconditionally (no context variable), so
        conditioning is done by drawing a large sample from the joint and
        weighting each draw by a Gaussian kernel around the conditioning
        values, evaluated in the flow's own space. That is simple and works
        for any subset of columns without refitting, at the cost of a
        bandwidth choice and Monte Carlo noise.

        target:
            Column whose conditional distribution you want.
        given:
            ``{column: value}`` in ORIGINAL units. Multiple columns are
            allowed; their kernel weights multiply.
        tol:
            Kernel bandwidth in the flow's space (roughly standard-deviation
            units there). Smaller conditions more sharply but keeps fewer
            effective samples; 0.2-0.5 is a reasonable range.
        n_mc:
            Draws taken before weighting.

        Returns: DataFrame with ``value`` (draws of `target`, original
            units) and ``weight`` (summing to 1).
            ``result.attrs["effective_sample_size"]`` is ``1 / sum(w^2)``:
            if it is small relative to `n_mc`, the conditioning point sits
            in a sparse region and the estimate is noisy -- raise `tol` or
            `n_mc`. A warning is emitted below 100.
        """
        self._check_fitted()
        self._check_column(target)
        if not given:
            raise ValueError("`given` must contain at least one column")
        if tol <= 0:
            raise ValueError("tol must be positive")

        samp_std = self._sample_std_batched(n_mc)
        col_idx = {c: i for i, c in enumerate(self.columns)}

        # Map the conditioning values into the flow's own space, where the
        # kernel is applied -- distances are comparable there, which they
        # are not in original units across columns of different scale.
        given_std = {}
        for col, val in given.items():
            self._check_column(col)
            # forward() works on whole rows, so build a dummy row at the
            # centre of the flow's space, overwrite the one entry we care
            # about, and read that entry back out after the transform.
            filler = self.marginals_.inverse(np.zeros((1, len(self.columns))))
            filler[0, col_idx[col]] = val
            given_std[col] = float(self.marginals_.forward(filler)[0, col_idx[col]])

        log_w = np.zeros(n_mc)
        for col, val_std in given_std.items():
            diff = (samp_std[:, col_idx[col]] - val_std) / tol
            log_w += -0.5 * diff ** 2
        w = np.exp(log_w - log_w.max())
        w /= w.sum()

        eff_n = float(1.0 / np.sum(w ** 2))
        if eff_n < 100:
            warnings.warn(
                f"effective sample size is only {eff_n:.0f} out of {n_mc} draws: this "
                "conditional estimate is noisy. Increase tol or n_mc.",
                RuntimeWarning, stacklevel=2,
            )

        target_vals = self.marginals_.inverse(samp_std)[:, col_idx[target]]
        out = pd.DataFrame({"value": target_vals, "weight": w})
        out.attrs["effective_sample_size"] = eff_n
        return out

    def conditional_probability(
        self, target: str, op: str, threshold: float, given: Mapping[str, float],
        tol: float = 0.35, n_mc: int = 60_000,
    ) -> tuple[float, float]:
        """
        ``P(target <op> threshold | given)``, via the same weighting as
        `conditional_distribution`::

            model.conditional_probability("asset_2", "<", -0.02,
                                          given={"asset_1": -0.03})

        Contrast that with the same call at ``given={"asset_1": 0.0}`` and
        you have measured tail dependence as a directly interpretable
        conditional probability rather than a lambda coefficient.

        Returns: ``(probability, effective_sample_size)``. Check the second
            value is not tiny before trusting the first.
        """
        cd = self.conditional_distribution(target, given, tol=tol, n_mc=n_mc)
        v, w = cd["value"].values, cd["weight"].values
        return float(w[self._apply_op(v, op, threshold)].sum()), cd.attrs["effective_sample_size"]

    def conditional_quantiles(
        self, target: str, given: Mapping[str, float],
        quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
        tol: float = 0.35, n_mc: int = 60_000,
    ) -> pd.Series:
        """
        Weighted quantiles of ``p(target | given)`` -- a compact summary of
        `conditional_distribution`, e.g. a conditional 5% VaR.

        Returns: Series indexed by the requested quantiles.
        """
        cd = self.conditional_distribution(target, given, tol=tol, n_mc=n_mc)
        order = np.argsort(cd["value"].values)
        v = cd["value"].values[order]
        cw = np.cumsum(cd["weight"].values[order])
        return pd.Series(np.interp(quantiles, cw, v), index=list(quantiles), name=target)

    # -----------------------------------------------------------------
    # uncertainty
    # -----------------------------------------------------------------
    def uncertainty_report(
        self, metric_fn: Callable[[NeuroCopula], float], n_posterior: int = 20,
        verbose: bool = False,
    ) -> dict:
        """
        Epistemic uncertainty on any query, by re-running it once per
        posterior weight draw. Requires ``bayesian=True``::

            model.uncertainty_report(lambda m: m.tail_dependence("A", "B", q=0.05))

        The spread you get back answers "how much would this number move if
        I had drawn a different sample of the same size" -- which is a
        different and usually more important question than the spread of the
        density itself.

        metric_fn: callable taking the model and returning a float.
        n_posterior: number of posterior weight draws.

        Returns: dict with ``samples``, ``mean``, ``std``, ``q05``, ``q95``
            (a 90% credible interval).
        """
        if not self.bayesian or self.bayes is None:
            raise RuntimeError(
                "uncertainty_report requires bayesian=True at construction "
                "(and a completed fit)."
            )
        original = self.flow
        results = []
        try:
            for i in range(n_posterior):
                self.flow = self.bayes.sample_model()
                val = float(metric_fn(self))
                results.append(val)
                if verbose:
                    print(f"  posterior draw {i}: {val:.4f}")
        finally:
            self.flow = original
        arr = np.asarray(results, dtype=float)
        return {
            "samples": arr,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "q05": float(np.quantile(arr, 0.05)),
            "q95": float(np.quantile(arr, 0.95)),
        }

    # -----------------------------------------------------------------
    # persistence
    # -----------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """
        Save the fitted model (flow weights, marginal transform, config,
        fit report) to a single file via `torch.save`.

        The file contains a pickled Python object, so only load files you
        trust -- the same caveat that applies to any `torch.save` artifact.
        """
        self._check_fitted()
        path = Path(path)
        torch.save(
            {
                "format_version": 1,
                "config": self.get_params(),
                "columns": self.columns,
                "marginals": self.marginals_,
                "state_dict": (self.bayes if self.bayesian else self.flow).state_dict(),
                "history": self.history,
                "best_epoch": self.best_epoch,
                "fit_report": self.fit_report_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> NeuroCopula:
        """
        Load a model saved by `save`. Only load files you trust.
        """
        blob = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        cfg = dict(blob["config"])
        if device is not None:
            cfg["device"] = device
        model = cls(**cfg)
        model.columns = blob["columns"]
        model.marginals_ = blob["marginals"]
        model.history = blob["history"]
        model.best_epoch = blob["best_epoch"]
        model.fit_report_ = blob["fit_report"]
        net = model._make_model(len(model.columns))
        net.load_state_dict(blob["state_dict"])
        net.eval()
        model._install(net)
        return model

    def get_params(self) -> dict:
        """Constructor arguments as a plain dict (for cloning or logging)."""
        return {
            "transforms": self.transforms,
            "hidden_features": self.hidden_features,
            "bins": self.bins,
            "seed": self.seed,
            "sampling_mode": self.sampling_mode,
            "copula_mode": self.copula_mode,
            "marginal_density": self.marginal_density,
            "device": str(self.device),
            "bayesian": self.bayesian,
            "bayes_init_logvar": self.bayes_init_logvar,
            "bayes_kl_weight": self.bayes_kl_weight,
            "bayes_n_eval_samples": self.bayes_n_eval_samples,
        }

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _validate_frame(self, df, fitted: bool = False) -> np.ndarray:
        """Coerce input to a clean float array with the expected columns."""
        if isinstance(df, pd.DataFrame):
            if fitted:
                missing = [c for c in self.columns if c not in df.columns]
                if missing:
                    raise KeyError(f"missing columns the model was fitted on: {missing}")
                df = df[self.columns]
            data = df.values.astype(np.float64)
        else:
            data = np.atleast_2d(np.asarray(df, dtype=np.float64))
            if fitted and data.shape[1] != len(self.columns):
                raise ValueError(
                    f"expected {len(self.columns)} columns, got {data.shape[1]}"
                )
        if data.ndim != 2:
            raise ValueError(f"expected 2-D data, got shape {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError(
                "data contains NaN or infinite values; drop or impute them first "
                "(e.g. df.dropna())"
            )
        if not fitted:
            constant = [self.columns[j] for j in range(data.shape[1])
                        if np.ptp(data[:, j]) == 0]
            if constant:
                raise ValueError(
                    f"these columns are constant and have no distribution to fit: {constant}"
                )
        return data

    def _check_fitted(self) -> None:
        if self.flow is None:
            raise RuntimeError("this NeuroCopula is not fitted yet; call fit(df) first.")

    def _check_column(self, col: str) -> None:
        if col not in self.columns:
            raise KeyError(f"unknown column {col!r}; the model was fitted on {self.columns}")

    @staticmethod
    def _apply_op(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
        try:
            return _OPS[op](values, threshold)
        except KeyError:
            raise ValueError(
                f"unsupported operator {op!r}; expected one of {sorted(_OPS)}"
            ) from None

    def __repr__(self) -> str:
        state = "unfitted"
        if self.flow is not None and self.fit_report_ is not None:
            state = (f"fitted on {self.fit_report_['n_train'] + self.fit_report_['n_val']} rows "
                     f"x {self.fit_report_['n_features']} cols")
        return (f"NeuroCopula(transforms={self.transforms}, "
                f"hidden_features={self.hidden_features}, "
                f"copula_mode={self.copula_mode}, bayesian={self.bayesian}, {state})")
