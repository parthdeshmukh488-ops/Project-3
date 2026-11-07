"""Synthetic markets with a known covariance, and a regime change.

Portfolio claims are comparative -- this estimator beats that one out of
sample -- and comparisons need a truth to be measured against.  On real data
there is none: the out-of-sample covariance is itself an estimate, so a horse
race on historical returns partly measures which estimator resembles the
sample covariance of the test period.

:func:`factor_market` fixes that by generating returns from a covariance
matrix that is *known exactly*, so out-of-sample variance can be computed in
closed form as ``w' S w`` rather than estimated from a holdout.  The structure
is the one equity returns actually have: a dominant market factor, a few
sector factors, and idiosyncratic noise -- which is also what gives the
covariance matrix a handful of eigenvalues above the Marchenko-Pastur edge and
a bulk of indistinguishable ones below it.

:func:`regime_market` adds a volatility regime change partway through, which
is what separates a VaR model that adapts from one that does not.  A static
historical VaR calibrated on the quiet period breaches repeatedly and
consecutively once volatility triples; the point of
:mod:`portlab.backtest_var` is to detect precisely that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Market", "factor_market", "regime_market"]


@dataclass(frozen=True)
class Market:
    """Generated returns together with the covariance that produced them."""

    returns: np.ndarray
    covariance: np.ndarray
    n_factors: int

    @property
    def n_obs(self):
        return self.returns.shape[0]

    @property
    def n_assets(self):
        return self.returns.shape[1]

    def true_variance(self, weights):
        """Exact out-of-sample variance of a portfolio -- no estimation."""
        w = np.asarray(weights, dtype=float).ravel()
        return float(w @ self.covariance @ w)

    def true_volatility(self, weights, periods_per_year=252):
        return float(np.sqrt(self.true_variance(weights) * periods_per_year))


def factor_market(
    n_assets=100,
    n_obs=500,
    n_factors=3,
    market_vol=0.16,
    factor_vol=0.08,
    idiosyncratic_vol=0.25,
    periods_per_year=252,
    seed=0,
):
    """Returns from a market factor, sector factors and idiosyncratic noise.

    The first factor loads positively on every asset -- the market -- with
    loadings drawn around one, which is what produces the single large
    eigenvalue every real equity correlation matrix has.  The remaining
    factors load on random subsets with either sign, giving the sector
    structure that shows up as the next few eigenvalues above the noise bulk.

    Volatilities are annualised on input and converted internally, so the
    defaults read as the numbers a practitioner would recognise: a 16% market,
    8% sector factors and 25% single-stock noise.
    """
    rng = np.random.default_rng(seed)
    daily = 1.0 / np.sqrt(periods_per_year)

    loadings = np.empty((n_assets, n_factors))
    loadings[:, 0] = rng.normal(1.0, 0.3, n_assets)
    for k in range(1, n_factors):
        loadings[:, k] = rng.normal(0.0, 0.7, n_assets)

    factor_vols = np.full(n_factors, factor_vol * daily)
    factor_vols[0] = market_vol * daily
    specific = rng.uniform(0.6, 1.4, n_assets) * idiosyncratic_vol * daily

    covariance = (
        loadings @ np.diag(factor_vols**2) @ loadings.T + np.diag(specific**2)
    )

    factors = rng.standard_normal((n_obs, n_factors)) * factor_vols
    noise = rng.standard_normal((n_obs, n_assets)) * specific
    return Market(factors @ loadings.T + noise, covariance, n_factors)


def regime_market(
    n_obs=1500,
    switch_at=1000,
    quiet_vol=0.12,
    stressed_vol=0.38,
    persistence=0.94,
    degrees_of_freedom=5,
    periods_per_year=252,
    seed=0,
):
    """A single return series that changes volatility regime partway through.

    Three features, each chosen because a different VaR method fails on it:

    * a **level shift** in volatility at ``switch_at``, which a historical VaR
      calibrated on the quiet period cannot see coming;
    * **autocorrelated volatility** within each regime, which makes breaches
      cluster rather than arrive independently -- the thing
      :func:`portlab.backtest_var.christoffersen_independence` exists for;
    * **Student-t innovations**, so the tails are genuinely fatter than normal
      and a Gaussian VaR is wrong even within a regime.

    Returns ``(returns, conditional_volatility)``; the second is the truth a
    forecast can be scored against.
    """
    rng = np.random.default_rng(seed)
    daily = 1.0 / np.sqrt(periods_per_year)
    target = np.where(
        np.arange(n_obs) < switch_at, quiet_vol * daily, stressed_vol * daily
    )

    volatility = np.empty(n_obs)
    volatility[0] = target[0]
    for t in range(1, n_obs):
        volatility[t] = (
            persistence * volatility[t - 1] + (1.0 - persistence) * target[t]
        )

    scale = np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
    innovations = rng.standard_t(degrees_of_freedom, n_obs) / scale
    return volatility * innovations, volatility
