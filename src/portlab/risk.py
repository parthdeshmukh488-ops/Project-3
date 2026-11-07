"""Value at risk and expected shortfall, by five methods that disagree.

The disagreement is the content.  A 99% one-day VaR is a statement about the
hundredth-worst day in a hundred, and the methods differ in what they assume
about that day:

:func:`gaussian_var`
    Returns are normal.  Fast, and wrong in the direction that matters --
    financial returns have fat tails, so it understates the loss exactly when
    the number is being relied on.
:func:`cornish_fisher_var`
    Normal, adjusted for the sample skewness and kurtosis.  A cheap
    improvement that captures most of the fat-tail correction without a new
    distribution.
:func:`historical_var`
    No distribution at all: just the empirical quantile.  Honest about the
    shape of the tail, and completely blind to the fact that volatility
    changes -- a quiet year of data gives a small VaR on the first day of a
    crisis.
:func:`filtered_historical_var`
    Fit GARCH, standardise, take the empirical quantile of the standardised
    residuals, then rescale by *tomorrow's* forecast volatility.  Keeps the
    empirical tail shape and adapts to the volatility regime, which is the
    combination that passes the backtests in :mod:`portlab.backtest_var` when
    the others do not.
:func:`evt_var`
    Fit a generalised Pareto distribution to the exceedances over a high
    threshold.  Extreme value theory says the tail *must* look like this
    asymptotically whatever the underlying distribution, so this is the only
    method with a defensible basis for extrapolating past the worst
    observation in the sample -- which is what a 99.9% VaR requires.

Expected shortfall is reported alongside each, because VaR says nothing about
how bad the breach is when it happens, and is not subadditive: the VaR of a
combined book can exceed the sum of its parts, which makes it unusable for
allocating risk across desks.  Expected shortfall is coherent and does not
have that problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri

__all__ = [
    "RiskMeasure",
    "GarchFit",
    "gaussian_var",
    "cornish_fisher_var",
    "historical_var",
    "fit_garch",
    "filtered_historical_var",
    "evt_var",
    "component_var",
]


@dataclass(frozen=True)
class RiskMeasure:
    """A VaR/ES pair, both reported as positive loss magnitudes."""

    var: float
    expected_shortfall: float
    confidence: float
    method: str

    def __str__(self):
        return (
            f"{self.method}: {self.confidence:.1%} VaR = {self.var:.4%}, "
            f"ES = {self.expected_shortfall:.4%}"
        )


def _normal_es_multiplier(confidence):
    """``E[Z | Z < q]`` for a standard normal, as a positive number."""
    from scipy.stats import norm

    q = ndtri(1.0 - confidence)
    return float(norm.pdf(q) / (1.0 - confidence))


def gaussian_var(returns, confidence=0.99):
    """Normal VaR and ES from the sample mean and standard deviation."""
    r = np.asarray(returns, dtype=float).ravel()
    mean, sd = r.mean(), r.std(ddof=1)
    var = -(mean + ndtri(1.0 - confidence) * sd)
    shortfall = -mean + sd * _normal_es_multiplier(confidence)
    return RiskMeasure(float(var), float(shortfall), confidence, "gaussian")


def cornish_fisher_var(returns, confidence=0.99):
    """Normal quantile expanded for skewness and excess kurtosis.

    The Cornish-Fisher expansion corrects the normal quantile using the third
    and fourth moments.  It works when the tails are mildly heavier than
    normal and fails loudly when they are not, because it is an asymptotic
    expansion in moments that may not exist: on Student-t returns with four
    degrees of freedom, whose population kurtosis is infinite, it returns a
    99% VaR of 6.2% against a historical 2.7% and an extreme-value 2.7%.

    Treat it as a cheap correction to a normal assumption, not as a tail
    model.  For an actual tail model use :func:`evt_var`, which is built on
    the one theorem that says what tails have to look like.
    """
    r = np.asarray(returns, dtype=float).ravel()
    mean, sd = r.mean(), r.std(ddof=1)
    standardised = (r - mean) / sd
    skew = float(np.mean(standardised**3))
    excess = float(np.mean(standardised**4)) - 3.0

    z = ndtri(1.0 - confidence)
    adjusted = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    var = -(mean + adjusted * sd)
    tail = r[r <= mean + adjusted * sd]
    shortfall = -float(tail.mean()) if tail.size else float(var)
    return RiskMeasure(float(var), shortfall, confidence, "cornish-fisher")


def historical_var(returns, confidence=0.99):
    """The empirical quantile, and the average loss beyond it."""
    r = np.asarray(returns, dtype=float).ravel()
    cutoff = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= cutoff]
    shortfall = -float(tail.mean()) if tail.size else -cutoff
    return RiskMeasure(-cutoff, shortfall, confidence, "historical")


@dataclass(frozen=True)
class GarchFit:
    """A fitted GARCH(1,1), with the conditional volatility path."""

    omega: float
    alpha: float
    beta: float
    volatility: np.ndarray
    residuals: np.ndarray
    log_likelihood: float

    @property
    def persistence(self):
        """``alpha + beta``: how slowly a volatility shock decays.

        Equity indices come in around 0.98, a shock half-life of about 35
        days.  At exactly one the process is integrated and has no
        unconditional variance at all.

        **A structural break in volatility inflates this, badly.**  Fitting a
        series whose volatility level shifts partway through -- which is what
        a crisis looks like -- drives the estimate to the stationarity
        boundary, because a single persistent process is the only way the
        model can express a permanent change in level.  Measured on the
        regime-change series in :mod:`portlab.market`: persistence comes out
        at the 0.9999 cap, a shock half-life of 6,900 days, and an implied
        unconditional volatility of 122% annualised, against a true level of
        12% rising to 38%.  Fitted on clean GARCH data the same code returns
        0.959 and a half-life of 16 days.

        This is the Lamoureux-Lastrapes effect, and the practical consequence
        is narrow: :attr:`unconditional_variance` and long-horizon forecasts
        become meaningless on broken data, while the one-step-ahead forecast
        -- the only thing a daily VaR needs -- stays accurate, because it is
        driven by the recursion rather than by the mean-reversion target.
        """
        return self.alpha + self.beta

    @property
    def unconditional_variance(self):
        if self.persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - self.persistence)

    def forecast(self, horizon=1):
        """Conditional variance forecast, iterated from the last observation."""
        variance = self.volatility[-1] ** 2
        shock = self.residuals[-1] ** 2
        forecasts = []
        for step in range(int(horizon)):
            variance = (
                self.omega + self.alpha * shock + self.beta * variance
                if step == 0
                else self.omega + self.persistence * variance
            )
            forecasts.append(variance)
        return np.sqrt(np.asarray(forecasts))


def _garch_recursion(params, residuals):
    omega, alpha, beta = params
    n = residuals.size
    variance = np.empty(n)
    variance[0] = residuals.var() if residuals.var() > 0 else 1e-12
    for t in range(1, n):
        variance[t] = omega + alpha * residuals[t - 1] ** 2 + beta * variance[t - 1]
    return variance


def fit_garch(returns, max_iter=500):
    """Gaussian maximum-likelihood GARCH(1,1).

    ``sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2``, fitted by
    maximising the exact Gaussian likelihood of the recursion.  Parameters are
    bounded away from the stationarity boundary (``alpha + beta <= 0.9999``)
    because real equity data sits close enough to it that an unconstrained
    optimiser will cross it, and an integrated fit has no unconditional
    variance for the forecast to revert to.

    Scaling matters more than it should: daily returns are ``O(0.01)`` so
    ``omega`` is ``O(1e-6)``, and an optimiser working in those units stops
    long before it has converged.  Returns are therefore rescaled to unit
    variance for the fit and the parameters scaled back afterwards.
    """
    r = np.asarray(returns, dtype=float).ravel()
    residuals = r - r.mean()
    scale = residuals.std()
    if scale <= 0:
        raise ValueError("returns have no variation")
    scaled = residuals / scale

    def negative_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
            return 1e10
        variance = _garch_recursion(params, scaled)
        if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
            return 1e10
        return 0.5 * float(
            np.sum(np.log(2.0 * np.pi * variance) + scaled**2 / variance)
        )

    best = None
    for start in ((0.05, 0.10, 0.85), (0.20, 0.05, 0.75), (0.01, 0.20, 0.70)):
        result = minimize(
            negative_log_likelihood,
            np.asarray(start),
            method="SLSQP",
            bounds=[(1e-8, 10.0), (0.0, 0.9999), (0.0, 0.9999)],
            constraints=[
                {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},
            ],
            options={"maxiter": max_iter, "ftol": 1e-12},
        )
        if best is None or result.fun < best.fun:
            best = result

    omega, alpha, beta = best.x
    variance = _garch_recursion(best.x, scaled)
    return GarchFit(
        omega=float(omega * scale**2),
        alpha=float(alpha),
        beta=float(beta),
        volatility=np.sqrt(variance) * scale,
        residuals=residuals,
        log_likelihood=float(-best.fun),
    )


def filtered_historical_var(returns, confidence=0.99, fit=None):
    """GARCH-filtered historical simulation.

    Standardise by the fitted conditional volatility, take the empirical
    quantile of the standardised residuals -- which keeps the real tail shape,
    unlike a normal assumption -- then rescale by the one-step-ahead
    volatility forecast.  The result is a VaR that moves with the volatility
    regime instead of lagging it by the length of the estimation window.
    """
    r = np.asarray(returns, dtype=float).ravel()
    fit = fit_garch(r) if fit is None else fit
    standardised = fit.residuals / fit.volatility
    forecast = float(fit.forecast(1)[0])

    cutoff = float(np.quantile(standardised, 1.0 - confidence))
    tail = standardised[standardised <= cutoff]
    var = -(r.mean() + cutoff * forecast)
    shortfall = -(r.mean() + float(tail.mean()) * forecast) if tail.size else var
    return RiskMeasure(float(var), float(shortfall), confidence,
                       "filtered-historical")


def _gpd_log_likelihood(params, exceedances):
    shape, scale = params
    if scale <= 0:
        return 1e10
    z = exceedances / scale
    if shape >= 0:
        if np.any(1.0 + shape * z <= 0):
            return 1e10
        return float(
            exceedances.size * np.log(scale)
            + (1.0 + 1.0 / shape) * np.sum(np.log1p(shape * z))
            if abs(shape) > 1e-8
            else exceedances.size * np.log(scale) + np.sum(z)
        )
    if np.any(1.0 + shape * z <= 0):
        return 1e10
    return float(
        exceedances.size * np.log(scale)
        + (1.0 + 1.0 / shape) * np.sum(np.log1p(shape * z))
    )


def evt_var(returns, confidence=0.99, threshold_quantile=0.90):
    """Peaks-over-threshold VaR from a fitted generalised Pareto tail.

    The Pickands-Balkema-de Haan theorem says that for almost any underlying
    distribution, exceedances over a high enough threshold converge to a
    generalised Pareto.  That is the only principled basis for quoting a VaR
    beyond the worst loss in the sample -- a historical quantile simply cannot
    go there, and a normal one goes there confidently and wrongly.

    The threshold is the usual bias-variance trade: too low and the asymptotic
    result does not apply, too high and there is nothing left to fit.  The 90th
    percentile of losses is the conventional compromise and is exposed so it
    can be varied, which is the honest way to use the method -- a tail estimate
    that moves a lot with the threshold is not to be trusted.
    """
    r = np.asarray(returns, dtype=float).ravel()
    losses = -r
    threshold = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > threshold] - threshold
    if exceedances.size < 10:
        raise ValueError("too few exceedances to fit a tail; lower the threshold")

    result = minimize(
        _gpd_log_likelihood,
        np.array([0.1, exceedances.mean()]),
        args=(exceedances,),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000},
    )
    shape, scale = result.x

    n, n_exceed = losses.size, exceedances.size
    tail_fraction = n_exceed / n
    ratio = (1.0 - confidence) / tail_fraction

    if abs(shape) < 1e-8:
        var = threshold - scale * np.log(ratio)
    else:
        var = threshold + (scale / shape) * (ratio ** (-shape) - 1.0)

    # Closed-form GPD expected shortfall; finite only for shape < 1.
    if shape < 1.0:
        shortfall = (var + scale - shape * threshold) / (1.0 - shape)
    else:
        shortfall = float("inf")
    return RiskMeasure(float(var), float(shortfall), confidence, "evt-pot")


def component_var(weights, covariance, confidence=0.99):
    """Each position's contribution to Gaussian portfolio VaR.

    Gaussian VaR is homogeneous of degree one in the weights, so Euler's
    theorem splits it exactly across positions and the parts sum to the whole.
    This is what makes it possible to say which desk is using the risk budget
    -- and, because contributions can be negative, which one is paying for
    itself as a hedge.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    volatility = np.sqrt(max(float(w @ cov @ w), 1e-300))
    z = -ndtri(1.0 - confidence)
    return z * w * (cov @ w) / volatility
