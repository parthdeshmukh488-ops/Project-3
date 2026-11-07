"""Covariance estimation, where portfolio optimisation actually fails.

Mean-variance optimisation has a reputation for being unusable in practice.
The usual explanation is that expected returns cannot be forecast, which is
true but incomplete: even with the means switched off entirely, a
minimum-variance portfolio built on a sample covariance matrix underperforms
an equal-weight portfolio out of sample for most realistic universe sizes.

The reason is that the sample covariance matrix is a bad estimator in a very
specific way.  With ``N`` assets and ``T`` observations there are
``N(N+1)/2`` parameters to estimate from ``NT`` numbers, so the ratio
``q = N/T`` controls everything.  Random matrix theory says the sample
eigenvalues of a *true identity* covariance spread out over
``[(1-sqrt q)^2, (1+sqrt q)^2]`` -- the Marchenko-Pastur law -- even though
every true eigenvalue is exactly one.  An optimiser inverts that matrix, so it
puts its largest bets precisely on the smallest eigenvalues, which are the
most badly estimated.  At ``q = 0.5`` the smallest sample eigenvalue is about
``0.086`` against a true value of 1: the optimiser sees a risk-free direction
that does not exist.

Three estimators are provided, in increasing order of how much structure they
assume:

:func:`ledoit_wolf`
    Shrink towards a scaled identity, with the intensity derived rather than
    tuned.  The optimal weight is available in closed form and needs no
    cross-validation.
:func:`constant_correlation_shrinkage`
    Shrink towards a target with the average pairwise correlation imposed on
    every pair -- a better target than the identity for equities, which are
    all positively correlated.
:func:`clip_eigenvalues`
    Replace every eigenvalue inside the Marchenko-Pastur bulk with their
    average, keeping the trace.  The sharpest statement of the idea: anything
    inside the bulk is indistinguishable from noise, so throw away its
    structure and keep only the eigenvalues that stick out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "sample_covariance",
    "ledoit_wolf",
    "constant_correlation_shrinkage",
    "marchenko_pastur_bounds",
    "marchenko_pastur_density",
    "clip_eigenvalues",
    "ShrinkageResult",
]


@dataclass(frozen=True)
class ShrinkageResult:
    """A shrunk covariance matrix and the intensity that produced it.

    ``intensity`` is worth reporting rather than hiding: it says how much the
    data was trusted.  Near zero means the sample matrix was already
    well-conditioned; near one means almost everything was discarded, which
    usually means there are not enough observations for the universe.
    """

    covariance: np.ndarray
    intensity: float
    target: np.ndarray

    @property
    def condition_number(self):
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return float(eigenvalues.max() / max(eigenvalues.min(), 1e-300))


def sample_covariance(returns, ddof=1):
    """Plain sample covariance.  Included to be beaten.

    ``ddof=1`` is the conventional unbiased normalisation.  Note that the
    estimators below use ``ddof=0`` by default instead: the Ledoit-Wolf
    intensity is derived for the maximum-likelihood estimator, and mixing the
    two leaves a ``T/(T-1)`` discrepancy that is small enough to go unnoticed
    and large enough to break an exact trace comparison.  Pass a matching
    ``ddof`` when comparing them directly.
    """
    data = np.asarray(returns, dtype=float)
    centred = data - data.mean(axis=0)
    return centred.T @ centred / (data.shape[0] - ddof)


def ledoit_wolf(returns, ddof=0):
    """Shrinkage towards a scaled identity, with the optimal intensity derived.

    The estimator is a convex combination ``d F + (1 - d) S`` of the sample
    covariance ``S`` and the target ``F = (tr S / N) I``.  Ledoit and Wolf show
    the intensity minimising expected squared Frobenius error is available
    without cross-validation:

    * ``d2`` is how far ``S`` is from the target -- the dispersion the target
      throws away;
    * ``b2`` is the estimation error in ``S`` itself, from the variance of the
      per-observation outer products around their mean;
    * the optimal intensity is ``b2 / d2``, capped at one.

    The interpretation is clean: shrink in proportion to how noisy the sample
    matrix is, relative to how much structure would be lost by replacing it.
    Both quantities are estimable from the same sample, which is why no tuning
    parameter is needed.
    """
    data = np.asarray(returns, dtype=float)
    n_obs, n_assets = data.shape
    if n_obs < 2:
        raise ValueError("need at least two observations")

    centred = data - data.mean(axis=0)
    sample = centred.T @ centred / (n_obs - ddof)

    mean_variance = np.trace(sample) / n_assets
    target = mean_variance * np.eye(n_assets)

    # d2: squared distance from the sample matrix to the target.
    dispersion = float(np.sum((sample - target) ** 2)) / n_assets

    # b2: the sampling error of S, as the mean squared deviation of the
    # per-observation outer products from their average.
    squared = np.einsum("ti,tj->ij", centred**2, centred**2) / n_obs
    noise = float(np.sum(squared - sample**2)) / (n_obs * n_assets)
    noise = min(noise, dispersion)

    intensity = 0.0 if dispersion <= 0 else float(np.clip(noise / dispersion, 0.0, 1.0))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return ShrinkageResult(covariance=shrunk, intensity=intensity, target=target)


def constant_correlation_shrinkage(returns, ddof=0):
    """Shrinkage towards a matrix with every correlation set to the average.

    A better target than the identity whenever the assets genuinely are
    correlated, which for equities they always are: the identity target throws
    away the single most important feature of an equity covariance matrix --
    that everything moves together -- while this one keeps it and discards only
    the differences between pairs.

    Variances are kept at their sample values; only the correlation structure
    is shrunk.
    """
    data = np.asarray(returns, dtype=float)
    n_obs, n_assets = data.shape
    centred = data - data.mean(axis=0)
    sample = centred.T @ centred / (n_obs - ddof)

    deviations = np.sqrt(np.clip(np.diag(sample), 1e-300, None))
    outer = np.outer(deviations, deviations)
    correlation = sample / outer
    off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]
    average = float(off_diagonal.mean()) if off_diagonal.size else 0.0

    target = average * outer
    np.fill_diagonal(target, np.diag(sample))

    dispersion = float(np.sum((sample - target) ** 2))
    squared = np.einsum("ti,tj->ij", centred**2, centred**2) / n_obs
    noise = float(np.sum(squared - sample**2)) / n_obs
    noise = min(noise, dispersion)

    intensity = 0.0 if dispersion <= 0 else float(np.clip(noise / dispersion, 0.0, 1.0))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return ShrinkageResult(covariance=shrunk, intensity=intensity, target=target)


def marchenko_pastur_bounds(n_assets, n_obs, variance=1.0):
    """Edges of the Marchenko-Pastur bulk, ``variance (1 +- sqrt q)^2``.

    Everything between them is what pure noise produces.  The gap is wider
    than intuition suggests: five years of daily data on a hundred assets
    gives ``q = 0.08``, and the bulk still spans ``0.51`` to ``1.63`` around a
    true eigenvalue of one.
    """
    q = n_assets / n_obs
    root = np.sqrt(q)
    return variance * (1.0 - root) ** 2, variance * (1.0 + root) ** 2


def marchenko_pastur_density(eigenvalues, n_assets, n_obs, variance=1.0):
    """The limiting spectral density of a white Wishart matrix.

    Used to check that the sample spectrum of genuinely uncorrelated returns
    follows the theory -- if it does not, the estimator is wrong before any
    portfolio is built on it.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    q = n_assets / n_obs
    lower, upper = marchenko_pastur_bounds(n_assets, n_obs, variance)
    inside = (eigenvalues > lower) & (eigenvalues < upper)
    density = np.zeros_like(eigenvalues)
    with np.errstate(invalid="ignore", divide="ignore"):
        density[inside] = np.sqrt(
            (upper - eigenvalues[inside]) * (eigenvalues[inside] - lower)
        ) / (2.0 * np.pi * variance * q * eigenvalues[inside])
    return density


def clip_eigenvalues(returns, keep_trace=True, ddof=0):
    """Replace every eigenvalue inside the Marchenko-Pastur bulk with their mean.

    The reasoning is the sharpest version of the shrinkage argument: an
    eigenvalue inside the bulk is statistically indistinguishable from what
    pure noise would produce, so its eigenvector carries no information worth
    keeping.  Eigenvalues above the upper edge are real structure -- the market
    factor, sectors -- and are left alone.

    The trace is preserved, so total variance is unchanged and only its
    *allocation across directions* is corrected -- without that the clipped
    matrix systematically understates risk.  Preserving it takes two steps,
    not one: rescaling the eigenvalues fixes the trace of the correlation
    matrix but breaks its unit diagonal, so the diagonal has to be
    renormalised afterwards or the covariance trace drifts by a tenth of a
    percent.
    """
    data = np.asarray(returns, dtype=float)
    n_obs, n_assets = data.shape
    centred = data - data.mean(axis=0)
    sample = centred.T @ centred / (n_obs - ddof)

    deviations = np.sqrt(np.clip(np.diag(sample), 1e-300, None))
    correlation = sample / np.outer(deviations, deviations)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    _, upper = marchenko_pastur_bounds(n_assets, n_obs, variance=1.0)

    noise = eigenvalues < upper
    cleaned = eigenvalues.copy()
    if np.any(noise):
        cleaned[noise] = eigenvalues[noise].mean()
    if keep_trace:
        cleaned *= eigenvalues.sum() / cleaned.sum()

    rebuilt = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)

    # Rescaling the eigenvalues preserves the trace of the correlation matrix
    # but not its unit diagonal, so the rebuilt matrix is no longer a
    # correlation matrix and the covariance trace drifts.  Renormalising the
    # diagonal back to one restores both: total variance comes out exactly
    # equal to the sample's, and only its allocation across directions has
    # changed, which is the point of the operation.
    diagonal = np.sqrt(np.clip(np.diag(rebuilt), 1e-300, None))
    rebuilt = rebuilt / np.outer(diagonal, diagonal)
    return rebuilt * np.outer(deviations, deviations)
