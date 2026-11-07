"""Covariance estimation, checked against random matrix theory."""

import numpy as np
import pytest

from portlab.covariance import (
    clip_eigenvalues,
    constant_correlation_shrinkage,
    ledoit_wolf,
    marchenko_pastur_bounds,
    marchenko_pastur_density,
    sample_covariance,
)


def test_sample_covariance_matches_numpy():
    data = np.random.default_rng(0).standard_normal((300, 8))
    assert np.allclose(sample_covariance(data, ddof=1), np.cov(data.T))
    assert np.allclose(sample_covariance(data, ddof=0), np.cov(data.T, ddof=0))


def test_marchenko_pastur_bounds_are_the_classical_edges():
    lower, upper = marchenko_pastur_bounds(100, 400)
    assert lower == pytest.approx((1 - 0.5) ** 2)
    assert upper == pytest.approx((1 + 0.5) ** 2)
    # q -> 0 collapses the bulk onto the true eigenvalue.
    tight = marchenko_pastur_bounds(10, 1_000_000)
    assert tight[0] == pytest.approx(1.0, abs=0.01)
    assert tight[1] == pytest.approx(1.0, abs=0.01)


def test_empirical_spectrum_follows_the_marchenko_pastur_law():
    """The theory the whole module rests on, checked directly.

    Pooled over 40 draws to suppress sampling noise, and restricted to the
    interior of the bulk because the density has inverse-square-root
    singularities at both edges that no histogram resolves.
    """
    n_assets, n_obs = 200, 800
    lower, upper = marchenko_pastur_bounds(n_assets, n_obs)
    inner = (lower + 0.08 * (upper - lower), upper - 0.08 * (upper - lower))

    eigenvalues = np.concatenate([
        np.linalg.eigvalsh(
            sample_covariance(
                np.random.default_rng(seed).standard_normal((n_obs, n_assets)),
                ddof=0,
            )
        )
        for seed in range(40)
    ])
    counts, edges = np.histogram(eigenvalues, bins=30, range=inner)
    empirical = counts / (eigenvalues.size * (edges[1] - edges[0]))
    theory = marchenko_pastur_density(
        0.5 * (edges[1:] + edges[:-1]), n_assets, n_obs
    )
    assert np.mean(np.abs(empirical / theory - 1.0)) < 0.05


def test_sample_eigenvalues_spread_out_even_when_the_truth_is_identity():
    """Why an optimiser cannot be handed a sample covariance matrix.

    Every true eigenvalue is one, yet the smallest sample eigenvalue at
    ``q = 0.5`` is near 0.09 -- and the optimiser's largest bets go precisely
    on the smallest eigenvalues.
    """
    data = np.random.default_rng(1).standard_normal((400, 200))
    eigenvalues = np.linalg.eigvalsh(sample_covariance(data, ddof=0))
    lower, upper = marchenko_pastur_bounds(200, 400)
    # The extreme eigenvalues converge to the edges, but slowly -- their
    # fluctuations are Tracy-Widom and vanish only as N^(-2/3) -- so the
    # tolerance here is finite-sample, not sloppiness.
    assert eigenvalues.min() == pytest.approx(lower, abs=0.06)
    assert eigenvalues.max() == pytest.approx(upper, abs=0.2)
    assert eigenvalues.min() < 0.12


def test_ledoit_wolf_beats_the_sample_matrix_when_the_truth_is_identity():
    data = np.random.default_rng(2).standard_normal((400, 150))
    truth = np.eye(150)
    shrunk = ledoit_wolf(data)
    assert np.linalg.norm(shrunk.covariance - truth) < 0.2 * np.linalg.norm(
        sample_covariance(data, ddof=0) - truth
    )
    assert 0.0 < shrunk.intensity <= 1.0
    assert shrunk.condition_number < np.linalg.cond(sample_covariance(data, ddof=0))


def test_shrinkage_intensity_falls_as_observations_accumulate():
    """More data means less shrinkage -- but only when there is structure.

    Worth being careful about: on white noise the identity target *is* the
    truth, so the optimal intensity is one at every sample size and never
    declines.  The decline only appears when the target is wrong, which is why
    this uses a factor market rather than independent normals.
    """
    from portlab.market import factor_market

    intensities = [
        ledoit_wolf(factor_market(n_assets=40, n_obs=n_obs, seed=3).returns).intensity
        for n_obs in (60, 250, 1000, 5000)
    ]
    assert intensities == sorted(intensities, reverse=True)
    assert intensities[0] > 3 * intensities[-1]


def test_shrinkage_preserves_the_trace():
    """Total variance is untouched; only its allocation across directions."""
    data = np.random.default_rng(4).standard_normal((300, 50)) * np.linspace(1, 3, 50)
    reference = np.trace(sample_covariance(data, ddof=0))
    assert np.trace(ledoit_wolf(data).covariance) == pytest.approx(reference, rel=1e-12)
    assert np.trace(clip_eigenvalues(data)) == pytest.approx(reference, rel=1e-12)


def test_estimators_stay_positive_definite_when_the_sample_is_singular():
    """T < N makes the sample matrix singular; the estimators must not be."""
    data = np.random.default_rng(5).standard_normal((40, 80))
    assert np.linalg.eigvalsh(sample_covariance(data, ddof=0)).min() < 1e-10
    for estimate in (
        ledoit_wolf(data).covariance,
        constant_correlation_shrinkage(data).covariance,
    ):
        assert np.linalg.eigvalsh(estimate).min() > 1e-10


def test_constant_correlation_target_keeps_the_sample_variances():
    data = np.random.default_rng(6).standard_normal((400, 20)) * np.linspace(1, 4, 20)
    result = constant_correlation_shrinkage(data)
    assert np.allclose(
        np.diag(result.target), np.diag(sample_covariance(data, ddof=0))
    )


def test_clipping_keeps_eigenvalues_above_the_edge_and_flattens_the_rest():
    """The structure survives; the noise is replaced by its average."""
    from portlab.market import factor_market

    market = factor_market(n_assets=120, n_obs=600, n_factors=3, seed=7)
    cleaned = clip_eigenvalues(market.returns)

    deviations = np.sqrt(np.diag(cleaned))
    correlation = cleaned / np.outer(deviations, deviations)
    eigenvalues = np.sort(np.linalg.eigvalsh(correlation))[::-1]

    # Three factors survive as large eigenvalues.
    assert eigenvalues[2] > 2.0

    # The bulk is compressed but not to a point.  Flattening the eigenvalues
    # breaks the unit diagonal, and renormalising it back -- which is what
    # keeps the covariance trace exact -- reintroduces some dispersion.  The
    # test is therefore that the bulk is much tighter than the sample's, not
    # that it is degenerate.
    sample_bulk = np.sort(np.linalg.eigvalsh(
        np.corrcoef(market.returns.T)
    ))[::-1][3:]
    bulk = eigenvalues[3:]
    assert bulk.std() < 0.35 * sample_bulk.std()
    assert np.linalg.eigvalsh(cleaned).min() > 0


def test_rejects_too_few_observations():
    with pytest.raises(ValueError):
        ledoit_wolf(np.zeros((1, 5)))
