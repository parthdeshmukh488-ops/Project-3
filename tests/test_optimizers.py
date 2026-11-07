"""Allocators, checked against closed forms and their own defining property."""

import numpy as np
import pytest

from portlab.market import factor_market
from portlab.optimizers import (
    black_litterman,
    diversification_ratio,
    equal_weight,
    hierarchical_risk_parity,
    max_diversification,
    min_variance,
    risk_contributions,
    risk_parity,
    simplex_projection,
)


def random_covariance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n, n))
    return factor @ factor.T / n + np.eye(n) * 0.2


ALL_LONG_ONLY = (
    lambda cov: min_variance(cov, long_only=True),
    risk_parity,
    hierarchical_risk_parity,
    lambda cov: max_diversification(cov, long_only=True),
)


def test_simplex_projection_is_exact():
    """Sums to one, non-negative, and is the nearest such point.

    Checked against a brute-force projection on random directions: the
    objective is strictly convex, so the minimiser is unique and any
    disagreement is an error rather than a tie.
    """
    rng = np.random.default_rng(0)
    for _ in range(40):
        v = rng.standard_normal(15) * 3.0
        projected = simplex_projection(v)
        assert projected.sum() == pytest.approx(1.0, abs=1e-12)
        assert projected.min() >= 0.0
        # No feasible point is closer.
        for _ in range(200):
            other = rng.dirichlet(np.ones(15))
            assert np.sum((v - projected) ** 2) <= np.sum((v - other) ** 2) + 1e-12


def test_simplex_projection_leaves_feasible_points_alone():
    point = np.array([0.2, 0.3, 0.5])
    assert np.allclose(simplex_projection(point), point)


def test_unconstrained_min_variance_matches_the_closed_form():
    cov = random_covariance(10, seed=1)
    weights = min_variance(cov)
    inverse = np.linalg.inv(cov) @ np.ones(10)
    assert np.allclose(weights, inverse / inverse.sum(), rtol=1e-10)


def test_min_variance_actually_minimises_variance():
    """No random feasible portfolio beats it."""
    cov = random_covariance(12, seed=2)
    rng = np.random.default_rng(3)
    best = min_variance(cov)
    best_variance = float(best @ cov @ best)
    for _ in range(2000):
        other = rng.dirichlet(np.ones(12))
        assert best_variance <= other @ cov @ other + 1e-12


def test_long_only_min_variance_is_feasible_and_no_better_than_unconstrained():
    cov = random_covariance(14, seed=4)
    constrained = min_variance(cov, long_only=True)
    unconstrained = min_variance(cov)
    assert constrained.min() >= -1e-12
    assert constrained.sum() == pytest.approx(1.0, abs=1e-10)
    assert constrained @ cov @ constrained >= unconstrained @ cov @ unconstrained - 1e-12


def test_long_only_min_variance_matches_unconstrained_when_that_is_positive():
    """Where the constraint does not bind, the two must coincide.

    Constructed with a diagonal covariance, whose unconstrained
    minimum-variance solution is inverse-variance weighted and therefore
    already non-negative.
    """
    cov = np.diag(np.linspace(0.01, 0.2, 10))
    assert np.allclose(
        min_variance(cov, long_only=True), min_variance(cov), atol=1e-8
    )


def test_risk_contributions_sum_to_portfolio_volatility():
    """Euler's theorem, which is what makes the decomposition meaningful."""
    cov = random_covariance(10, seed=5)
    weights = np.random.default_rng(6).dirichlet(np.ones(10))
    contributions = risk_contributions(weights, cov)
    assert contributions.sum() == pytest.approx(
        np.sqrt(weights @ cov @ weights), rel=1e-12
    )


def test_risk_parity_equalises_risk_contributions():
    for seed in range(5):
        cov = random_covariance(15, seed=seed)
        weights = risk_parity(cov)
        contributions = risk_contributions(weights, cov)
        assert contributions.max() - contributions.min() < 1e-10
        assert weights.min() > 0.0
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_risk_parity_respects_unequal_budgets():
    cov = random_covariance(12, seed=7)
    budgets = np.linspace(1.0, 4.0, 12)
    budgets /= budgets.sum()
    weights = risk_parity(cov, budgets=budgets)
    shares = risk_contributions(weights, cov)
    assert np.allclose(shares / shares.sum(), budgets, atol=1e-10)


def test_risk_parity_on_a_diagonal_covariance_is_inverse_volatility():
    """The one case with a closed form, so it can be checked exactly."""
    variances = np.array([0.04, 0.01, 0.09, 0.16])
    weights = risk_parity(np.diag(variances))
    expected = (1.0 / np.sqrt(variances)) / np.sum(1.0 / np.sqrt(variances))
    assert np.allclose(weights, expected, rtol=1e-10)


@pytest.mark.parametrize("allocator", ALL_LONG_ONLY)
def test_long_only_allocators_produce_valid_portfolios(allocator):
    for seed in range(4):
        weights = allocator(random_covariance(16, seed=seed))
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert weights.min() >= -1e-12


def test_hierarchical_risk_parity_is_more_diversified_than_min_variance():
    """Its selling point: it does not concentrate the way an optimiser does."""
    market = factor_market(n_assets=60, n_obs=300, seed=8)
    cov = np.cov(market.returns.T)
    hrp = hierarchical_risk_parity(cov)
    mv = min_variance(cov, long_only=True)
    assert 1.0 / np.sum(hrp**2) > 3.0 * (1.0 / np.sum(mv**2))


def test_max_diversification_maximises_the_diversification_ratio():
    cov = random_covariance(12, seed=9)
    best = max_diversification(cov)
    ratio = diversification_ratio(best, cov)
    assert ratio > diversification_ratio(equal_weight(12), cov)
    assert ratio > diversification_ratio(min_variance(cov, long_only=True), cov)
    rng = np.random.default_rng(10)
    for _ in range(2000):
        other = rng.dirichlet(np.ones(12))
        assert ratio >= diversification_ratio(other, cov) - 1e-9


def test_black_litterman_returns_the_prior_when_the_view_agrees_with_it():
    """A view that says nothing new must move nothing."""
    cov = random_covariance(8, seed=11)
    market = equal_weight(8)
    prior = 2.5 * cov @ market
    picking = np.zeros((1, 8))
    picking[0, 2] = 1.0
    posterior = black_litterman(cov, market, picking, [prior[2]])
    assert np.allclose(posterior, prior, atol=1e-12)


def test_black_litterman_tilts_towards_a_view_and_scales_with_confidence():
    cov = random_covariance(8, seed=12)
    market = equal_weight(8)
    prior = 2.5 * cov @ market

    picking = np.zeros((1, 8))
    picking[0, 3] = 1.0
    bullish = prior[3] + 0.05

    vague = black_litterman(cov, market, picking, [bullish],
                            view_uncertainty=[[1.0]])
    confident = black_litterman(cov, market, picking, [bullish],
                                view_uncertainty=[[1e-10]])

    assert prior[3] < vague[3] < confident[3]
    assert confident[3] == pytest.approx(bullish, rel=1e-4)
