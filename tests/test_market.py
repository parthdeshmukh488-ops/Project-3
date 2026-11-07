"""The generators, which only earn their place if their truth is true."""

import numpy as np
import pytest

from portlab.covariance import marchenko_pastur_bounds, sample_covariance
from portlab.market import factor_market, regime_market


def test_realised_covariance_converges_to_the_stated_one():
    """The whole point of the generator: the truth is exactly known."""
    market = factor_market(n_assets=20, n_obs=400_000, seed=0)
    realised = sample_covariance(market.returns, ddof=0)
    assert np.allclose(realised, market.covariance, rtol=0.05, atol=1e-8)


def test_true_variance_is_exact_rather_than_estimated():
    market = factor_market(n_assets=15, n_obs=200, seed=1)
    weights = np.random.default_rng(2).dirichlet(np.ones(15))
    assert market.true_variance(weights) == pytest.approx(
        float(weights @ market.covariance @ weights), rel=1e-15
    )
    assert market.true_volatility(weights) == pytest.approx(
        np.sqrt(market.true_variance(weights) * 252), rel=1e-15
    )


def test_the_factor_structure_shows_up_as_eigenvalues_above_the_noise_bulk():
    """Exactly as many large eigenvalues as there are factors.

    This is what makes the market a fair test of
    :func:`portlab.covariance.clip_eigenvalues`: the number of eigenvalues
    that should survive clipping is known in advance.
    """
    n_assets, n_obs, n_factors = 100, 500, 3
    market = factor_market(
        n_assets=n_assets, n_obs=n_obs, n_factors=n_factors, seed=3
    )
    _, upper = marchenko_pastur_bounds(n_assets, n_obs)

    correlation = np.corrcoef(market.returns.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(correlation))[::-1]
    assert int(np.sum(eigenvalues > upper)) == n_factors
    assert eigenvalues[0] > 5 * eigenvalues[1]  # the market factor dominates


def test_assets_are_positively_correlated_as_equities_are():
    market = factor_market(n_assets=50, n_obs=5000, seed=4)
    correlation = np.corrcoef(market.returns.T)
    off_diagonal = correlation[~np.eye(50, dtype=bool)]
    assert 0.1 < off_diagonal.mean() < 0.6
    assert (off_diagonal > 0).mean() > 0.9


def test_stated_volatilities_are_annualised():
    market = factor_market(
        n_assets=40, n_obs=200_000, market_vol=0.16, idiosyncratic_vol=0.25, seed=5
    )
    annual = market.returns.std(axis=0) * np.sqrt(252)
    assert 0.15 < annual.mean() < 0.6


def test_covariance_is_positive_definite():
    market = factor_market(n_assets=80, n_obs=100, seed=6)
    assert np.linalg.eigvalsh(market.covariance).min() > 0


def test_regime_market_changes_volatility_where_it_says_it_does():
    returns, volatility = regime_market(
        n_obs=4000, switch_at=2000, quiet_vol=0.12, stressed_vol=0.38, seed=7
    )
    quiet = returns[:1800].std() * np.sqrt(252)
    stressed = returns[2400:].std() * np.sqrt(252)
    assert quiet == pytest.approx(0.12, rel=0.35)
    assert stressed == pytest.approx(0.38, rel=0.35)
    assert stressed > 2.0 * quiet


def test_regime_market_has_fat_tails_within_a_regime():
    """Student-t innovations, so a Gaussian VaR is wrong even without a switch."""
    returns, volatility = regime_market(n_obs=200_000, switch_at=200_000, seed=8)
    standardised = returns / volatility
    excess_kurtosis = float(np.mean(standardised**4)) - 3.0
    assert excess_kurtosis > 2.0  # t(5) theory is 6, estimated noisily


def test_regime_market_volatility_is_autocorrelated():
    """Needed for breaches to have the chance to cluster at all."""
    _, volatility = regime_market(n_obs=3000, seed=9)
    assert np.corrcoef(volatility[:-1], volatility[1:])[0, 1] > 0.95
