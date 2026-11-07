"""VaR and ES: closed forms where they exist, coverage where they do not."""

import numpy as np
import pytest
from scipy.stats import norm

from portlab.risk import (
    component_var,
    cornish_fisher_var,
    evt_var,
    filtered_historical_var,
    fit_garch,
    gaussian_var,
    historical_var,
)


def test_gaussian_var_matches_the_closed_form():
    returns = np.random.default_rng(0).standard_normal(400_000) * 0.01
    measure = gaussian_var(returns, 0.99)
    assert measure.var == pytest.approx(norm.ppf(0.99) * 0.01, rel=0.01)
    # ES of a standard normal below its 1% quantile: phi(q) / 0.01.
    expected = 0.01 * norm.pdf(norm.ppf(0.01)) / 0.01
    assert measure.expected_shortfall == pytest.approx(expected, rel=0.01)
    assert "VaR" in str(measure)


def test_expected_shortfall_always_exceeds_var():
    rng = np.random.default_rng(1)
    returns = rng.standard_t(6, 20_000) * 0.01
    for measure in (
        gaussian_var(returns),
        historical_var(returns),
        evt_var(returns),
        cornish_fisher_var(returns),
    ):
        assert measure.expected_shortfall > measure.var


def test_var_increases_with_confidence():
    returns = np.random.default_rng(2).standard_normal(50_000) * 0.01
    for method in (gaussian_var, historical_var, evt_var):
        levels = [method(returns, c).var for c in (0.90, 0.95, 0.99, 0.995)]
        assert levels == sorted(levels)


def test_historical_var_is_the_empirical_quantile():
    returns = np.random.default_rng(3).standard_normal(10_000) * 0.02
    assert historical_var(returns, 0.95).var == pytest.approx(
        -np.quantile(returns, 0.05), rel=1e-12
    )


def test_gaussian_var_understates_a_fat_tail():
    """The failure that matters, in the direction that matters."""
    returns = np.random.default_rng(4).standard_t(4, 200_000) * 0.01 / np.sqrt(2)
    assert gaussian_var(returns, 0.99).var < 0.9 * historical_var(returns, 0.99).var
    assert gaussian_var(returns, 0.999).var < 0.7 * historical_var(returns, 0.999).var


def test_extreme_value_tail_tracks_the_empirical_one_far_out():
    """Where EVT earns its place: past where a quantile can be counted."""
    returns = np.random.default_rng(5).standard_t(4, 200_000) * 0.01 / np.sqrt(2)
    for confidence in (0.99, 0.995, 0.999):
        assert evt_var(returns, confidence).var == pytest.approx(
            historical_var(returns, confidence).var, rel=0.12
        )


def test_cornish_fisher_blows_up_on_infinite_kurtosis():
    """Documented limitation, pinned so it cannot be forgotten.

    Student-t with four degrees of freedom has no finite fourth moment, so an
    expansion in sample moments has nothing to converge to.  It does not fail
    quietly -- it returns a number more than twice the empirical quantile.
    """
    returns = np.random.default_rng(6).standard_t(4, 200_000) * 0.01 / np.sqrt(2)
    assert cornish_fisher_var(returns, 0.99).var > 2.0 * historical_var(
        returns, 0.99
    ).var


def test_evt_is_stable_across_reasonable_thresholds():
    """A tail estimate that moves a lot with the threshold is not usable."""
    returns = np.random.default_rng(7).standard_t(5, 100_000) * 0.01
    estimates = [
        evt_var(returns, 0.995, threshold_quantile=q).var
        for q in (0.85, 0.90, 0.95, 0.97)
    ]
    assert max(estimates) / min(estimates) < 1.1


def test_evt_needs_enough_exceedances():
    with pytest.raises(ValueError):
        evt_var(np.random.default_rng(8).standard_normal(50), threshold_quantile=0.99)


@pytest.mark.slow
def test_garch_recovers_its_own_parameters():
    omega, alpha, beta = 2e-6, 0.09, 0.89
    rng = np.random.default_rng(9)
    n = 8000
    residuals = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        variance[t] = omega + alpha * residuals[t - 1] ** 2 + beta * variance[t - 1]
        residuals[t] = np.sqrt(variance[t]) * rng.standard_normal()

    fit = fit_garch(residuals)
    assert fit.alpha == pytest.approx(alpha, abs=0.03)
    assert fit.beta == pytest.approx(beta, abs=0.03)
    assert fit.persistence == pytest.approx(alpha + beta, abs=0.015)
    assert np.sqrt(fit.unconditional_variance) == pytest.approx(
        np.sqrt(omega / (1 - alpha - beta)), rel=0.1
    )


def test_forecasts_revert_to_the_unconditional_level_on_clean_data():
    omega, alpha, beta = 2e-6, 0.09, 0.85
    rng = np.random.default_rng(14)
    n = 4000
    residuals = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        variance[t] = omega + alpha * residuals[t - 1] ** 2 + beta * variance[t - 1]
        residuals[t] = np.sqrt(variance[t]) * rng.standard_normal()

    fit = fit_garch(residuals)
    assert fit.persistence < 0.99
    horizon = int(5 * np.log(0.5) / np.log(fit.persistence))
    assert fit.forecast(horizon)[-1] == pytest.approx(
        np.sqrt(fit.unconditional_variance), rel=0.02
    )


def test_a_volatility_level_shift_inflates_persistence_to_the_boundary():
    """Lamoureux-Lastrapes, pinned because it changes how the fit is read.

    A single GARCH process cannot express a permanent change in volatility
    level, so the only way it can fit one is by making shocks nearly
    permanent.  The estimate goes to the stationarity cap and the implied
    unconditional volatility becomes nonsense: 86% annualised here, and 122%
    with the switch earlier in the sample, against a truth of 12% rising to
    38%.  How far it inflates depends on how much of the sample sits in each
    regime, so the assertion below is a factor rather than a level.

    The damage is confined to long-horizon quantities.  The one-step forecast,
    which is what a daily VaR uses, stays close to the realised volatility of
    the days that follow.
    """
    from portlab.market import regime_market

    returns, truth = regime_market(n_obs=2000, switch_at=1200, seed=10)
    fit = fit_garch(returns)

    assert fit.persistence > 0.995
    assert np.sqrt(fit.unconditional_variance) > 2.0 * truth.max()

    # ... and yet the one-step forecast is still right.
    forecast = float(fit.forecast(1)[0])
    assert forecast == pytest.approx(fit.volatility[-1], rel=0.3)
    assert 0.4 < forecast / truth[-1] < 2.5


def test_garch_tracks_a_volatility_regime_change():
    from portlab.market import regime_market

    returns, truth = regime_market(n_obs=2000, switch_at=1200, seed=11)
    fit = fit_garch(returns)
    assert np.corrcoef(fit.volatility[50:], truth[50:])[0, 1] > 0.7
    assert fit.volatility[1500:].mean() > 2.0 * fit.volatility[:1100].mean()


def test_filtered_historical_var_responds_to_the_current_regime():
    """The property that makes it pass the backtests the others fail."""
    from portlab.market import regime_market

    returns, _ = regime_market(n_obs=2000, switch_at=1000, seed=12)
    quiet = filtered_historical_var(returns[:900], 0.99).var
    stressed = filtered_historical_var(returns[1100:1900], 0.99).var
    assert stressed > 2.0 * quiet

    # A plain historical VaR on the same windows moves far less, because it
    # is an unweighted quantile of the whole window.
    quiet_h = historical_var(returns[:900], 0.99).var
    stressed_h = historical_var(returns[1100:1900], 0.99).var
    assert stressed / quiet > 0.9 * (stressed_h / quiet_h)


def test_component_var_sums_to_portfolio_var():
    """Euler again: the parts have to add up to the whole."""
    rng = np.random.default_rng(13)
    factor = rng.standard_normal((9, 9))
    cov = factor @ factor.T / 9 + np.eye(9) * 0.1
    weights = rng.dirichlet(np.ones(9))
    total = norm.ppf(0.99) * np.sqrt(weights @ cov @ weights)
    assert component_var(weights, cov, 0.99).sum() == pytest.approx(total, rel=1e-12)


def test_component_var_can_be_negative_for_a_hedge():
    """A position that reduces portfolio risk should show a negative share."""
    cov = np.array([[0.04, -0.035], [-0.035, 0.04]])
    contributions = component_var(np.array([0.9, 0.1]), cov, 0.99)
    assert contributions[1] < 0.0


def test_fit_garch_rejects_constant_input():
    with pytest.raises(ValueError):
        fit_garch(np.ones(500))
