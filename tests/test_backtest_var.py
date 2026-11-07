"""VaR backtests: do they fire when they should, and only then?"""

import numpy as np
import pytest

from portlab.backtest_var import (
    breaches,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    traffic_light,
)


def test_breaches_use_the_loss_sign_convention():
    """VaR is a positive loss magnitude; a breach is a return below its negative.

    The most common source of a silently inverted backtest, so it is checked
    directly rather than only through the tests that depend on it.
    """
    returns = np.array([0.01, -0.02, -0.005, -0.03])
    forecasts = np.array([0.025, 0.025, 0.025, 0.025])
    assert list(breaches(returns, forecasts)) == [False, False, False, True]


def test_breaches_requires_matching_lengths():
    with pytest.raises(ValueError):
        breaches(np.zeros(10), np.zeros(9))


def test_a_correctly_specified_model_is_not_rejected():
    """Across many samples the rejection rate must be about the nominal size."""
    rejections = 0
    trials = 200
    for seed in range(trials):
        indicator = np.random.default_rng(seed).random(2000) < 0.01
        rejections += christoffersen_conditional_coverage(indicator).rejected_at_5pct
    assert rejections / trials < 0.10


def test_too_many_breaches_are_caught_by_coverage():
    indicator = np.random.default_rng(0).random(2000) < 0.04
    result = kupiec_pof(indicator, 0.99)
    assert result.rejected_at_5pct
    assert result.n_breaches > 3 * result.n_expected
    assert traffic_light(indicator, 0.99)[0] == "red"


def test_too_few_breaches_are_also_caught():
    """An over-conservative model is a real failure, not a safe one."""
    indicator = np.zeros(2000, dtype=bool)
    assert kupiec_pof(indicator, 0.99).rejected_at_5pct


def test_clustered_breaches_pass_coverage_and_fail_independence():
    """The case that justifies running both tests.

    Twenty consecutive breaches in two thousand days is exactly the expected
    count, so the coverage test sees nothing at all -- its statistic depends
    only on the total, and any reordering gives the same answer.  The
    independence test rejects overwhelmingly.
    """
    indicator = np.zeros(2000, dtype=bool)
    indicator[500:520] = True

    coverage = kupiec_pof(indicator, 0.99)
    independence = christoffersen_independence(indicator, 0.99)
    combined = christoffersen_conditional_coverage(indicator, 0.99)

    assert coverage.n_breaches == int(coverage.n_expected)
    assert coverage.p_value > 0.99
    assert not coverage.rejected_at_5pct
    assert independence.statistic > 100.0
    assert independence.rejected_at_5pct
    assert combined.rejected_at_5pct


def test_the_combined_statistic_is_the_sum_of_its_parts():
    """The restrictions are independent, so the statistics add."""
    indicator = np.random.default_rng(1).random(3000) < 0.015
    coverage = kupiec_pof(indicator, 0.99)
    independence = christoffersen_independence(indicator, 0.99)
    combined = christoffersen_conditional_coverage(indicator, 0.99)
    assert combined.statistic == pytest.approx(
        coverage.statistic + independence.statistic, rel=1e-12
    )


def test_likelihood_ratios_are_never_negative():
    """Rounding at the exact null used to make them so, and p-values above one."""
    indicator = np.zeros(2000, dtype=bool)
    indicator[::100] = True  # exactly 1%
    for result in (
        kupiec_pof(indicator, 0.99),
        christoffersen_independence(indicator, 0.99),
        christoffersen_conditional_coverage(indicator, 0.99),
    ):
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0


def test_independence_is_untestable_without_both_transition_types():
    """No breaches at all means no evidence either way, and it says so."""
    result = christoffersen_independence(np.zeros(500, dtype=bool), 0.99)
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_traffic_light_zones_in_order():
    rng = np.random.default_rng(2)
    green = rng.random(250) < 0.01
    amber = np.zeros(250, dtype=bool)
    amber[:7] = True
    red = np.zeros(250, dtype=bool)
    red[:15] = True
    assert traffic_light(green, 0.99)[0] == "green"
    assert traffic_light(amber, 0.99)[0] == "amber"
    assert traffic_light(red, 0.99)[0] == "red"


@pytest.mark.slow
def test_filtered_historical_beats_plain_historical_through_a_regime_change():
    """The end-to-end claim of the package, in one test.

    A rolling historical VaR does adapt -- but over the length of its window,
    not the length of the shock, so it breaches far too often while the regime
    is changing.  The GARCH filter responds within days.
    """
    from portlab.market import regime_market
    from portlab.risk import filtered_historical_var, historical_var

    returns, _ = regime_market(n_obs=2000, switch_at=1400, seed=11)
    window, confidence, step = 500, 0.99, 5

    rates = {}
    for name, method in (
        ("historical", historical_var),
        ("filtered", filtered_historical_var),
    ):
        forecasts = np.full(returns.size, np.nan)
        last = np.nan
        for t in range(window, returns.size):
            if (t - window) % step == 0:
                last = method(returns[t - window : t], confidence).var
            forecasts[t] = last
        scored = slice(window, returns.size)
        indicator = breaches(returns[scored], forecasts[scored])
        rates[name] = indicator[1400 - window :].mean()

    assert rates["historical"] > 0.025, "expected the static method to breach"
    assert rates["filtered"] < rates["historical"]
