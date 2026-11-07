"""Backtesting a VaR model: did it break the number of times it promised?

A 99% VaR model says losses should exceed it on 1% of days.  That is a
falsifiable claim, and testing it is the only thing that separates a risk
number from a decoration.  Two things can go wrong, and they need separate
tests because a model can pass one and fail the other badly.

**Unconditional coverage** -- the right *number* of breaches.
:func:`kupiec_pof` is a likelihood ratio test of exactly that.  Its weakness
is that it is blind to timing: a model that breaches ten times in two weeks
and never again passes as comfortably as one that breaches evenly.

**Independence** -- breaches not clustered.  :func:`christoffersen_independence`
tests whether a breach today predicts a breach tomorrow, by fitting a two-state
Markov chain to the breach sequence and comparing it against the
no-dependence restriction.  Clustering is the signature of a model that does
not react to volatility, and it is the dangerous failure: ten breaches spread
over four years is a slightly miscalibrated model, while ten in a fortnight is
a firm discovering its risk system does not work during exactly the week it
needed to.

:func:`christoffersen_conditional_coverage` combines them.  The sum of the two
statistics is chi-squared with two degrees of freedom, since the restrictions
are independent.

:func:`traffic_light` reports the Basel zones, which is how a regulator reads
the same breach count -- a cruder test, included because it is the one that
has consequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

__all__ = [
    "BacktestResult",
    "breaches",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "kupiec_pof",
    "traffic_light",
]


@dataclass(frozen=True)
class BacktestResult:
    """A likelihood-ratio test of a VaR model."""

    statistic: float
    p_value: float
    n_breaches: int
    n_expected: float
    n_obs: int
    test: str

    @property
    def rejected_at_5pct(self):
        return self.p_value < 0.05

    def __str__(self):
        verdict = "REJECTED" if self.rejected_at_5pct else "not rejected"
        return (
            f"{self.test}: {self.n_breaches} breaches vs {self.n_expected:.1f} "
            f"expected, LR = {self.statistic:.2f}, p = {self.p_value:.4f} "
            f"({verdict})"
        )


def breaches(returns, var_forecasts):
    """Boolean series: did the loss exceed the forecast?

    ``var_forecasts`` is a positive loss magnitude, as the functions in
    :mod:`portlab.risk` return it, so a breach is ``return < -var``.  The sign
    convention is the most common source of a silently wrong backtest, which
    is why the conversion lives in one place.
    """
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(var_forecasts, dtype=float).ravel()
    if r.shape != v.shape:
        raise ValueError("returns and forecasts must have the same length")
    return r < -v


def kupiec_pof(breach_series, confidence=0.99):
    """Kupiec's proportion-of-failures test of unconditional coverage.

    Under the null the breaches are Bernoulli with probability ``p = 1 - c``,
    so the likelihood ratio against the unrestricted Bernoulli with the
    observed rate is chi-squared with one degree of freedom.  Note what it
    cannot see: the statistic depends on the breach *count* alone, so any
    reordering of the same breaches gives exactly the same answer.
    """
    indicator = np.asarray(breach_series, dtype=bool).ravel()
    n = indicator.size
    failures = int(indicator.sum())
    p = 1.0 - confidence

    if failures == 0:
        statistic = -2.0 * n * np.log(1.0 - p)
    elif failures == n:
        statistic = -2.0 * n * np.log(p)
    else:
        rate = failures / n
        statistic = -2.0 * (
            (n - failures) * np.log(1.0 - p)
            + failures * np.log(p)
            - (n - failures) * np.log(1.0 - rate)
            - failures * np.log(rate)
        )
    # A likelihood ratio cannot be negative; rounding at the exact null makes
    # it so, and a small negative value turns into a p-value above one.
    statistic = max(float(statistic), 0.0)
    return BacktestResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, df=1)),
        n_breaches=failures,
        n_expected=float(n * p),
        n_obs=int(n),
        test="Kupiec POF",
    )


def _transition_counts(indicator):
    previous = indicator[:-1]
    current = indicator[1:]
    return (
        int(np.sum(~previous & ~current)),
        int(np.sum(~previous & current)),
        int(np.sum(previous & ~current)),
        int(np.sum(previous & current)),
    )


def christoffersen_independence(breach_series, confidence=0.99):
    """Test whether breaches cluster, via a two-state Markov chain.

    Fits transition probabilities ``P(breach | no breach yesterday)`` and
    ``P(breach | breach yesterday)`` and tests the restriction that they are
    equal.  Rejection means today's breach predicts tomorrow's -- the
    signature of a model whose VaR does not move with volatility, so that a
    turbulent fortnight produces a run of breaches rather than the occasional
    isolated one.
    """
    indicator = np.asarray(breach_series, dtype=bool).ravel()
    n00, n01, n10, n11 = _transition_counts(indicator)

    denominator_0 = n00 + n01
    denominator_1 = n10 + n11
    total = denominator_0 + denominator_1
    if denominator_0 == 0 or denominator_1 == 0 or total == 0:
        # No transitions of one kind: the restriction is untestable rather
        # than satisfied, so report a statistic of zero and a p-value of one
        # instead of inventing evidence either way.
        return BacktestResult(0.0, 1.0, int(indicator.sum()),
                              float(indicator.size * (1 - confidence)),
                              int(indicator.size), "Christoffersen independence")

    pi01 = n01 / denominator_0
    pi11 = n11 / denominator_1
    pi = (n01 + n11) / total

    def xlogy(count, probability):
        return 0.0 if count == 0 else count * np.log(probability)

    restricted = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    unrestricted = (
        xlogy(n00, 1.0 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1.0 - pi11)
        + xlogy(n11, pi11)
    )
    statistic = -2.0 * (restricted - unrestricted)
    statistic = max(float(statistic), 0.0)
    return BacktestResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, df=1)),
        n_breaches=int(indicator.sum()),
        n_expected=float(indicator.size * (1.0 - confidence)),
        n_obs=int(indicator.size),
        test="Christoffersen independence",
    )


def christoffersen_conditional_coverage(breach_series, confidence=0.99):
    """Coverage and independence together, chi-squared with two degrees.

    The two restrictions are independent, so the statistics add.  This is the
    test to quote: a model has to get both the number of breaches and their
    timing right, and passing one while failing the other is common enough
    that reporting only the first is misleading.
    """
    coverage = kupiec_pof(breach_series, confidence)
    independence = christoffersen_independence(breach_series, confidence)
    statistic = coverage.statistic + independence.statistic
    return BacktestResult(
        statistic=float(statistic),
        p_value=float(chi2.sf(statistic, df=2)),
        n_breaches=coverage.n_breaches,
        n_expected=coverage.n_expected,
        n_obs=coverage.n_obs,
        test="Christoffersen conditional coverage",
    )


def traffic_light(breach_series, confidence=0.99):
    """Basel traffic-light zones from the breach count.

    Green, amber or red according to the cumulative binomial probability of
    seeing at most this many breaches under the null -- the regulator's
    version of :func:`kupiec_pof`, with the thresholds at 95% and 99.99%.
    Cruder, and the one with capital consequences attached.
    """
    from scipy.stats import binom

    indicator = np.asarray(breach_series, dtype=bool).ravel()
    n = indicator.size
    failures = int(indicator.sum())
    cumulative = float(binom.cdf(failures, n, 1.0 - confidence))
    if cumulative < 0.95:
        return "green", cumulative
    if cumulative < 0.9999:
        return "amber", cumulative
    return "red", cumulative
