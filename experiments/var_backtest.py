"""Which value-at-risk model survives a regime change?

A VaR model is a falsifiable claim: at 99%, losses should exceed the forecast
on 1% of days, and those breaches should not cluster.  This run puts five
models through a market that changes volatility regime partway through, and
scores them on both counts.

The market is built to break the models in different ways: a level shift in
volatility at day 1000, autocorrelated volatility within each regime so that
breaches have the opportunity to cluster, and Student-t innovations so the
tails are genuinely fatter than normal.  Each of those defeats a different
method, which is the point -- a single failure mode would let one model look
good for the wrong reason.

Everything is run as a rolling out-of-sample forecast: at each day the model
sees only the preceding window and forecasts tomorrow.  That matters more than
it sounds.  Fitting a GARCH model on the whole sample and then "backtesting"
it uses the stressed period to calibrate the forecast for the stressed period,
which is how filtered historical simulation gets reported as near-perfect in
papers that do it.

Two things are scored: **unconditional coverage** (Kupiec), the right number
of breaches, and **independence** (Christoffersen), breaches not clustered.

What the run actually shows is not what I expected.  I assumed independence
would be the discriminating test -- that the static methods would keep roughly
the right breach count overall and fail on clustering.  They do not: with a
rolling window the static methods *do* adapt, because the window eventually
rolls into the stressed period, so the breaches spread out and every method
passes independence comfortably.

They fail on coverage instead, and the per-regime breach rates show why.
Splitting the scored days at the switch, a rolling historical VaR breaches at
1.3% while things are quiet and **4.2%** once volatility triples -- it adapts,
but over the length of its window rather than over the length of the shock.
The GARCH filter holds 1.0% in the stressed period, because its volatility
forecast responds within days.

That is a more useful finding than the one I was looking for: the problem with
a rolling-window VaR is not that it never catches up, it is that catching up
takes exactly as long as the window, and a risk number that is correct two
years after the regime changed is not a risk number.

    python experiments/var_backtest.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from portlab.backtest_var import (
    breaches,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    traffic_light,
)
from portlab.market import regime_market
from portlab.risk import (
    cornish_fisher_var,
    evt_var,
    filtered_historical_var,
    gaussian_var,
    historical_var,
)

RESULTS = Path(__file__).resolve().parent / "results"

METHODS = {
    "gaussian": lambda window, c: gaussian_var(window, c).var,
    "cornish-fisher": lambda window, c: cornish_fisher_var(window, c).var,
    "historical": lambda window, c: historical_var(window, c).var,
    "EVT (peaks over threshold)": lambda window, c: evt_var(window, c).var,
    "filtered historical (GARCH)": lambda window, c: filtered_historical_var(
        window, c
    ).var,
}


def rolling_forecasts(returns, window=500, confidence=0.99, step=1):
    """One-step-ahead VaR from a rolling window, for every method.

    ``step`` subsamples the refit schedule: the GARCH refit dominates the cost
    and the forecast moves slowly, so refitting every few days and holding the
    forecast in between is both realistic and much faster.  The forecast is
    always made strictly before the day it is scored on.

    How far this can be pushed was measured rather than assumed, because a
    stale volatility forecast is exactly what a regime change punishes.
    Refitting every day and every fifth day give identical breach counts (19
    against 15 expected, Kupiec ``p = 0.32``); every tenth day costs a breach
    and drops ``p`` to 0.089, and every twenty-fifth drops it to 0.054.  Five
    is used here -- the same answer as daily at a fifth of the cost, and the
    boundary is close enough that it is worth knowing where it is.
    """
    returns = np.asarray(returns, dtype=float).ravel()
    n = returns.size
    forecasts = {name: np.full(n, np.nan) for name in METHODS}

    last = {name: np.nan for name in METHODS}
    for t in range(window, n):
        if (t - window) % step == 0:
            history = returns[t - window : t]
            for name, estimate in METHODS.items():
                try:
                    last[name] = estimate(history, confidence)
                except Exception:  # pragma: no cover - degenerate windows only
                    last[name] = np.nan
        for name in METHODS:
            forecasts[name][t] = last[name]

    scored = slice(window, n)
    return returns[scored], {name: values[scored] for name, values in forecasts.items()}


def score(realised, forecasts, confidence, switch_index=None):
    rows = []
    for name, values in forecasts.items():
        valid = np.isfinite(values)
        indicator = breaches(realised[valid], values[valid])
        coverage = kupiec_pof(indicator, confidence)
        independence = christoffersen_independence(indicator, confidence)
        combined = christoffersen_conditional_coverage(indicator, confidence)
        colour, _ = traffic_light(indicator, confidence)

        row = dict(
            method=name,
            n_breaches=int(indicator.sum()),
            n_expected=float(coverage.n_expected),
            breach_rate=float(indicator.mean()),
            kupiec_p=coverage.p_value,
            independence_p=independence.p_value,
            conditional_p=combined.p_value,
            traffic_light=colour,
            mean_var=float(np.nanmean(values)),
        )
        if switch_index is not None:
            before = indicator[:switch_index]
            after = indicator[switch_index:]
            row["breach_rate_quiet"] = float(before.mean()) if before.size else np.nan
            row["breach_rate_stressed"] = float(after.mean()) if after.size else np.nan
        rows.append(row)
    return rows


def plot(realised, forecasts, rows, switch_day, path):
    from style import AQUA, BLUE, INK, MUTED, ORANGE, RED, VIOLET, new_figure, save, style_axes

    fig, axes = new_figure(1, 2, figsize=(14.0, 4.8))

    ax = style_axes(
        axes[0],
        "Forecasts through the regime change",
        "day (out-of-sample)",
        "daily loss",
    )
    ax.plot(-realised, color="#c9c9c4", lw=0.7, zorder=1)
    for colour, name in ((ORANGE, "historical"), (BLUE, "filtered historical (GARCH)")):
        ax.plot(forecasts[name], color=colour, lw=1.8, zorder=3)
    ax.axvline(switch_day, color=INK, lw=1.2, ls="--", zorder=4)
    ax.text(switch_day, ax.get_ylim()[1] * 0.97, " volatility triples",
            fontsize=8.5, color=INK, va="top")
    ax.text(0.03, 0.97, "filtered historical (GARCH)", transform=ax.transAxes,
            color=BLUE, fontsize=9, fontweight="bold", va="top")
    ax.text(0.03, 0.90, "rolling historical", transform=ax.transAxes,
            color=ORANGE, fontsize=9, fontweight="bold", va="top")
    ax.text(0.03, 0.83, "grey: realised losses", transform=ax.transAxes,
            color=MUTED, fontsize=8.5, va="top")

    ax = style_axes(
        axes[1],
        "Coverage and independence, by method",
        "p-value",
        "",
    )
    ordered = sorted(rows, key=lambda r: r["conditional_p"])
    positions = np.arange(len(ordered))
    ax.barh(positions - 0.19, [r["kupiec_p"] for r in ordered], height=0.36,
            color=ORANGE, zorder=3)
    ax.barh(positions + 0.19, [r["independence_p"] for r in ordered], height=0.36,
            color=BLUE, zorder=3)
    ax.axvline(0.05, color=RED, lw=1.4, ls="--", zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{r['method']}\n{r['n_breaches']} breaches" for r in ordered], fontsize=7.5
    )
    ax.set_xlim(0, 1.08)
    ax.text(0.07, 0.02, "5%: reject left of here", transform=ax.transAxes,
            color=RED, fontsize=8, va="bottom")
    ax.text(0.55, 0.12, "orange: coverage (Kupiec)", transform=ax.transAxes,
            color=ORANGE, fontsize=9, fontweight="bold")
    ax.text(0.55, 0.05, "blue: independence", transform=ax.transAxes,
            color=BLUE, fontsize=9, fontweight="bold")

    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    confidence = 0.99
    n_obs = 1600 if args.quick else 3000
    switch_at = 1100 if args.quick else 2000
    window = 400 if args.quick else 500
    step = 25 if args.quick else 5

    returns, _ = regime_market(n_obs=n_obs, switch_at=switch_at, seed=11)
    realised, forecasts = rolling_forecasts(returns, window, confidence, step)
    switch_day = max(switch_at - window, 0)
    rows = score(realised, forecasts, confidence, switch_index=switch_day)

    print(f"{len(realised)} out-of-sample days at {confidence:.0%}, "
          f"{rows[0]['n_expected']:.0f} breaches expected")
    print(f"\n{'method':<30}{'breaches':>9}{'quiet':>8}{'stress':>8}"
          f"{'Kupiec p':>10}{'indep p':>9}{'light':>7}")
    for row in sorted(rows, key=lambda r: -r["conditional_p"]):
        print(f"{row['method']:<30}{row['n_breaches']:>9}"
              f"{row['breach_rate_quiet']:>8.3f}{row['breach_rate_stressed']:>8.3f}"
              f"{row['kupiec_p']:>10.4f}{row['independence_p']:>9.4f}"
              f"{row['traffic_light']:>7}")

    payload = dict(confidence=confidence, n_scored=len(realised),
                   switch_day=switch_day, rows=rows)
    # A --quick run is a smoke test, not a result.  Writing it to the same
    # path as the full run silently replaces the numbers the README quotes
    # with ones from a hundredth of the work -- which is exactly what happened
    # while verifying the CI smoke tests, leaving the committed JSON
    # contradicting the committed figures.
    suffix = "_quick" if args.quick else ""
    (RESULTS / f"var_backtest{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not args.quick:
        plot(realised, forecasts, rows, switch_day, RESULTS / "var_backtest.png")


if __name__ == "__main__":
    main()
