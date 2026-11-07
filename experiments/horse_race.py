"""Does any of this beat equal weights?

Mean-variance optimisation has a reputation for being unusable, and the
reputation is mostly deserved -- but for a reason that is often misattributed.
The usual explanation is that expected returns cannot be forecast.  This run
switches expected returns off entirely, so every allocator uses only the
covariance matrix, and the question becomes narrower and answerable: given
that risk *can* be estimated better than return, does using it beat not using
it?

The experiment is run on a synthetic factor market, which makes it a fair test
in a way a historical backtest is not.  Because the true covariance is known,
out-of-sample risk is computed in closed form as ``w' S w`` rather than
estimated from a holdout -- so the comparison is not partly a measure of which
estimator happens to resemble the test period's sample covariance.

Two axes are varied:

* the **allocator** -- equal weight, minimum variance both unconstrained and
  long-only, risk parity, hierarchical risk parity, maximum diversification;
* the **covariance estimator** -- sample, Ledoit-Wolf, constant-correlation
  shrinkage, Marchenko-Pastur eigenvalue clipping.

and one parameter sweeps underneath both: ``q = N / T``, the ratio of assets to
observations, which is what actually controls whether any of this works.

The first version of this experiment only ran minimum variance *long-only*,
and produced the surprising result that the sample covariance was as good as
any shrinkage estimator at every ``q``.  That is a real effect rather than a
bug, and a well-known one: a no-short-sale constraint is itself a form of
shrinkage (Jagannathan and Ma), because the binding non-negativity constraints
act exactly like reducing the largest covariance entries.  With the constraint
in place there is very little left for an estimator to improve.

Running the unconstrained problem alongside makes the point visible.  There
the sample covariance does fail, badly and in the textbook way, and the two
rescues can be compared on equal terms -- shrink the estimate, or constrain
the weights.  Both work; which one works better depends on ``q``.

    python experiments/horse_race.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from portlab.covariance import (
    clip_eigenvalues,
    constant_correlation_shrinkage,
    ledoit_wolf,
    sample_covariance,
)
from portlab.market import factor_market
from portlab.optimizers import (
    equal_weight,
    hierarchical_risk_parity,
    max_diversification,
    min_variance,
    risk_parity,
)

RESULTS = Path(__file__).resolve().parent / "results"

ESTIMATORS = {
    "sample": lambda r: sample_covariance(r, ddof=0),
    "Ledoit-Wolf": lambda r: ledoit_wolf(r).covariance,
    "constant-corr": lambda r: constant_correlation_shrinkage(r).covariance,
    "MP-clipped": clip_eigenvalues,
}

ALLOCATORS = {
    "equal weight": lambda cov: equal_weight(cov.shape[0]),
    "min variance (long only)": lambda cov: min_variance(cov, long_only=True),
    "min variance (unconstrained)": lambda cov: min_variance(cov, long_only=False),
    "risk parity": risk_parity,
    "hierarchical RP": hierarchical_risk_parity,
    "max diversification": lambda cov: max_diversification(cov, long_only=True),
}


def one_draw(n_assets, n_obs, seed):
    """Estimate on one sample; score every portfolio on the true covariance."""
    market = factor_market(n_assets=n_assets, n_obs=n_obs, seed=seed)
    rows = []
    for estimator_name, estimate in ESTIMATORS.items():
        covariance = estimate(market.returns)
        for allocator_name, allocate in ALLOCATORS.items():
            if allocator_name == "equal weight" and estimator_name != "sample":
                continue  # does not use the estimate at all
            weights = allocate(covariance)
            rows.append(
                dict(
                    estimator=estimator_name,
                    allocator=allocator_name,
                    true_volatility=market.true_volatility(weights),
                    concentration=float(np.sum(weights**2)),
                    max_weight=float(weights.max()),
                    n_assets=n_assets,
                    n_obs=n_obs,
                )
            )
    return rows


def aggregate(rows):
    keys = sorted({(r["estimator"], r["allocator"]) for r in rows})
    out = []
    for estimator, allocator in keys:
        subset = [
            r for r in rows if r["estimator"] == estimator and r["allocator"] == allocator
        ]
        out.append(
            dict(
                estimator=estimator,
                allocator=allocator,
                mean_volatility=float(np.mean([r["true_volatility"] for r in subset])),
                std_volatility=float(np.std([r["true_volatility"] for r in subset])),
                effective_assets=float(
                    np.mean([1.0 / r["concentration"] for r in subset])
                ),
                n_draws=len(subset),
            )
        )
    return sorted(out, key=lambda r: r["mean_volatility"])


def sweep_q(ratios, n_assets, n_draws):
    rows = []
    for ratio in ratios:
        n_obs = int(round(n_assets / ratio))
        for seed in range(n_draws):
            for row in one_draw(n_assets, n_obs, seed):
                row["q"] = ratio
                rows.append(row)
        print(f"  q = {ratio:.2f}  (T = {n_obs})  done")
    return rows


def plot(summary, sweep, path):
    from style import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, new_figure, save, style_axes

    fig, axes = new_figure(1, 2, figsize=(13.0, 4.8))

    ax = style_axes(
        axes[0],
        "Out-of-sample volatility at q = 0.2  (true covariance, not a holdout)",
        "annualised volatility",
        "",
    )
    rows = sorted(summary, key=lambda r: -r["mean_volatility"])
    labels = [
        f"{r['allocator']}"
        + ("" if r["allocator"] == "equal weight" else f"  ({r['estimator']})")
        for r in rows
    ]
    values = [r["mean_volatility"] for r in rows]
    errors = [r["std_volatility"] for r in rows]
    colours = [
        ORANGE if r["allocator"] == "equal weight"
        else BLUE if r["estimator"] != "sample"
        else VIOLET
        for r in rows
    ]
    positions = np.arange(len(rows))
    ax.barh(positions, values, xerr=errors, color=colours, zorder=3,
            error_kw=dict(ecolor=MUTED, elinewidth=1.0, capsize=2))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(right=max(values) * 1.22)
    for pos, value, err in zip(positions, values, errors):
        ax.text(value + err + 0.004, pos, f"{value:.3f}", va="center", fontsize=7,
                color=MUTED)
    ax.text(0.60, 0.97, "orange: uses no estimate\npurple: sample covariance\n"
                        "blue: a shrunk estimate",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="top")

    ax = style_axes(
        axes[1],
        "Two different ways to rescue the same disaster",
        "q = assets / observations",
        "annualised volatility",
    )
    series = [
        ("unconstrained, sample cov", VIOLET, "min variance (unconstrained)", "sample"),
        ("unconstrained, Ledoit-Wolf", BLUE, "min variance (unconstrained)",
         "Ledoit-Wolf"),
        ("long-only, sample cov", AQUA, "min variance (long only)", "sample"),
        ("equal weight", ORANGE, "equal weight", "sample"),
    ]
    ratios = sorted({r["q"] for r in sweep})
    for label, colour, allocator, estimator in series:
        y = [
            np.mean([
                r["true_volatility"] for r in sweep
                if r["q"] == ratio and r["allocator"] == allocator
                and r["estimator"] == estimator
            ])
            for ratio in ratios
        ]
        ax.plot(ratios, y, color=colour, lw=2.0, marker="o", ms=5)
        ax.text(ratios[-1], y[-1], f"  {label}", color=colour, fontsize=8.5,
                fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("q = assets / observations", fontsize=9, color=MUTED)
    ax.set_xlim(right=ratios[-1] * 1.55)

    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    n_assets = 40 if args.quick else 100
    n_draws = 4 if args.quick else 25
    ratios = (0.1, 0.4) if args.quick else (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)

    print(f"Horse race at q = 0.2, {n_assets} assets, {n_draws} draws")
    base = []
    for seed in range(n_draws):
        base.extend(one_draw(n_assets, int(n_assets / 0.2), seed))
    summary = aggregate(base)
    print(f"  {'allocator':<30}{'estimator':<16}{'ann. vol':>10}{'eff. assets':>13}")
    for row in summary:
        print(f"  {row['allocator']:<30}{row['estimator']:<16}"
              f"{row['mean_volatility']:>10.4f}{row['effective_assets']:>13.1f}")

    print("\nSweeping q = assets / observations")
    sweep = sweep_q(ratios, n_assets, max(n_draws // 2, 2))

    payload = dict(summary=summary, sweep=sweep, n_assets=n_assets, n_draws=n_draws)
    # A --quick run is a smoke test, not a result.  Writing it to the same
    # path as the full run silently replaces the numbers the README quotes
    # with ones from a hundredth of the work -- which is exactly what happened
    # while verifying the CI smoke tests, leaving the committed JSON
    # contradicting the committed figures.
    suffix = "_quick" if args.quick else ""
    (RESULTS / f"horse_race{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")
    if not args.quick:
        plot(summary, sweep, RESULTS / "horse_race.png")


if __name__ == "__main__":
    main()
