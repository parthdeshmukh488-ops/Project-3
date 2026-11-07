"""portlab -- portfolio construction and risk, measured against a known truth.

Portfolio claims are comparative, and comparisons need something to be right
about.  On historical data the out-of-sample covariance is itself an estimate,
so a horse race partly measures which estimator resembles the test period's
sample covariance.  Everything here is therefore validated on synthetic
markets whose covariance is known exactly, so out-of-sample risk is computed
in closed form rather than estimated from a holdout.

``covariance``
    Sample, Ledoit-Wolf, constant-correlation shrinkage and Marchenko-Pastur
    eigenvalue clipping -- with the shrinkage intensity derived rather than
    tuned, and the noise bulk identified from random matrix theory.
``optimizers``
    Minimum variance, risk parity, hierarchical risk parity, maximum
    diversification and Black-Litterman.  The two that need a numerical
    method use algorithms whose convergence is a property: accelerated
    projected gradient with an exact simplex projection, and cyclical
    coordinate descent with a closed-form scalar update.
``risk``
    Value at risk and expected shortfall five ways, from Gaussian through
    GARCH-filtered historical simulation to a fitted extreme-value tail.
``backtest_var``
    Kupiec, Christoffersen and the Basel traffic light -- the tests that say
    whether a risk number was ever a risk number.
``market``
    Factor markets with a known covariance, and a market that changes
    volatility regime partway through.
"""

from .backtest_var import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
)
from .covariance import clip_eigenvalues, ledoit_wolf, sample_covariance
from .market import factor_market, regime_market
from .optimizers import (
    black_litterman,
    equal_weight,
    hierarchical_risk_parity,
    max_diversification,
    min_variance,
    risk_contributions,
    risk_parity,
)
from .risk import evt_var, filtered_historical_var, fit_garch, gaussian_var, historical_var

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "black_litterman",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "clip_eigenvalues",
    "equal_weight",
    "evt_var",
    "factor_market",
    "filtered_historical_var",
    "fit_garch",
    "gaussian_var",
    "hierarchical_risk_parity",
    "historical_var",
    "kupiec_pof",
    "ledoit_wolf",
    "max_diversification",
    "min_variance",
    "regime_market",
    "risk_contributions",
    "risk_parity",
    "sample_covariance",
]
