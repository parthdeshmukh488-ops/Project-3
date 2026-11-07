"""Portfolio construction, with convergence rather than a call to a solver.

Five allocators, chosen because they make different bets about what can be
estimated:

:func:`min_variance`
    Uses the whole covariance matrix and no expected returns.  The most
    aggressive user of the estimate, and therefore the most exposed to its
    errors -- which is why :mod:`portlab.covariance` matters so much.
:func:`risk_parity`
    Equalises each asset's *contribution* to portfolio risk.  Uses the
    covariance matrix but never inverts it, so it degrades gracefully as the
    estimate deteriorates.
:func:`hierarchical_risk_parity`
    Lopez de Prado's method: cluster the assets, then allocate down the tree.
    Uses only the correlation *ordering* plus variances, so it is barely
    affected by estimation error in individual covariances.
:func:`max_diversification`
    Maximises the ratio of weighted average volatility to portfolio
    volatility.
:func:`black_litterman`
    The one that does use expected returns, by starting from the market's
    implied ones and tilting towards stated views in proportion to confidence.

Two of them need a numerical method, and both are implemented so their
convergence is a property rather than a hope.  The long-only minimum-variance
problem is solved by accelerated projected gradient with an exact Euclidean
projection onto the simplex; risk parity by the cyclical coordinate descent of
Griveau-Billion, Richard and Roncalli, which reduces each step to a scalar
quadratic with a closed-form positive root and converges for any positive
definite covariance.  Neither needs a general-purpose optimiser, a starting
guess or a tolerance that has to be tuned per problem.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

__all__ = [
    "simplex_projection",
    "min_variance",
    "risk_parity",
    "risk_contributions",
    "hierarchical_risk_parity",
    "max_diversification",
    "equal_weight",
    "black_litterman",
]


def equal_weight(n_assets):
    """The benchmark everything has to beat, and usually does not."""
    return np.full(int(n_assets), 1.0 / int(n_assets))


def simplex_projection(vector):
    """Exact Euclidean projection onto ``{x >= 0, sum x = 1}``.

    The sort-based algorithm of Duchi et al.: the projection is
    ``max(v - theta, 0)`` for a single scalar ``theta``, found by sorting the
    input and locating the largest prefix whose implied threshold keeps every
    member of that prefix positive.  Exact in ``O(n log n)`` -- no iteration,
    so the outer gradient method inherits a projection with no error of its
    own.
    """
    v = np.asarray(vector, dtype=float).ravel()
    n = v.size
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered)
    candidate = ordered - (cumulative - 1.0) / np.arange(1, n + 1)
    rho = int(np.nonzero(candidate > 0)[0][-1])
    theta = (cumulative[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def min_variance(covariance, long_only=False, max_iter=10_000, tol=1e-12):
    """Minimum-variance portfolio, fully invested.

    Unconstrained, the solution is the closed form
    ``w = S^-1 1 / (1' S^-1 1)``, solved as a linear system rather than by
    forming the inverse -- with a near-singular sample covariance the explicit
    inverse loses several digits for no benefit.

    Long-only, the problem is a convex quadratic programme over the simplex
    and is solved by accelerated projected gradient (FISTA).  The step size is
    ``1 / L`` with ``L`` the largest eigenvalue of the covariance, which is the
    Lipschitz constant of the gradient, so convergence is guaranteed at
    ``O(1/k^2)`` without any line search or tuning.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    ones = np.ones(n)

    if not long_only:
        solved = np.linalg.solve(cov, ones)
        return solved / solved.sum()

    step = 1.0 / float(np.linalg.eigvalsh(cov).max())
    weights = equal_weight(n)
    momentum = weights.copy()
    t = 1.0

    for _ in range(max_iter):
        gradient = cov @ momentum
        updated = simplex_projection(momentum - step * gradient)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = updated + ((t - 1.0) / t_next) * (updated - weights)
        if np.max(np.abs(updated - weights)) < tol:
            return updated
        weights, t = updated, t_next
    return weights


def risk_contributions(weights, covariance):
    """Each asset's share of total portfolio risk (Euler allocation).

    Portfolio volatility is homogeneous of degree one in the weights, so by
    Euler's theorem it decomposes exactly into per-asset contributions
    ``w_i (S w)_i / sqrt(w' S w)`` which sum to the total.  That exactness is
    what makes "risk contribution" a meaningful quantity rather than an
    accounting convention, and it is what risk parity equalises.
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    volatility = np.sqrt(max(float(w @ cov @ w), 1e-300))
    return w * (cov @ w) / volatility


def risk_parity(covariance, budgets=None, max_iter=5_000, tol=1e-12):
    """Portfolio whose assets contribute risk in the given proportions.

    Equal risk contribution by default.  Solved by cyclical coordinate descent
    on the convex surrogate

        f(x) = (1/2) x' S x - sum_i b_i log(x_i)

    whose stationary point has risk contributions exactly proportional to
    ``b``.  Fixing every coordinate but one reduces the first-order condition
    to the scalar quadratic

        S_ii x_i^2 + (sum_{j != i} S_ij x_j) x_i - b_i = 0

    with exactly one positive root, so each update is a closed-form square
    root.  The surrogate is strictly convex on the positive orthant and the
    logarithm is a barrier keeping every weight strictly positive, so the
    iteration converges for any positive definite covariance from any positive
    start -- no projection, no line search, and no possibility of a weight
    going negative.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    budgets = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float)
    budgets = budgets / budgets.sum()

    # Start from the inverse-volatility portfolio: already close, and strictly
    # positive, which the barrier requires.
    weights = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    weights = weights / weights.sum()

    for _ in range(max_iter):
        previous = weights.copy()
        for i in range(n):
            cross = float(cov[i] @ weights) - cov[i, i] * weights[i]
            discriminant = cross * cross + 4.0 * cov[i, i] * budgets[i]
            weights[i] = (-cross + np.sqrt(discriminant)) / (2.0 * cov[i, i])
        if np.max(np.abs(weights / weights.sum() - previous / previous.sum())) < tol:
            break
    return weights / weights.sum()


def _quasi_diagonal_order(link):
    """Leaf order of the dendrogram: correlated assets end up adjacent.

    This is the step that makes the recursive bisection meaningful -- without
    it the two halves at each split would be arbitrary groupings rather than
    clusters.
    """
    tree = to_tree(link, rd=False)
    order = []

    def walk(node):
        if node.is_leaf():
            order.append(node.id)
            return
        walk(node.get_left())
        walk(node.get_right())

    walk(tree)
    return np.asarray(order, dtype=int)


def _inverse_variance_weights(cov, indices):
    diagonal = np.diag(cov)[indices]
    weights = 1.0 / np.clip(diagonal, 1e-300, None)
    return weights / weights.sum()


def _cluster_variance(cov, indices):
    weights = _inverse_variance_weights(cov, indices)
    block = cov[np.ix_(indices, indices)]
    return float(weights @ block @ weights)


def hierarchical_risk_parity(covariance):
    """Lopez de Prado's hierarchical risk parity.

    Three steps.  Turn the correlation matrix into a distance,
    ``d = sqrt((1 - rho) / 2)``, which is a proper metric so clustering it is
    meaningful.  Order the assets by the resulting dendrogram so correlated
    ones are adjacent.  Then split the ordered list recursively, giving each
    side a share inversely proportional to its own inverse-variance cluster
    variance.

    The appeal is what it does *not* use: the covariance matrix is never
    inverted and individual off-diagonal entries only enter through the
    clustering, so the method is far less sensitive to estimation error than
    minimum variance -- which is the whole problem with minimum variance.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    deviations = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    correlation = np.clip(cov / np.outer(deviations, deviations), -1.0, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1.0 - correlation), 0.0, None))

    # Condensed upper triangle, as the clustering routine expects.
    condensed = distance[np.triu_indices(n, k=1)]
    order = _quasi_diagonal_order(linkage(condensed, method="single"))

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cluster in clusters:
            if cluster.size <= 1:
                continue
            split = cluster.size // 2
            left, right = cluster[:split], cluster[split:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    return weights / weights.sum()


def max_diversification(covariance, long_only=True, **kwargs):
    """Maximise weighted average volatility over portfolio volatility.

    The diversification ratio is scale invariant, so maximising it is the same
    as minimising variance after rescaling each asset to unit volatility --
    which turns an awkward ratio problem into the convex one
    :func:`min_variance` already solves, exactly and without approximation.
    """
    cov = np.asarray(covariance, dtype=float)
    deviations = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    scaled = cov / np.outer(deviations, deviations)
    synthetic = min_variance(scaled, long_only=long_only, **kwargs)
    weights = synthetic / deviations
    return weights / weights.sum()


def diversification_ratio(weights, covariance):
    """Weighted average volatility divided by portfolio volatility."""
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    deviations = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return float(w @ deviations / np.sqrt(max(w @ cov @ w, 1e-300)))


def black_litterman(
    covariance,
    market_weights,
    picking,
    views,
    risk_aversion=2.5,
    tau=0.05,
    view_uncertainty=None,
):
    """Posterior expected returns, from market equilibrium plus stated views.

    Starts from the returns the market's own weights imply -- reverse
    optimisation, ``pi = lambda S w_mkt`` -- rather than from historical
    averages, which is the point of the method: historical mean returns are so
    noisy that a mean-variance optimiser fed them produces nonsense, whereas
    equilibrium returns are a stable prior that views tilt away from in
    proportion to their stated confidence.

    ``picking`` is the ``(k, n)`` matrix selecting which combinations of assets
    each view is about, and ``views`` the ``k`` expected returns for them.
    ``view_uncertainty`` defaults to He and Litterman's choice,
    ``diag(P tau S P')``, which makes each view's confidence proportional to
    the prior variance of the portfolio it describes -- so a view about a
    volatile combination is automatically treated as less precise.
    """
    cov = np.asarray(covariance, dtype=float)
    market = np.asarray(market_weights, dtype=float).ravel()
    picks = np.atleast_2d(np.asarray(picking, dtype=float))
    view_returns = np.asarray(views, dtype=float).ravel()

    prior = risk_aversion * cov @ market
    prior_cov = tau * cov

    if view_uncertainty is None:
        omega = np.diag(np.diag(picks @ prior_cov @ picks.T))
    else:
        omega = np.atleast_2d(np.asarray(view_uncertainty, dtype=float))

    middle = picks @ prior_cov @ picks.T + omega
    adjustment = prior_cov @ picks.T @ np.linalg.solve(
        middle, view_returns - picks @ prior
    )
    return prior + adjustment
