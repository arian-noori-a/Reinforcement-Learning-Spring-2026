"""Reliable RL evaluation statistics, reimplemented from rliable.

This is a small, dependency-free reimplementation of the core estimators from
*"Deep Reinforcement Learning at the Edge of the Statistical Precipice"*
(Agarwal, Schwarzer, Castro, Courville, Bellemare; NeurIPS 2021) and its
``rliable`` library. The motivation, in one line: with only a handful of seeds,
point estimates like ``mean +/- std`` are dominated by outliers and their CIs are
unreliable, so the paper recommends interval estimates of robust aggregate
metrics computed by **stratified bootstrap**, plus distributional views.

What lives here:

* :func:`interquartile_mean` -- the IQM (25% trimmed mean), the paper's
  recommended aggregate: more robust than the mean, more efficient than the
  median.
* :func:`aggregate_mean` / :func:`aggregate_median` / :func:`interquartile_mean`
  -- aggregates that map a ``(num_runs, num_tasks)`` score matrix to a scalar.
* :func:`stratified_bootstrap_ci` -- a CI for any such aggregate, resampling runs
  *with replacement independently within each task* (the stratification).
* :func:`performance_profile` -- the score distribution ``F(tau) = fraction of
  runs with score >= tau``, with a bootstrap confidence band.
* :func:`probability_of_improvement` -- ``P(X > Y)`` averaged over tasks (an
  average Mann-Whitney statistic), with a bootstrap CI.

Scores are passed as a ``(num_runs, num_tasks)`` array per algorithm; with a
single task that is just a column vector. Normalising scores (e.g. by the
always-right return so ``1.0 == solved``) before calling these is the caller's
job.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Aggregate = Callable[[np.ndarray], float]


def _flatten(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).ravel()


def trim_mean(values: np.ndarray, proportiontocut: float) -> float:
    """Mean after removing ``proportiontocut`` of the mass from *each* tail.

    Matches ``scipy.stats.trim_mean``: the number trimmed per tail is
    ``floor(proportiontocut * n)``, so e.g. 25% trimming of 5 values drops the
    single smallest and largest and averages the middle three.
    """
    sorted_vals = np.sort(_flatten(values))
    n = sorted_vals.size
    lower = int(proportiontocut * n)
    upper = n - lower
    if upper <= lower:
        return float(sorted_vals.mean())
    return float(sorted_vals[lower:upper].mean())


def interquartile_mean(scores: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50% of all run-task scores."""
    return trim_mean(scores, 0.25)


def aggregate_mean(scores: np.ndarray) -> float:
    """Mean over all run-task scores."""
    return float(_flatten(scores).mean())


def aggregate_median(scores: np.ndarray) -> float:
    """Median over all run-task scores."""
    return float(np.median(_flatten(scores)))


def _stratified_resample(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One stratified-bootstrap replicate of a ``(num_runs, num_tasks)`` matrix.

    Each task column is resampled independently: ``num_runs`` runs are drawn with
    replacement *within* that task, preserving per-task run counts.
    """
    n_runs, n_tasks = scores.shape
    idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
    return np.take_along_axis(scores, idx, axis=0)


def stratified_bootstrap_ci(
    scores: np.ndarray,
    aggregate: Aggregate = interquartile_mean,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Point estimate and stratified-bootstrap CI for an aggregate metric.

    Args:
        scores: ``(num_runs, num_tasks)`` score matrix for one algorithm.
        aggregate: Aggregate mapping the matrix to a scalar (default IQM).
        reps: Number of bootstrap replicates.
        confidence: Two-sided confidence level for the interval.

    Returns:
        ``(point, ci_low, ci_high)`` where ``point`` is the aggregate on the
        observed scores and the interval is the percentile bootstrap CI.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    rng = np.random.default_rng(seed)
    point = aggregate(scores)
    stats = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        stats[b] = aggregate(_stratified_resample(scores, rng))
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def performance_profile(
    scores: np.ndarray,
    taus: np.ndarray,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-score distribution ``F(tau)`` with a bootstrap confidence band.

    ``F(tau)`` is the fraction of all run-task scores at least ``tau``; sweeping
    ``tau`` traces how many runs clear each performance level, exposing the whole
    distribution rather than a single aggregate.

    Returns ``(profile, band_low, band_high)``, each aligned with ``taus``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    taus = np.asarray(taus, dtype=np.float64)

    def profile_of(mat: np.ndarray) -> np.ndarray:
        flat = mat.ravel()
        return (flat[None, :] >= taus[:, None]).mean(axis=1)

    profile = profile_of(scores)
    rng = np.random.default_rng(seed)
    boot = np.empty((reps, taus.size), dtype=np.float64)
    for b in range(reps):
        boot[b] = profile_of(_stratified_resample(scores, rng))
    alpha = (1.0 - confidence) / 2.0
    band_low = np.quantile(boot, alpha, axis=0)
    band_high = np.quantile(boot, 1.0 - alpha, axis=0)
    return profile, band_low, band_high


def _prob_improvement_point(x: np.ndarray, y: np.ndarray) -> float:
    """``P(X > Y)`` averaged over tasks (ties count as 1/2)."""
    n_tasks = x.shape[1]
    per_task = np.empty(n_tasks, dtype=np.float64)
    for t in range(n_tasks):
        xt = x[:, t][:, None]
        yt = y[:, t][None, :]
        per_task[t] = np.mean((xt > yt) + 0.5 * (xt == yt))
    return float(per_task.mean())


def probability_of_improvement(
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    *,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Probability that algorithm X improves on Y, with a bootstrap CI.

    For each task this is the Mann-Whitney probability that a random X run beats
    a random Y run (ties = 1/2); the reported value averages that over tasks.
    ``0.5`` means indistinguishable, ``1.0`` means X dominates Y everywhere.

    Returns ``(point, ci_low, ci_high)``.
    """
    x = np.asarray(scores_x, dtype=np.float64)
    y = np.asarray(scores_y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    point = _prob_improvement_point(x, y)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        stats[b] = _prob_improvement_point(
            _stratified_resample(x, rng), _stratified_resample(y, rng)
        )
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)
