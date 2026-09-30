"""Section B of the homework: the exploration metric.

Edit this file through this ``%%writefile hw/student_metric.py`` cell, then re-run
it and ``importlib.reload(student_metric)`` (the check cell below) to pick up your
edits without restarting the kernel.
"""

from __future__ import annotations

import numpy as np


def coverage_entropy(
    visit_counts: "np.ndarray", reachable_mask: "np.ndarray"
) -> tuple[float, float]:
    """Summarise how much of the map was explored, and how evenly.

    Two scalars over the cells that *can* be reached (``reachable_mask``); cells
    outside that mask never count.

    Args:
        visit_counts: ``(N, N)`` int array -- how often each grid cell was visited.
        reachable_mask: ``(N, N)`` bool array -- the cells that count toward both
            numbers.

    Returns:
        ``(coverage, entropy_bits)``:

        * **coverage** -- the share of reachable cells visited at least once
          (``0.0`` when none, ``1.0`` when all). A breadth measure.
        * **entropy_bits** -- the Shannon entropy, in *bits*, of the visit
          distribution over the reachable cells: large when visits are spread
          evenly, small when the agent keeps revisiting a few cells. Define it as
          ``0.0`` when nothing reachable has been visited yet.
    """
    counts = np.asarray(visit_counts)
    mask = np.asarray(reachable_mask, dtype=bool)

    if counts.shape != mask.shape:
        raise ValueError(
            "visit_counts and reachable_mask must have the same shape; "
            f"got {counts.shape} and {mask.shape}"
        )

    reachable_counts = counts[mask].astype(np.float64, copy=False)
    num_reachable = reachable_counts.size
    if num_reachable == 0:
        return 0.0, 0.0

    coverage = float(np.count_nonzero(reachable_counts > 0) / num_reachable)

    total_visits = float(reachable_counts.sum())
    if total_visits <= 0.0:
        return coverage, 0.0

    probabilities = reachable_counts[reachable_counts > 0] / total_visits
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return coverage, entropy_bits
