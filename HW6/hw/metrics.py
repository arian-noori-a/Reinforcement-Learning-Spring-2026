"""Per-run metric recording for the DeepSea exploration comparison.

A :class:`RunRecorder` is threaded through an agent's training loop (via the
optional ``recorder=`` argument on ``train``) and accumulates everything the
analysis needs from a *single* run, without changing what the agent learns:

* **Visitation** -- a cumulative ``(N, N)`` count of how often each grid cell was
  landed in, and the env-step at which each cell was *first* visited. These feed
  the state-visitation heatmaps and first-visit-time metrics.
* **Coverage / entropy snapshots** -- at a fixed env-step cadence we snapshot the
  fraction of *reachable* cells seen so far and the Shannon entropy (bits) of the
  visitation distribution, giving the coverage and entropy curves vs env-steps.
* **Episodes** -- the ``(env_step, return, max_column)`` of every finished
  episode, for return-vs-steps learning curves and the deep-exploration
  "reaches the far end vs dithers" signal.
* **Intrinsic series** -- an optional ``(env_step, value)`` stream for agents with
  an exploration bonus (RND), used to watch the bonus decay.

Only standard numpy arrays and Python scalars are stored, so a run serialises to
a single ``.npz`` via :meth:`RunRecorder.save` and reloads with
:func:`load_run`. The recorder deliberately knows nothing about torch or any
specific agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import student_metric


def reachable_mask(size: int) -> np.ndarray:
    """Boolean ``(N, N)`` mask of cells reachable in DeepSea.

    From the top-left start the agent moves down one row and at most one column
    right per step, so at row ``r`` only columns ``0..r`` are reachable -- the
    lower triangle (including the diagonal). The unreachable upper triangle is
    excluded when normalising coverage and when drawing heatmaps.
    """
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    return cols <= rows


# Alias so :meth:`RunRecorder.__init__` can fall back to the DeepSea mask without
# the ``reachable_mask`` parameter shadowing the module-level function.
_default_reachable_mask = reachable_mask


def _default_goal_mask(size: int) -> np.ndarray:
    """The DeepSea goal: a single cell at the bottom-right."""
    mask = np.zeros((size, size), dtype=bool)
    mask[size - 1, size - 1] = True
    return mask


class RunRecorder:
    """Accumulates the per-run metrics for one training run.

    Args:
        size: DeepSea grid side length ``N``.
        total_steps: Number of env steps the run will take (used to place a final
            snapshot exactly at the end).
        snapshot_every: Env steps between coverage/entropy snapshots. Defaults to
            ``total_steps // 200`` (>=1) so every run yields ~200 snapshot points
            on a shared grid, making cross-seed aggregation a simple stack.
        agent: Agent tag stored in the record (e.g. ``"double_dqn"``).
        seed: Seed stored in the record.
        always_right_return: The env's reference (solved) return, stored so
            analysis can normalise scores to ``~1.0 == solved``.
        reachable_mask: Boolean ``(size, size)`` mask of cells that count toward
            coverage. Defaults to DeepSea's lower triangle; pass the env's own
            mask for other environments (e.g. Mountain Car's full phase box).
        goal_mask: Boolean ``(size, size)`` mask of the goal/treasure cell(s);
            the first env-step a cell in this mask is visited is recorded as
            ``goal_step``. Defaults to DeepSea's single bottom-right cell.
    """

    def __init__(
        self,
        size: int,
        total_steps: int,
        *,
        snapshot_every: int | None = None,
        agent: str = "",
        seed: int = 0,
        always_right_return: float = 0.0,
        reachable_mask: np.ndarray | None = None,
        goal_mask: np.ndarray | None = None,
    ) -> None:
        self.size = size
        self.total_steps = total_steps
        self.snapshot_every = snapshot_every or max(total_steps // 200, 1)
        self.agent = agent
        self.seed = seed
        self.always_right_return = always_right_return

        self._reachable = (
            _default_reachable_mask(size)
            if reachable_mask is None
            else np.asarray(reachable_mask, dtype=bool)
        )
        self._n_reachable = int(self._reachable.sum())
        self._goal_mask = (
            _default_goal_mask(size)
            if goal_mask is None
            else np.asarray(goal_mask, dtype=bool)
        )
        self._goal_step = -1  # env-step of first goal visit (-1 = never)

        self.visit_counts = np.zeros((size, size), dtype=np.int64)
        self.first_visit = np.full((size, size), -1, dtype=np.int64)

        # Snapshot series (coverage fraction + entropy in bits) vs env-step.
        self._snap_steps: list[int] = []
        self._snap_coverage: list[float] = []
        self._snap_entropy: list[float] = []

        # Per-episode series.
        self._ep_steps: list[int] = []
        self._ep_returns: list[float] = []
        self._ep_max_col: list[int] = []
        self._ep_cur_max_col = 0

        # Optional intrinsic-bonus series (RND).
        self._intr_steps: list[int] = []
        self._intr_values: list[float] = []

    # -- recording hooks --------------------------------------------------
    def record_reset(self, row: int, col: int) -> None:
        """Note the cell the agent starts an episode in (env-step 0 for cell)."""
        self._touch(0, row, col)
        self._ep_cur_max_col = col

    def record_step(self, env_step: int, row: int, col: int) -> None:
        """Record the cell landed in after one env step, and snapshot on cadence."""
        self._touch(env_step, row, col)
        if col > self._ep_cur_max_col:
            self._ep_cur_max_col = col
        if env_step % self.snapshot_every == 0 or env_step == self.total_steps:
            self._snapshot(env_step)

    def record_episode(self, env_step: int, episode_return: float) -> None:
        """Record a finished episode's return and deepest column reached."""
        self._ep_steps.append(env_step)
        self._ep_returns.append(float(episode_return))
        self._ep_max_col.append(self._ep_cur_max_col)
        self._ep_cur_max_col = 0

    def record_intrinsic(self, env_step: int, value: float) -> None:
        """Record one sample of an agent's exploration bonus (RND)."""
        self._intr_steps.append(env_step)
        self._intr_values.append(float(value))

    # -- internals --------------------------------------------------------
    def _touch(self, env_step: int, row: int, col: int) -> None:
        if self.first_visit[row, col] < 0:
            self.first_visit[row, col] = env_step
        self.visit_counts[row, col] += 1
        if self._goal_step < 0 and self._goal_mask[row, col]:
            self._goal_step = env_step

    def _snapshot(self, env_step: int) -> None:
        # The coverage + entropy computation is the student's exploration metric; it is
        # called by attribute so re-running the notebook's ``%%writefile`` cell +
        # ``importlib.reload(student_metric)`` picks up edits without a restart.
        coverage, entropy = student_metric.coverage_entropy(
            self.visit_counts, self._reachable
        )
        self._snap_steps.append(env_step)
        self._snap_coverage.append(coverage)
        self._snap_entropy.append(entropy)

    # -- output -----------------------------------------------------------
    def result(self) -> dict[str, Any]:
        """Bundle the recorded series into a flat dict of arrays/scalars.

        Values are a mix of arrays and scalars (``Any`` keeps the ``**`` splat
        into :func:`numpy.savez_compressed` clean); callers reading a series back
        should wrap it in ``np.asarray`` before using array methods.
        """
        return {
            "agent": self.agent,
            "seed": self.seed,
            "size": self.size,
            "total_steps": self.total_steps,
            "snapshot_every": self.snapshot_every,
            "always_right_return": self.always_right_return,
            "n_reachable": self._n_reachable,
            "goal_step": self._goal_step,
            "reachable_mask": self._reachable,
            "goal_mask": self._goal_mask,
            "visit_counts": self.visit_counts,
            "first_visit": self.first_visit,
            "snap_steps": np.asarray(self._snap_steps, dtype=np.int64),
            "snap_coverage": np.asarray(self._snap_coverage, dtype=np.float64),
            "snap_entropy": np.asarray(self._snap_entropy, dtype=np.float64),
            "ep_steps": np.asarray(self._ep_steps, dtype=np.int64),
            "ep_returns": np.asarray(self._ep_returns, dtype=np.float64),
            "ep_max_col": np.asarray(self._ep_max_col, dtype=np.int64),
            "intr_steps": np.asarray(self._intr_steps, dtype=np.int64),
            "intr_values": np.asarray(self._intr_values, dtype=np.float64),
        }

    def save(self, path: str | Path) -> Path:
        """Write the run record to ``path`` as a compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.result())
        return path


def load_run(path: str | Path) -> dict[str, np.ndarray]:
    """Load a run record saved by :meth:`RunRecorder.save`.

    Scalars come back as 0-d arrays from ``np.load``; use ``arr.item()`` to read
    them. Returns a plain dict keyed exactly as :meth:`RunRecorder.result`.
    """
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
