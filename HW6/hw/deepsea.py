"""Configurable DeepSea exploration environment.

DeepSea (Osband et al., bsuite) is a hard-exploration benchmark. The world is
an ``N x N`` grid. The agent starts in the top-left cell and is forced to
descend exactly one row per step, so every episode lasts ``N`` steps. At each
step the agent picks one of two actions; in every column a fixed (optionally
randomized) mapping decides which of the two actions means "go right" and which
means "go left". Going right costs a tiny penalty, and the only large reward
(``+1``) sits in the bottom-right cell. Reaching it therefore requires choosing
"right" on every single step, which a naive epsilon-greedy explorer finds with
probability ``2**-N`` -- making this a clean test of deep, directed exploration.

**Noisy-TV variant.** Setting ``noise_dim > 0`` turns selected columns into a
"noisy TV" (Burda et al., 2018): while the agent sits in one of
``noise_columns`` (by default the left wall, column 0 -- the *opposite* side from
the treasure) its observation is padded with ``noise_dim`` fresh uniform random
values; everywhere else those features are zero. The noise is action-independent
and reward-irrelevant, so it is pure aleatoric distraction. It is designed to
expose the failure mode of prediction-error exploration: a random-network-
distillation predictor can never fit the ever-changing noise, so its novelty
bonus stays permanently high *at the TV* and lures the agent left, away from the
treasure. Value-/uncertainty-based exploration (bootstrapped DQN) is unmoved,
because reward-irrelevant input noise produces no lasting disagreement between
its value heads. ``noise_dim = 0`` (the default) is the ordinary noise-free
DeepSea, byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from environment import Environment, StepResult

# Actions are a binary choice; their meaning per column is set by the mapping.
ACTION_LEFT = 0
ACTION_RIGHT = 1

# ``StepResult`` lives in :mod:`environment` (shared with Mountain Car); it is
# re-exported here so ``from deepsea import StepResult`` keeps working.
__all__ = ["ACTION_LEFT", "ACTION_RIGHT", "DeepSea", "StepResult"]

# Glyphs cycled through to draw a "noisy TV" cell in the ASCII render so the
# noisy columns visibly flicker like television static.
_NOISE_GLYPHS = "#%&@=+"


class DeepSea(Environment):
    """A configurable DeepSea environment.

    Args:
        size: Grid side length ``N`` (also the episode length).
        move_cost: Penalty subtracted for each rightward move. Defaults to
            ``0.01 / size`` so the optimal return is ``1 - 0.01 = 0.99``.
        treasure_reward: Reward for reaching the bottom-right cell.
        randomize_actions: If ``True``, draw a per-column action mapping so that
            "right" is not always the same raw action. Fixed for the env's life.
        slip_prob: Probability that the resolved move is flipped on any step,
            giving a stochastic-transition variant. ``0`` is deterministic.
        noise_dim: Number of "noisy TV" features appended to the observation
            (``0`` = the ordinary noise-free DeepSea). When ``> 0`` these
            features carry fresh uniform noise while the agent is in a
            ``noise_columns`` cell and zeros elsewhere.
        noise_columns: Columns that act as a noisy TV (defaults to the left wall,
            ``(0,)``, when ``noise_dim > 0``). Ignored when ``noise_dim == 0``.
        seed: Seed for the action mapping, slipping, and TV-noise RNGs.
        action_mapping: Explicit per-column "right" action (length ``size``,
            values in ``{0, 1}``). Overrides ``randomize_actions``/``seed`` for
            the mapping; used to reconstruct an exact env from a checkpoint.
    """

    kind = "deepsea"

    def __init__(
        self,
        size: int = 10,
        *,
        move_cost: float | None = None,
        treasure_reward: float = 1.0,
        randomize_actions: bool = True,
        slip_prob: float = 0.0,
        noise_dim: int = 0,
        noise_columns: tuple[int, ...] | list[int] | None = None,
        seed: int | None = None,
        action_mapping: np.ndarray | list[int] | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("size must be >= 2")
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError("slip_prob must be in [0, 1]")
        if noise_dim < 0:
            raise ValueError("noise_dim must be >= 0")

        self.size = size
        self.move_cost = 0.01 / size if move_cost is None else move_cost
        self.treasure_reward = treasure_reward
        self.slip_prob = slip_prob

        # Noisy-TV columns: default to the left wall (column 0) when noise is on.
        if noise_columns is None:
            cols: tuple[int, ...] = (0,) if noise_dim > 0 else ()
        else:
            cols = tuple(int(c) for c in noise_columns)
            if any(c < 0 or c >= size for c in cols):
                raise ValueError("noise_columns must each be in [0, size)")
        self.noise_dim = noise_dim
        self.noise_columns = tuple(sorted(set(cols)))
        self._noise_cols = set(self.noise_columns)
        # A dedicated RNG so the TV stream never repeats across episodes and never
        # perturbs the dynamics (mapping/slip) RNG. Offset the seed so it is not
        # correlated with the mapping draws.
        self._noise_rng = np.random.default_rng(None if seed is None else seed + 1)
        self._noise = np.zeros(noise_dim, dtype=np.float32)

        self._rng = np.random.default_rng(seed)

        # action_mapping[col] is the raw action that means "go right" in `col`.
        if action_mapping is not None:
            mapping = np.asarray(action_mapping, dtype=np.int64)
            if mapping.shape != (size,) or not np.isin(mapping, (0, 1)).all():
                raise ValueError("action_mapping must be length `size` of 0/1")
            self._action_mapping = mapping
        elif randomize_actions:
            self._action_mapping = self._rng.integers(0, 2, size=size).astype(np.int64)
        else:
            self._action_mapping = np.full(size, ACTION_RIGHT, dtype=np.int64)

        self._row = 0
        self._column = 0
        self._done = True

    # -- spaces -----------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return 2

    @property
    def observation_size(self) -> int:
        return self.size * self.size + self.noise_dim

    @property
    def name(self) -> str:
        tag = f", noisy x{self.noise_dim}" if self.noise_dim else ""
        return f"DeepSea(N={self.size}{tag})"

    @property
    def reference_return(self) -> float:
        """Reference (solved) return -- the always-right return for DeepSea."""
        return self.always_right_return

    @property
    def grid_cell(self) -> tuple[int, int]:
        """Current ``(row, col)`` for the visitation metrics (== :attr:`position`)."""
        return self.position

    def reachable_mask(self, size: int) -> np.ndarray:
        """Cells reachable in DeepSea: the lower triangle (incl. diagonal)."""
        from metrics import reachable_mask as _reachable_mask

        return _reachable_mask(size)

    def goal_mask(self, size: int) -> np.ndarray:
        """The single treasure cell at the bottom-right."""
        mask = np.zeros((size, size), dtype=bool)
        mask[size - 1, size - 1] = True
        return mask

    def render_state(self) -> tuple[int, int]:
        return self.position

    @property
    def always_right_return(self) -> float:
        """Return of the always-right policy -- the conventional DeepSea target.

        Going right on every step makes ``size`` rightward moves (including the
        clamped move in the bottom-right cell that collects the treasure), so
        with the default ``move_cost = 0.01 / size`` this is ``1 - 0.01 = 0.99``.

        Note this is the *reference* return quoted in the literature, not the
        strict maximum: an agent can shave one move cost by idling for one step
        against the left wall (a free clamped left move at column 0) before
        running right, giving ``treasure_reward - move_cost * (size - 1)``.
        """
        return self.treasure_reward - self.move_cost * self.size

    @property
    def position(self) -> tuple[int, int]:
        """Current ``(row, column)`` of the agent (row clamped to the grid)."""
        return min(self._row, self.size - 1), self._column

    @property
    def action_mapping(self) -> np.ndarray:
        """Copy of the per-column raw action that means "go right"."""
        return self._action_mapping.copy()

    @property
    def observation(self) -> np.ndarray:
        """One-hot observation of the agent's current cell."""
        return self._observation()

    # -- rebuild ----------------------------------------------------------
    def params(self) -> dict[str, Any]:
        """Constructor parameters needed to rebuild this env exactly."""
        return {
            "size": self.size,
            "move_cost": self.move_cost,
            "treasure_reward": self.treasure_reward,
            "slip_prob": self.slip_prob,
            "noise_dim": self.noise_dim,
            "noise_columns": list(self.noise_columns),
            "action_mapping": self._action_mapping.tolist(),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> DeepSea:
        return cls(
            size=int(params["size"]),
            move_cost=float(params["move_cost"]),
            treasure_reward=float(params["treasure_reward"]),
            slip_prob=float(params["slip_prob"]),
            # ``.get`` so checkpoints predating the noisy-TV variant rebuild as
            # the ordinary noise-free DeepSea.
            noise_dim=int(params.get("noise_dim", 0)),
            noise_columns=params.get("noise_columns"),
            action_mapping=params["action_mapping"],
        )

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        """Return an ASCII picture of the grid.

        ``A`` marks the agent, ``$`` the treasure cell, ``.`` an empty cell, and
        ``*`` the agent sitting on the treasure. When the env has a noisy TV,
        cells in the noise columns flicker through :data:`_NOISE_GLYPHS` so the
        distracting region is visible at a glance (the agent's ``A`` still wins
        the cell it occupies).
        """
        a_row, a_col = self.position
        # A throwaway RNG so the TV static re-randomises on every render call
        # without touching the reproducible observation-noise stream.
        flicker = np.random.default_rng()
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                is_agent = r == a_row and c == a_col
                is_treasure = r == self.size - 1 and c == self.size - 1
                if is_agent and is_treasure:
                    cells.append("*")
                elif is_agent:
                    cells.append("A")
                elif is_treasure:
                    cells.append("$")
                elif c in self._noise_cols:
                    cells.append(
                        _NOISE_GLYPHS[int(flicker.integers(len(_NOISE_GLYPHS)))]
                    )
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    # -- dynamics ---------------------------------------------------------
    def _refresh_noise(self) -> None:
        """Redraw the noisy-TV features for the current cell.

        Called exactly once per :meth:`reset`/:meth:`step` so repeated reads of
        :attr:`observation` within one env step stay consistent. The features are
        a fresh uniform draw while the agent is in a noise column (an unfittable,
        ever-changing distractor) and zeros everywhere else.
        """
        if self.noise_dim == 0:
            return
        if self._column in self._noise_cols:
            self._noise = self._noise_rng.random(self.noise_dim).astype(np.float32)
        else:
            self._noise = np.zeros(self.noise_dim, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        """One-hot cell encoding, optionally padded with the noisy-TV features.

        Shape is ``(N*N,)`` for the plain env and ``(N*N + noise_dim,)`` once a
        noisy TV is configured.
        """
        obs = np.zeros((self.size, self.size), dtype=np.float32)
        # After the final step the agent steps off the bottom; clamp the row so
        # the terminal observation marks the bottom-row cell it reached.
        row = min(self._row, self.size - 1)
        obs[row, self._column] = 1.0
        flat = obs.reshape(-1)
        if self.noise_dim:
            return np.concatenate([flat, self._noise])
        return flat

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Reseed the TV stream too, so a seeded reset reproduces the whole
            # run; unseeded resets leave it advancing (non-repeating) on purpose.
            self._noise_rng = np.random.default_rng(seed + 1)
        self._row = 0
        self._column = 0
        self._done = False
        self._refresh_noise()
        return self._observation(), {"action_mapping": self._action_mapping.copy()}

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        if action not in (ACTION_LEFT, ACTION_RIGHT):
            raise ValueError(f"action must be 0 or 1, got {action}")

        # Resolve the raw action into a direction using this column's mapping.
        go_right = action == self._action_mapping[self._column]
        if self.slip_prob > 0.0 and self._rng.random() < self.slip_prob:
            go_right = not go_right

        is_last_row = self._row == self.size - 1

        reward = 0.0
        if go_right:
            self._column = min(self._column + 1, self.size - 1)
            reward -= self.move_cost
        else:
            self._column = max(self._column - 1, 0)

        # The treasure sits in the bottom-right cell.
        if is_last_row and self._column == self.size - 1:
            reward += self.treasure_reward

        self._row += 1
        self._done = self._row == self.size

        # Redraw the TV for the cell just landed in (depends on the new column).
        self._refresh_noise()

        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=self._done,
            truncated=False,
            info={"row": self._row, "column": self._column},
        )
