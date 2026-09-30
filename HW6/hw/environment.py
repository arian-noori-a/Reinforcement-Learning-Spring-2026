"""Shared environment abstraction for the exploration agents.

Originally every layer was hardcoded to :class:`deepsea.DeepSea`. To compare the
*same* agents on a second hard-exploration problem (sparse Mountain Car) without
touching their learning rules, the pieces each layer needs from an environment
are collected behind a small :class:`Environment` interface:

* the **agents** need only ``observation_size`` / ``num_actions`` / ``reset`` /
  ``step`` / ``observation`` -- so they are already environment-agnostic;
* the **metric recorder** needs a discrete ``grid_cell`` plus a
  ``reachable_mask`` and ``goal_mask`` over a shared ``size x size`` grid, and a
  ``reference_return`` to normalise scores;
* **checkpointing / cloning** need ``kind`` + ``params()`` + ``from_params()``;
* the **viewer** needs ``render_state()`` / ``render()``.

By convention ``size`` is the *visitation-grid side length* for every env. For
DeepSea it is also ``N`` (the grid / episode length); for Mountain Car it is just
the phase-space discretisation resolution -- the dynamics do not depend on it.

:func:`make_env` builds the environment selected by a config's ``env_kind``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    """Gymnasium-style return value for :meth:`Environment.step`."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class Environment(ABC):
    """The interface every environment in this testbed implements.

    Concrete envs (:class:`deepsea.DeepSea`, :class:`mountain_car.MountainCar`)
    subclass this. The :class:`~metrics.RunRecorder`, :mod:`analysis`,
    :mod:`checkpoint`, and the viewer talk only to these members, so adding a new
    environment never requires changing an agent's learning rule.
    """

    #: Short tag stored in checkpoints and used to dispatch reconstruction/viz.
    kind: str = ""

    # -- spaces -----------------------------------------------------------
    @property
    @abstractmethod
    def observation_size(self) -> int:
        """Flattened observation length fed to the networks."""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Size of the discrete action set."""

    @property
    @abstractmethod
    def observation(self) -> np.ndarray:
        """The current observation (same encoding ``reset``/``step`` return)."""

    # -- dynamics ---------------------------------------------------------
    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Begin a new episode; return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Advance one step under ``action``."""

    # -- identity ---------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in printouts and figure titles."""

    # -- metric hooks (a shared ``size x size`` grid view) ----------------
    @property
    @abstractmethod
    def grid_cell(self) -> tuple[int, int]:
        """Current discrete ``(row, col)`` in ``[0, size)^2`` for visitation."""

    @abstractmethod
    def reachable_mask(self, size: int) -> np.ndarray:
        """Boolean ``(size, size)`` mask of cells that count toward coverage."""

    @abstractmethod
    def goal_mask(self, size: int) -> np.ndarray:
        """Boolean ``(size, size)`` mask of the goal/treasure cell(s)."""

    @property
    @abstractmethod
    def reference_return(self) -> float:
        """Reference episode return that scores normalise against (``~1`` solved)."""

    # -- visualization ----------------------------------------------------
    @abstractmethod
    def render_state(self) -> tuple[float, float]:
        """A 2-tuple summarising the current state for trajectory plotting."""

    @abstractmethod
    def render(self) -> str:
        """An ASCII picture of the current state."""

    # -- rebuild ----------------------------------------------------------
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Constructor parameters needed to rebuild this env exactly."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, Any]) -> Environment:
        """Rebuild an env from a :meth:`params` dict."""

    def clone(self) -> Environment:
        """A fresh, identically-configured env (used for eval/live previews)."""
        return type(self).from_params(self.params())


def make_env(cfg: Any) -> Environment:
    """Build the environment selected by ``cfg.env_kind``.

    ``cfg`` is any agent ``Config`` (a :class:`common.BaseConfig` subclass): the
    factory reads ``env_kind``, the shared ``size`` (grid side length), the
    Mountain-Car-only ``mc_*`` knobs, and the DeepSea-only ``ds_noise_*`` knobs
    (the noisy-TV variant), falling back to sensible defaults.
    """
    kind = getattr(cfg, "env_kind", "deepsea")
    if kind == "deepsea":
        from deepsea import DeepSea

        return DeepSea(
            size=cfg.size,
            seed=cfg.seed,
            noise_dim=getattr(cfg, "ds_noise_dim", 0),
            noise_columns=getattr(cfg, "ds_noise_columns", (0,)),
        )
    if kind == "mountain_car":
        from mountain_car import MountainCar

        return MountainCar(
            bins=cfg.size,
            max_steps=getattr(cfg, "mc_max_steps", 200),
            goal_position=getattr(cfg, "mc_goal_position", 0.5),
            seed=cfg.seed,
        )
    raise ValueError(f"unknown env_kind {kind!r}")
