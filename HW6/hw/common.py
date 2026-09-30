"""Plumbing shared by the DeepSea agents.

This collects the boring, agent-agnostic pieces so each agent module can focus
on the learning rule that makes it distinct:

* :class:`BaseConfig` -- the training hyperparameters every agent shares; each
  agent's ``Config`` subclasses it and adds its own knobs.
* :func:`linear_epsilon` -- the linear epsilon-greedy schedule (Double DQN / RND).
* :class:`RunningMeanStd` -- online mean/variance estimator (RND reward norm).
* :func:`live_preview` -- flash a live greedy rollout during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from environment import Environment
    from visualize import Policy


@dataclass
class BaseConfig:
    """Training hyperparameters common to every DeepSea agent."""

    size: int = 10
    # -- environment selection --
    # ``size`` is the visitation-grid side length for every env: for DeepSea it
    # is also N (grid / episode length); for Mountain Car it is the phase-space
    # discretisation resolution (the dynamics do not depend on it).
    env_kind: str = "deepsea"  # "deepsea" or "mountain_car"
    mc_max_steps: int = 200  # Mountain Car only: episode truncation horizon
    mc_goal_position: float = 0.5  # Mountain Car only: goal position (x >= this)
    # -- DeepSea noisy-TV extension (env_kind == "deepsea") --
    # A localized "noisy TV": ``ds_noise_dim`` random observation features that
    # fire only while the agent sits in one of ``ds_noise_columns`` (default: the
    # left wall, column 0 -- the opposite side from the treasure). The noise is
    # action-independent and reward-irrelevant, so it traps prediction-error
    # exploration (RND) while leaving value-based exploration (bootstrap) intact.
    # ``ds_noise_dim = 0`` is the ordinary noise-free DeepSea.
    ds_noise_dim: int = 0
    ds_noise_columns: tuple[int, ...] = (0,)
    total_steps: int = 100_000
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 5e-4
    gamma: float = 0.99
    buffer_capacity: int = 50_000
    batch_size: int = 128
    min_buffer: int = 1_000  # steps of data before learning starts
    train_every: int = 1  # env steps between gradient updates
    target_update_every: int = 500  # gradient updates between target syncs
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # -- checkpointing --
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10_000  # env steps between checkpoints
    run_name: str | None = None  # defaults to a timestamped name
    # -- live visualization --
    viz_every: int = 0  # env steps between live policy previews (0 disables)
    viz_fps: float = 5.0  # animation speed for the live preview


class _EpsilonSchedule(Protocol):
    """The subset of a config that :func:`linear_epsilon` reads."""

    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int


def linear_epsilon(step: int, cfg: _EpsilonSchedule) -> float:
    """Linearly anneal epsilon from ``epsilon_start`` to ``epsilon_end``."""
    frac = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


class RunningMeanStd:
    """Online mean/variance estimator (Welford / Chan parallel update).

    Tracks the running statistics of a scalar stream -- here the per-state RND
    prediction error -- so intrinsic rewards can be normalised by their standard
    deviation as the predictor learns.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))


def live_preview(
    env: Environment,
    policy: Policy,
    *,
    step: int,
    label: str,
    reference_return: float,
    fps: float,
) -> None:
    """Flash a live greedy rollout of ``policy`` on ``env``, then continue.

    Rolls out one episode under ``policy`` on the supplied (eval) env and plays
    it in an auto-advancing window that tears itself down, so a training loop can
    preview the current policy and immediately resume. ``label`` names the
    return shown in the title (e.g. ``"greedy return"`` or ``"ensemble-mean
    return"``). The per-env player is picked from ``env.kind``. Imported lazily
    so matplotlib isn't pulled in on headless runs.
    """
    from visualize import rollout

    if env.kind == "mountain_car":
        from mountain_car_viz import play_trajectory
    else:
        from visualize import play_trajectory

    path, ret = rollout(env, policy)
    title = f"step {step}  |  {label} {ret:.3f}  (reference {reference_return:.3f})"
    play_trajectory(cast(Any, env), path, fps=fps, title=title)
