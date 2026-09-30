"""Saving and loading of training checkpoints for the DeepSea agents.

A checkpoint bundles everything needed to (a) resume reasoning about a run and
(b) reconstruct the exact environment + network(s) for visualization: an
``agent`` tag, the training step, wall-clock experiment time, the network and
optimizer weights, the training config, and the env parameters (including the
per-column action mapping).

Every agent -- plain Double DQN, RND, and Bootstrapped DQN -- writes through
:func:`save_checkpoint`. The only thing that varies between them is the set of
network/optimizer ``state_dict``s passed in ``state`` (e.g. ``online_state`` for
the single-network agents vs. ``ensemble_state`` for the bootstrap ensemble) and
any agent-specific ``extra`` payload. The ``agent`` tag tells the viewer which
shape to expect.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from environment import Environment


def _env_dict(env: Environment) -> dict[str, Any]:
    """Serialise the env-kind tag + parameters needed to rebuild it exactly."""
    return {"env_kind": env.kind, **env.params()}


def save_checkpoint(
    path: str | Path,
    *,
    agent: str,
    step: int,
    elapsed_seconds: float,
    config: Any,
    env: Environment,
    state: dict[str, Any],
    extra: dict | None = None,
) -> Path:
    """Write a checkpoint to ``path``, creating parent dirs as needed.

    Args:
        agent: Tag identifying the agent type (e.g. ``"double_dqn"``), used by
            the viewer to pick how to rebuild the policy.
        config: The agent's dataclass config (stored via :func:`asdict`).
        env: The training environment (its parameters are stored for an exact
            rebuild).
        state: The network/optimizer ``state_dict``s to store, merged into the
            payload as-is (e.g. ``{"online_state": ..., "optimizer_state": ...}``).
        extra: Any additional agent-specific payload (e.g. RND statistics).
    """
    payload: dict[str, Any] = {
        "agent": agent,
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "env": _env_dict(env),
    }
    payload.update(state)
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint dict from ``path``."""
    return torch.load(path, map_location=map_location, weights_only=False)


def env_from_checkpoint(ckpt: dict) -> Environment:
    """Rebuild the exact training environment stored in a checkpoint.

    Dispatches on the stored ``env_kind``; checkpoints predating the tag are
    treated as DeepSea (their payload already matches ``DeepSea.from_params``).
    """
    params = dict(ckpt["env"])
    kind = params.pop("env_kind", "deepsea")
    if kind == "deepsea":
        from deepsea import DeepSea

        return DeepSea.from_params(params)
    if kind == "mountain_car":
        from mountain_car import MountainCar

        return MountainCar.from_params(params)
    raise ValueError(f"unknown env_kind {kind!r}")


def format_elapsed(seconds: float) -> str:
    """Human-readable ``H:MM:SS`` for an elapsed-seconds value."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
