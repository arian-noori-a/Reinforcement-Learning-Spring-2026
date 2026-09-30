"""The baseline Q-network shared by the DeepSea agents.

:class:`QNetwork` is the multi-layer perceptron mapping a flattened observation to
one Q-value per action. It is the value network of the plain Double-DQN baseline
(:mod:`double_dqn`) and the extrinsic-reward critic of :mod:`rnd_dqn`.

The exploration-specific networks are *not* defined here: you build them yourself
in the homework -- the RND target/predictor in ``hw/student_rnd.py`` and the
Bootstrapped-DQN ensemble heads in ``hw/student_bootstrap.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_size: int, out_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    """Build a plain ``Linear``/``ReLU`` stack ending in a linear projection."""
    layers: list[nn.Module] = []
    last = in_size
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_size))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Simple MLP mapping a flattened observation to per-action Q-values."""

    def __init__(
        self, obs_size: int, num_actions: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.net = _mlp(obs_size, num_actions, hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
