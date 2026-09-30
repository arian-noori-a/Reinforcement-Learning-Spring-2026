"""Section D of the homework: Bootstrapped DQN with randomized priors.

Edit this file through this ``%%writefile hw/student_bootstrap.py`` cell, then
re-run it and ``importlib.reload(student_bootstrap)`` (the quick-test cell below)
to pick up your edits without restarting the kernel.

Conventions used throughout this file:

* ``obs`` -- a flattened observation vector; ``obs_size`` is its length.
* ``B`` -- the batch size (number of transitions in a minibatch).
* ``K`` -- the ensemble size (number of value heads), i.e. ``agent.cfg.num_heads``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from bootstrap_dqn import BootstrappedDQN


def _mlp(
    input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]
) -> nn.Sequential:
    """Construct a Linear/ReLU MLP with a final linear output layer."""
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class RandomizedPriorHead(nn.Module):
    """One trainable Q-network plus an independent frozen randomized prior."""

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        hidden_sizes: tuple[int, ...],
        prior_scale: float,
    ) -> None:
        super().__init__()
        self.trainable = _mlp(obs_size, num_actions, hidden_sizes)
        self.prior = _mlp(obs_size, num_actions, hidden_sizes)
        self.prior_scale = float(prior_scale)

        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trainable(obs) + self.prior_scale * self.prior(obs)


def build_bootstrap_ensemble(
    obs_size: int,
    num_actions: int,
    num_heads: int,
    prior_scale: float,
    hidden_sizes: tuple[int, ...],
) -> "nn.ModuleList":
    """Build independent trainable Q-heads with frozen randomized priors."""
    return nn.ModuleList(
        [
            RandomizedPriorHead(
                obs_size=obs_size,
                num_actions=num_actions,
                hidden_sizes=hidden_sizes,
                prior_scale=prior_scale,
            )
            for _ in range(num_heads)
        ]
    )


def boot_sample_head(agent: "BootstrappedDQN", rng: "np.random.Generator") -> int:
    """Uniformly sample one acting head for the entire next episode."""
    return int(rng.integers(0, agent.cfg.num_heads))


def boot_select_action(agent: "BootstrappedDQN", obs: "np.ndarray", head: int) -> int:
    """Return the greedy action under the selected ensemble head."""
    obs_tensor = torch.as_tensor(
        obs, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.ensemble[head](obs_tensor)
    return int(q_values.argmax(dim=1).item())


def boot_eval_action(agent: "BootstrappedDQN", obs: "np.ndarray") -> int:
    """Return the greedy action under the ensemble-mean Q-values."""
    obs_tensor = torch.as_tensor(
        obs, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        all_q = torch.stack([head(obs_tensor) for head in agent.ensemble], dim=0)
        mean_q = all_q.mean(dim=0)
    return int(mean_q.argmax(dim=1).item())


def boot_head_target(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Build one head's Double-DQN bootstrap target."""
    with torch.no_grad():
        next_action = online_k(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_k(next_obs).gather(1, next_action).squeeze(1)
        target = reward + agent.cfg.gamma * (1.0 - done) * next_q
    return target


def boot_head_loss(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Return this head's bootstrap-mask-weighted mean Huber loss."""
    chosen_q = online_k(obs).gather(1, action.long().unsqueeze(1)).squeeze(1)
    target = boot_head_target(agent, online_k, target_k, reward, next_obs, done)
    per_transition = F.smooth_l1_loss(chosen_q, target, reduction="none")

    mask = mask.to(dtype=per_transition.dtype)
    return (per_transition * mask).sum() / mask.sum()
