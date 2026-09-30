"""Section C of the homework: Random Network Distillation (RND).

Edit this file through this ``%%writefile hw/student_rnd.py`` cell, then re-run it
and ``importlib.reload(student_rnd)`` (the quick-test cell below) to pick up your
edits without restarting the kernel.

**Action selection is provided** -- RND acts with the same standard epsilon-greedy
behaviour policy as plain DQN, so you implement only the RND-specific learning core
below.

Conventions used throughout this file:

* ``obs`` -- a flattened observation vector; ``obs_size`` is its length.
* ``B`` -- the batch size (number of transitions in a minibatch).
* ``rms`` -- a running mean/std tracker (``common.RunningMeanStd``): an online
  estimator of the mean and standard deviation of a stream of numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from rnd_dqn import RNDDoubleDQN


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


def build_rnd_networks(
    obs_size: int, output_dim: int, hidden_sizes: tuple[int, ...]
) -> tuple[nn.Module, nn.Module]:
    """Build the two networks RND compares to measure novelty.

    RND scores novelty by how well a *trained predictor* reproduces the output of a
    *fixed random target* on a state: familiar states are easy to predict, novel
    ones are not.

    Args:
        obs_size: length of a flattened observation -- the input width both
            networks must accept (they are called on a ``(B, obs_size)`` tensor).
        output_dim: width of the embedding both networks must emit (their output
            is ``(B, output_dim)``).
        hidden_sizes: a suggested hidden-layer shape, e.g. ``(64, 64)``. Only a
            hint -- you are free to ignore it.

    Returns:
        ``(target, predictor)``, two ``torch.nn.Module`` objects that each map a
        ``(B, obs_size)`` float tensor to a ``(B, output_dim)`` embedding:

        * **target** -- a randomly initialised network that is *never trained*. Its
          parameters must not receive gradients (set ``requires_grad=False`` on
          them) so it stays a fixed, arbitrary function of the state.
        * **predictor** -- a trainable network, regressed toward the target on
          visited states elsewhere in the code.
    """
    target = _mlp(obs_size, output_dim, hidden_sizes)
    predictor = _mlp(obs_size, output_dim, hidden_sizes)

    for parameter in target.parameters():
        parameter.requires_grad_(False)

    return target, predictor


def rnd_intrinsic_bonus(
    agent: "RNDDoubleDQN", next_obs: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return the normalised RND novelty bonus and predictor regression loss."""
    # The target is fixed, so its output is a constant regression label.
    with torch.no_grad():
        target_embedding = agent.rnd_target(next_obs)

    predictor_embedding = agent.rnd_predictor(next_obs)

    # e_psi(s) = 1/2 ||f_psi(s) - f_bar(s)||_2^2, one value per state.
    raw_error = 0.5 * (predictor_embedding - target_embedding).pow(2).sum(dim=1)
    predictor_loss = raw_error.mean()

    # Reward-side novelty must not backpropagate into the predictor. Update the
    # running statistics using this batch, then divide by the running std.
    detached_error = raw_error.detach()
    agent.intrinsic_rms.update(detached_error.cpu().numpy())
    scale = max(float(agent.intrinsic_rms.std), 1e-8)
    intrinsic = detached_error / scale

    return intrinsic, predictor_loss


def rnd_td_target(
    agent: "RNDDoubleDQN",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Build the Double-DQN target using the shaped extrinsic+intrinsic reward."""
    with torch.no_grad():
        # Online network selects a*, target network evaluates Q_target(s', a*).
        next_action = agent.online(next_obs).argmax(dim=1, keepdim=True)
        next_q = agent.target(next_obs).gather(1, next_action).squeeze(1)
        target = shaped_reward + agent.cfg.gamma * (1.0 - done) * next_q
    return target


def rnd_q_loss(
    agent: "RNDDoubleDQN",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Return the Huber loss between Q(s,a) and the RND Double-DQN target."""
    chosen_q = agent.online(obs).gather(1, action.long().unsqueeze(1)).squeeze(1)
    target = rnd_td_target(agent, shaped_reward, next_obs, done)
    return F.smooth_l1_loss(chosen_q, target)
