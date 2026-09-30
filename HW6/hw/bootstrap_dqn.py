"""Bootstrapped DQN with randomized prior functions for DeepSea.

This implements Bootstrapped DQN (Osband et al., 2016) augmented with the
*randomized prior functions* of Osband et al. (2018), the canonical deep-
exploration agent for the DeepSea benchmark. The idea, in three pieces:

* **Ensemble of heads.** We keep ``K`` independent Q-networks ("heads"). Each
  head ``k`` is a trainable network ``f_theta_k`` plus a *fixed*, randomly
  initialised prior network ``p_k`` that is never trained::

      Q_k(s, a) = f_theta_k(s, a) + prior_scale * p_k(s, a)

  The frozen prior injects a persistent, per-head source of disagreement that
  ordinary weight initialisation washes out as training proceeds, so the
  ensemble keeps expressing principled uncertainty in unseen states.

* **Bootstrap masks.** Every stored transition carries a binary mask ``m_k`` per
  head (``Bernoulli(mask_prob)``); head ``k`` only trains on transitions whose
  mask bit is set. Different heads therefore see different bootstrap samples of
  the data and stay diverse.

* **Posterior sampling for exploration.** At the start of each episode we draw
  one head uniformly at random and act *greedily* with respect to it for the
  whole episode -- "deep exploration" via approximate Thompson sampling. There
  is no epsilon-greedy noise; exploration comes entirely from ensemble
  disagreement, which is what lets this method crack DeepSea where epsilon-
  greedy needs ~``2**N`` episodes.

The per-head update is a Double DQN target, mirroring :mod:`double_dqn`: the
online head selects the next action and its target copy evaluates it. Each
ensemble member (trainable network + frozen randomized prior) is built by the
student in ``hw/student_bootstrap.py``.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

import student_bootstrap
from checkpoint import format_elapsed, save_checkpoint
from common import BaseConfig, live_preview
from environment import Environment, make_env
from replay import BootstrapReplayBuffer

if TYPE_CHECKING:
    from metrics import RunRecorder


@dataclass
class Config(BaseConfig):
    # -- bootstrap / randomized prior --
    num_heads: int = 10  # ensemble size K
    prior_scale: float = 3.0  # weight on the fixed prior network
    mask_prob: float = 0.5  # Bernoulli prob a transition trains a given head


class BootstrappedDQN:
    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # An ensemble of K independent (trainable + frozen-prior) heads, plus a
        # target copy, both student-built. Loading the full state_dict keeps each
        # target head's frozen prior identical to its online counterpart.
        def make_ensemble() -> nn.ModuleList:
            return student_bootstrap.build_bootstrap_ensemble(
                env.observation_size,
                env.num_actions,
                cfg.num_heads,
                cfg.prior_scale,
                cfg.hidden_sizes,
            )

        self.ensemble = make_ensemble().to(self.device)
        self.target_ensemble = make_ensemble().to(self.device)
        self.target_ensemble.load_state_dict(self.ensemble.state_dict())
        self.target_ensemble.eval()

        # Only the trainable sub-networks are optimised; priors stay frozen.
        params = [p for p in self.ensemble.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
        self.buffer = BootstrapReplayBuffer(cfg.buffer_capacity)

        self.active_head = 0
        # Mean cross-head std of greedy Q over the last batch, for recording the
        # ensemble's disagreement (its exploration signal) as it converges.
        self.last_disagreement = 0.0
        self.run_name = cfg.run_name or "deepsea_boot"
        self.run_dir = Path(cfg.checkpoint_dir) / self.run_name
        self.eval_env: Environment | None = None

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, head: int) -> int:
        """Greedy action under a single ensemble head (student implementation)."""
        return student_bootstrap.boot_select_action(self, obs, head)

    @torch.no_grad()
    def _greedy_policy(self):
        """Greedy policy over the ensemble-mean Q-values, for live previews."""

        @torch.no_grad()
        def policy(env: Environment) -> int:
            # Evaluation acts on the whole posterior (ensemble-mean Q), not one head.
            return student_bootstrap.boot_eval_action(self, env.observation)

        return policy

    def visualize(self, step: int) -> None:
        """Flash a live ensemble-mean greedy rollout, then continue training."""
        assert self.eval_env is not None
        live_preview(
            self.eval_env,
            self._greedy_policy(),
            step=step,
            label="ensemble-mean return",
            reference_return=self.env.reference_return,
            fps=self.cfg.viz_fps,
        )

    def update(self) -> float:
        cfg = self.cfg
        obs, action, reward, next_obs, done, mask = self.buffer.sample(cfg.batch_size)

        obs = torch.as_tensor(obs, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)  # (B, K)

        total_loss = obs.new_zeros(())
        active_heads = 0
        head_max_q: list[torch.Tensor] = []
        for k in range(cfg.num_heads):
            m = mask[:, k]
            if float(m.sum()) == 0.0:
                continue  # no bootstrap samples for this head in the batch
            online_k = self.ensemble[k]
            target_k = self.target_ensemble[k]

            # Masked per-head Double-DQN loss (student implementation): Q(s, a) for
            # the actions taken vs. this head's Double-DQN target, over the head's
            # bootstrap-masked transitions.
            total_loss = total_loss + student_bootstrap.boot_head_loss(
                self, online_k, target_k, obs, action, reward, next_obs, done, m
            )
            active_heads += 1

            # Greedy state value per head, for the disagreement metric below.
            with torch.no_grad():
                head_max_q.append(online_k(obs).max(dim=1).values)

        if active_heads == 0:
            return 0.0
        total_loss = total_loss / active_heads

        # Ensemble disagreement: mean over the batch of the cross-head std of the
        # greedy state value. Shrinks as the heads agree, the bootstrap analog of
        # a decaying exploration bonus.
        if len(head_max_q) >= 2:
            stacked = torch.stack(head_max_q, dim=0)  # (active_heads, B)
            self.last_disagreement = float(stacked.std(dim=0).mean())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return float(total_loss.item())

    def save(self, step: int, elapsed: float, mean_return: float) -> Path:
        """Save a checkpoint as ``<checkpoint_dir>/<run_name>/step_XXXXXXXX.pt``.

        Stores the whole ensemble (trainable + frozen prior weights) under
        ``ensemble_state`` so the exact agent can be rebuilt for visualization.
        """
        path = self.run_dir / f"step_{step:08d}.pt"
        save_checkpoint(
            path,
            agent="bootstrap_dqn",
            step=step,
            elapsed_seconds=elapsed,
            config=self.cfg,
            env=self.env,
            state={
                "ensemble_state": self.ensemble.state_dict(),
                "target_state": self.target_ensemble.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            extra={"mean100_return": mean_return},
        )
        return path

    def train(self, recorder: RunRecorder | None = None) -> None:
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        self.run_name = cfg.run_name or time.strftime("deepsea_boot_%Y%m%d_%H%M%S")
        self.run_dir = Path(cfg.checkpoint_dir) / self.run_name
        start_time = time.time()

        if cfg.viz_every:
            self.eval_env = self.env.clone()

        obs, _ = self.env.reset(seed=cfg.seed)
        if recorder is not None:
            recorder.record_reset(*self.env.grid_cell)
        # Posterior sampling: draw the acting head for the first episode (student).
        self.active_head = student_bootstrap.boot_sample_head(self, rng)
        updates = 0
        episode_return = 0.0
        episode_idx = 0
        returns: deque = deque(maxlen=100)

        for step in range(1, cfg.total_steps + 1):
            action = self.select_action(obs, self.active_head)
            result = self.env.step(action)

            # Each transition gets an independent bootstrap mask over heads.
            mask = (rng.random(cfg.num_heads) < cfg.mask_prob).astype(np.float32)
            self.buffer.push(
                obs, action, result.reward, result.observation, result.terminated, mask
            )
            obs = result.observation
            episode_return += result.reward
            if recorder is not None:
                recorder.record_step(step, *self.env.grid_cell)

            # End the episode on termination *or* truncation (the latter only
            # happens in finite-horizon envs like Mountain Car). The buffer's
            # done flag stays ``terminated`` so truncation doesn't zero the
            # bootstrap target.
            if result.terminated or result.truncated:
                returns.append(episode_return)
                episode_idx += 1
                if recorder is not None:
                    recorder.record_episode(step, episode_return)
                obs, _ = self.env.reset()
                if recorder is not None:
                    recorder.record_reset(*self.env.grid_cell)
                # Re-sample the acting head for the next episode (Thompson step).
                self.active_head = student_bootstrap.boot_sample_head(self, rng)
                episode_return = 0.0

            if len(self.buffer) >= cfg.min_buffer and step % cfg.train_every == 0:
                self.update()
                updates += 1
                if recorder is not None:
                    recorder.record_intrinsic(step, self.last_disagreement)
                if updates % cfg.target_update_every == 0:
                    self.target_ensemble.load_state_dict(self.ensemble.state_dict())

            mean_return = sum(returns) / len(returns) if returns else 0.0

            if step % 5_000 == 0 and returns:
                print(
                    f"step {step:>7d} | episodes {episode_idx:>5d} | "
                    f"heads {cfg.num_heads:>2d} | mean100 return {mean_return:7.4f} | "
                    f"reference {self.env.reference_return:.4f} | "
                    f"time {format_elapsed(time.time() - start_time)}"
                )

            if cfg.viz_every and step % cfg.viz_every == 0:
                self.visualize(step)

            if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
                path = self.save(step, time.time() - start_time, mean_return)
                print(f"  checkpoint -> {path}")


def run_training(cfg: Config, recorder: RunRecorder | None = None) -> BootstrappedDQN:
    """Build the env + bootstrapped agent for ``cfg`` and train. Returns it."""
    env = make_env(cfg)
    print(
        f"{env.name} reference_return={env.reference_return:.4f} "
        f"heads={cfg.num_heads} prior_scale={cfg.prior_scale} device={cfg.device}"
    )
    agent = BootstrappedDQN(env, cfg)
    agent.train(recorder=recorder)
    return agent
