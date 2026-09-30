"""Double DQN with Random Network Distillation (RND) intrinsic rewards.

This implements the exploration bonus of Burda et al. (2018), "Exploration by
Random Network Distillation", on top of the same Double DQN base used in
:mod:`double_dqn`. The idea, in three pieces:

* **A frozen random target.** A randomly initialised network ``f_bar(s)`` maps an
  observation to an embedding and is *never trained*. It is an arbitrary but
  fixed function of the state.

* **A trained predictor.** A second network ``f(s)`` is trained (by regression)
  to reproduce the target's output on the states the agent actually visits. On
  states seen often, the predictor matches the target well; on novel states it
  has large error because it has never been trained there.

* **Prediction error as a novelty bonus.** The per-state intrinsic reward is how
  far the predictor's embedding sits from the target's on that state. It is high
  on unfamiliar states and decays as they become familiar, so adding it to the
  environment reward pulls the agent toward the unexplored frontier. On DeepSea
  the rarely visited right-hand columns stay novel, which is exactly the signal a
  naive epsilon-greedy explorer lacks.

The agent learns a *single* Q-function on the combined reward
``r = r_extrinsic + intrinsic_coef * r_intrinsic_normalised``; we do not split
the value function into separate intrinsic/extrinsic heads as the original PPO
implementation does -- this keeps the agent a minimal extension of
:mod:`double_dqn`. Intrinsic rewards are normalised by a running standard
deviation (:class:`common.RunningMeanStd`), which keeps the bonus on a stable
scale as it shrinks during training. Because the saved checkpoint still stores a
plain ``online_state`` Q-network, RND checkpoints visualise with the ordinary
``invoke view``.
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

import student_rnd
from checkpoint import format_elapsed, save_checkpoint
from common import (
    BaseConfig,
    RunningMeanStd,
    linear_epsilon,
    live_preview,
)
from environment import Environment, make_env
from networks import QNetwork
from replay import ReplayBuffer

if TYPE_CHECKING:
    from metrics import RunRecorder


@dataclass
class Config(BaseConfig):
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    # -- random network distillation --
    rnd_hidden_sizes: tuple[int, ...] = (64, 64)
    rnd_output_dim: int = 64  # embedding width of the target/predictor nets
    rnd_learning_rate: float = 1e-3  # Adam lr for the predictor
    intrinsic_coef: float = 1.0  # weight on the normalised novelty bonus


class RNDDoubleDQN:
    def __init__(self, env: Environment, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.online = QNetwork(
            env.observation_size, env.num_actions, cfg.hidden_sizes
        ).to(self.device)
        self.target = QNetwork(
            env.observation_size, env.num_actions, cfg.hidden_sizes
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # RND: a frozen random target and a trainable predictor (student-built).
        # The target's parameters never receive gradients and are excluded from
        # the optimizer below; only the predictor is optimised.
        self.rnd_target, self.rnd_predictor = student_rnd.build_rnd_networks(
            env.observation_size, cfg.rnd_output_dim, cfg.rnd_hidden_sizes
        )
        self.rnd_target = self.rnd_target.to(self.device)
        self.rnd_target.eval()
        self.rnd_predictor = self.rnd_predictor.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.learning_rate
        )
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_predictor.parameters(), lr=cfg.rnd_learning_rate
        )
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.intrinsic_rms = RunningMeanStd()
        # Mean normalised novelty bonus of the most recent update, for recording.
        self.last_intrinsic_mean = 0.0

        self.run_name = cfg.run_name or "deepsea_rnd"
        self.run_dir = Path(cfg.checkpoint_dir) / self.run_name
        self.eval_env: Environment | None = None

    @torch.no_grad()
    def _greedy_policy(self):
        """Greedy policy over the current online network, for live previews."""

        @torch.no_grad()
        def policy(env: Environment) -> int:
            obs = torch.as_tensor(env.observation, device=self.device).unsqueeze(0)
            return int(self.online(obs).argmax(dim=1).item())

        return policy

    def visualize(self, step: int) -> None:
        """Flash a live greedy rollout of the current policy, then continue."""
        assert self.eval_env is not None
        live_preview(
            self.eval_env,
            self._greedy_policy(),
            step=step,
            label="greedy return",
            reference_return=self.env.reference_return,
            fps=self.cfg.viz_fps,
        )

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy behaviour policy (provided -- standard DQN exploration)."""
        if random.random() < epsilon:
            return random.randrange(self.env.num_actions)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        q = self.online(obs_t)
        return int(q.argmax(dim=1).item())

    def update(self) -> float:
        cfg = self.cfg
        obs, action, reward, next_obs, done = self.buffer.sample(cfg.batch_size)

        obs = torch.as_tensor(obs, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # RND novelty bonus for the landed-in states, plus the predictor loss
        # (student implementation, called by attribute so notebook edits apply).
        intrinsic, predictor_loss = student_rnd.rnd_intrinsic_bonus(self, next_obs)
        self.last_intrinsic_mean = float(intrinsic.mean())
        shaped_reward = reward + cfg.intrinsic_coef * intrinsic

        # Double DQN critic loss on the shaped reward (student implementation):
        # Q(s, a) for the actions taken vs. the Double-DQN target.
        q_loss = student_rnd.rnd_q_loss(
            self, obs, action, shaped_reward, next_obs, done
        )

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Train the predictor to match the frozen target on these states.
        self.rnd_optimizer.zero_grad()
        predictor_loss.backward()
        self.rnd_optimizer.step()

        return float(q_loss.item())

    def save(self, step: int, elapsed: float, mean_return: float) -> Path:
        """Save a checkpoint as ``<checkpoint_dir>/<run_name>/step_XXXXXXXX.pt``.

        Reuses :func:`checkpoint.save_checkpoint` (so the policy is viewable with
        the ordinary ``invoke view``) and stores the RND predictor/target and
        intrinsic-reward statistics as extra payload.
        """
        path = self.run_dir / f"step_{step:08d}.pt"
        save_checkpoint(
            path,
            agent="rnd_dqn",
            step=step,
            elapsed_seconds=elapsed,
            config=self.cfg,
            env=self.env,
            state={
                "online_state": self.online.state_dict(),
                "target_state": self.target.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            extra={
                "mean100_return": mean_return,
                "rnd_predictor_state": self.rnd_predictor.state_dict(),
                "rnd_target_state": self.rnd_target.state_dict(),
                "rnd_optimizer_state": self.rnd_optimizer.state_dict(),
                "intrinsic_rms": {
                    "mean": self.intrinsic_rms.mean,
                    "var": self.intrinsic_rms.var,
                    "count": self.intrinsic_rms.count,
                },
            },
        )
        return path

    def train(self, recorder: RunRecorder | None = None) -> None:
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.run_name = cfg.run_name or time.strftime("deepsea_rnd_%Y%m%d_%H%M%S")
        self.run_dir = Path(cfg.checkpoint_dir) / self.run_name
        start_time = time.time()

        if cfg.viz_every:
            self.eval_env = self.env.clone()

        obs, _ = self.env.reset(seed=cfg.seed)
        if recorder is not None:
            recorder.record_reset(*self.env.grid_cell)
        updates = 0
        episode_return = 0.0
        episode_idx = 0
        returns: deque = deque(maxlen=100)

        for step in range(1, cfg.total_steps + 1):
            epsilon = linear_epsilon(step, cfg)
            action = self.select_action(obs, epsilon)
            result = self.env.step(action)

            self.buffer.push(
                obs, action, result.reward, result.observation, result.terminated
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
                episode_return = 0.0

            if len(self.buffer) >= cfg.min_buffer and step % cfg.train_every == 0:
                self.update()
                updates += 1
                if recorder is not None:
                    recorder.record_intrinsic(step, self.last_intrinsic_mean)
                if updates % cfg.target_update_every == 0:
                    self.target.load_state_dict(self.online.state_dict())

            mean_return = sum(returns) / len(returns) if returns else 0.0

            if step % 5_000 == 0 and returns:
                print(
                    f"step {step:>7d} | episodes {episode_idx:>5d} | "
                    f"eps {epsilon:5.3f} | int_std {self.intrinsic_rms.std:7.4f} | "
                    f"mean100 return {mean_return:7.4f} | "
                    f"reference {self.env.reference_return:.4f} | "
                    f"time {format_elapsed(time.time() - start_time)}"
                )

            if cfg.viz_every and step % cfg.viz_every == 0:
                self.visualize(step)

            if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
                path = self.save(step, time.time() - start_time, mean_return)
                print(f"  checkpoint -> {path}")


def run_training(cfg: Config, recorder: RunRecorder | None = None) -> RNDDoubleDQN:
    """Build the env + RND agent for ``cfg`` and train. Returns the agent."""
    env = make_env(cfg)
    print(
        f"{env.name} reference_return={env.reference_return:.4f} "
        f"intrinsic_coef={cfg.intrinsic_coef} device={cfg.device}"
    )
    agent = RNDDoubleDQN(env, cfg)
    agent.train(recorder=recorder)
    return agent
