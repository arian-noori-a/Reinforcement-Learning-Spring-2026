"""Vanilla Double DQN for the DeepSea environment.

This is a plain Double DQN (van Hasselt et al., 2016): a replay buffer, an
online and a target Q-network, epsilon-greedy behaviour, and the double-Q
target where the online network *selects* the next action and the target
network *evaluates* it. No prioritized replay, dueling heads, n-step returns,
noisy nets, or other extensions -- just the core algorithm.

The shared plumbing (network, replay buffer, epsilon schedule, config base,
checkpoint format, live preview) lives in :mod:`networks`, :mod:`replay`,
:mod:`common`, and :mod:`checkpoint`; this module keeps only the Double DQN
learning rule and training loop.
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
import torch.nn.functional as F

from checkpoint import format_elapsed, save_checkpoint
from common import BaseConfig, linear_epsilon, live_preview
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


class DoubleDQN:
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

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.learning_rate
        )
        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        self.run_name = cfg.run_name or "deepsea"
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

        # Q(s, a) for the actions actually taken.
        q = self.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net selects the argmax action, target net
        # evaluates it.
        with torch.no_grad():
            next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = reward + cfg.gamma * next_q * (1.0 - done)

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save(self, step: int, elapsed: float, mean_return: float) -> Path:
        """Save a checkpoint as ``<checkpoint_dir>/<run_name>/step_XXXXXXXX.pt``."""
        path = self.run_dir / f"step_{step:08d}.pt"
        save_checkpoint(
            path,
            agent="double_dqn",
            step=step,
            elapsed_seconds=elapsed,
            config=self.cfg,
            env=self.env,
            state={
                "online_state": self.online.state_dict(),
                "target_state": self.target.state_dict(),
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

        self.run_name = cfg.run_name or time.strftime("deepsea_%Y%m%d_%H%M%S")
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
                if updates % cfg.target_update_every == 0:
                    self.target.load_state_dict(self.online.state_dict())

            mean_return = sum(returns) / len(returns) if returns else 0.0

            if step % 5_000 == 0 and returns:
                print(
                    f"step {step:>7d} | episodes {episode_idx:>5d} | "
                    f"eps {epsilon:5.3f} | mean100 return {mean_return:7.4f} | "
                    f"reference {self.env.reference_return:.4f} | "
                    f"time {format_elapsed(time.time() - start_time)}"
                )

            if cfg.viz_every and step % cfg.viz_every == 0:
                self.visualize(step)

            if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
                path = self.save(step, time.time() - start_time, mean_return)
                print(f"  checkpoint -> {path}")


def run_training(cfg: Config, recorder: RunRecorder | None = None) -> DoubleDQN:
    """Build the env + agent for ``cfg`` and train. Returns the agent."""
    env = make_env(cfg)
    print(f"{env.name} reference_return={env.reference_return:.4f} device={cfg.device}")
    agent = DoubleDQN(env, cfg)
    agent.train(recorder=recorder)
    return agent
