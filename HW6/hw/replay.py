"""Experience replay buffers shared by the DeepSea agents.

* :class:`ReplayBuffer` -- uniform replay over plain transitions, used by
  :mod:`double_dqn` and :mod:`rnd_dqn`.
* :class:`BootstrapReplayBuffer` -- the same, but every transition also carries a
  per-head bootstrap mask, used by :mod:`bootstrap_dqn`.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Uniform experience replay over ``(s, a, r, s', done)`` transitions."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.stack(obs),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.stack(next_obs),
            np.array(done, dtype=np.float32),
        )


class BootstrapReplayBuffer:
    """Uniform replay over ``(s, a, r, s', done, mask)`` transitions.

    ``mask`` is a ``(num_heads,)`` binary vector recording which heads are
    allowed to learn from the transition.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done, mask))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, action, reward, next_obs, done, mask = zip(*batch)
        return (
            np.stack(obs),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.stack(next_obs),
            np.array(done, dtype=np.float32),
            np.stack(mask).astype(np.float32),
        )
