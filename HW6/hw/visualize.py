"""Simple matplotlib visualization for the DeepSea environment.

Two helpers:

* :func:`rollout` -- run one episode under a policy and collect the path.
* :func:`plot_trajectory` -- draw the grid and the full agent path at once.
* :func:`animate_trajectory` -- replay the path step by step (window or GIF).
* :func:`demo` -- save a random vs. always-right comparison (``invoke demo``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes

from deepsea import DeepSea
from environment import Environment

# A policy maps an environment to an action. The parameter is intentionally
# ``Any`` so env-specific policies (e.g. DeepSea's always-right, Mountain Car's
# energy-pumping) are accepted alongside the generic greedy ones.
Policy = Callable[[Any], int]


def random_policy(env: DeepSea) -> int:
    return int(np.random.randint(env.num_actions))


def always_right_policy(env: DeepSea) -> int:
    """Pick whichever raw action means "right" in the current column."""
    row, col = env.position
    return int(env.action_mapping[col])


def rollout(env: Environment, policy: Policy, seed: int | None = None):
    """Run one episode and return ``(path, total_reward)``.

    ``path`` is the list of per-step :meth:`Environment.render_state` tuples the
    agent passed through (DeepSea: ``(row, column)`` cells; Mountain Car:
    ``(position, velocity)``), starting at the reset state. The episode ends on
    termination *or* truncation so finite-horizon envs (Mountain Car) stop.
    """
    env.reset(seed=seed)
    path = [env.render_state()]
    total = 0.0
    done = False
    while not done:
        action = policy(env)
        result = env.step(action)
        total += result.reward
        path.append(env.render_state())
        done = result.terminated or result.truncated
    return path, total


def _tv_columns(env: DeepSea) -> tuple[int, ...]:
    """The env's noisy-TV columns (empty for an ordinary noise-free DeepSea)."""
    return tuple(getattr(env, "noise_columns", ()) or ())


def _tv_overlay(env: DeepSea, ax: Axes):
    """Add a flickerable "TV static" image over the noisy-TV columns.

    Returns ``(image, sample)`` where ``sample()`` produces a fresh static frame
    (NaN outside the TV columns so the rest of the board shows through), or
    ``None`` when the env has no noisy TV. Animators re-``set_data`` each frame to
    make the static flicker; the static plotter just leaves the first frame.
    """
    tv_cols = _tv_columns(env)
    if not tv_cols:
        return None
    n = env.size
    rng = np.random.default_rng()

    def sample() -> np.ndarray:
        data = np.full((n, n), np.nan)
        for c in tv_cols:
            data[:, c] = rng.random(n)
        return data

    cmap = plt.cm.gray.copy()
    cmap.set_bad(alpha=0.0)  # transparent outside the TV columns
    im = ax.imshow(sample(), cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.85, zorder=2)
    return im, sample


def _draw_grid(env: DeepSea, ax: Axes) -> None:
    """Draw the static DeepSea board: checkerboard, treasure, and mapping."""
    n = env.size

    # Checkerboard background for readability.
    board = np.indices((n, n)).sum(axis=0) % 2
    ax.imshow(board, cmap="Greys", alpha=0.12, vmin=0, vmax=1)

    # Tint the noisy-TV columns (if any) and label them, so the noisy variant is
    # visibly distinct from plain DeepSea even before the static is drawn.
    tv_cols = _tv_columns(env)
    for c in tv_cols:
        ax.add_patch(
            plt.Rectangle(
                (c - 0.5, -0.5), 1, n, color="tab:purple", alpha=0.10, zorder=0
            )
        )
    if tv_cols:
        ax.text(
            float(np.mean(tv_cols)),
            -1.35,
            "noisy TV",
            ha="center",
            va="center",
            fontsize=8,
            color="tab:purple",
        )

    # Highlight the treasure cell (bottom-right).
    ax.add_patch(
        plt.Rectangle((n - 1.5, n - 1.5), 1, 1, color="gold", alpha=0.6, zorder=1)
    )
    ax.text(n - 1, n - 1, "$", ha="center", va="center", fontsize=12, zorder=2)

    # Annotate, above each column, which raw action (0/1) means "go right"
    # there -- this is the per-column action mapping the agent must discover.
    mapping = env.action_mapping
    for col in range(n):
        ax.text(
            col,
            -0.7,
            str(int(mapping[col])),
            ha="center",
            va="center",
            color="#888888",
            fontsize=8,
        )
    ax.text(
        -0.5,
        -1.1,
        "right-action per column:",
        ha="left",
        va="center",
        fontsize=8,
        color="#888888",
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -1.4)  # row 0 at the top, with room for annotations
    ax.set_aspect("equal")


def plot_trajectory(
    env: DeepSea,
    path: list[tuple[int, int]] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Draw the DeepSea grid and (optionally) the full trajectory at once.

    The top of each column is annotated with the raw action (0/1) that means
    "go right" there. The treasure cell is highlighted in gold.
    """
    n = env.size
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * n + 1, 0.6 * n + 1))

    _draw_grid(env, ax)
    _tv_overlay(env, ax)  # one static frame for the saved figure

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, "-o", color="tab:blue", lw=2, ms=5, zorder=3)
        ax.plot(xs[0], ys[0], "o", color="tab:green", ms=10, zorder=4, label="start")
        ax.plot(xs[-1], ys[-1], "s", color="tab:red", ms=10, zorder=4, label="end")
        ax.legend(loc="upper right", fontsize=8)

    if title:
        ax.set_title(title)
    return ax


def animate_trajectory(
    env: DeepSea,
    path: list[tuple[int, int]],
    fps: float = 3.0,
    save: str | None = None,
    title: str | None = None,
) -> FuncAnimation:
    """Animate the agent walking the trajectory one step at a time.

    Args:
        env: The environment (for the static board).
        path: Sequence of ``(row, column)`` cells from :func:`rollout`.
        fps: Frames (steps) per second.
        save: If given (e.g. ``"rollout.gif"``), write a GIF; else show a window.
        title: Base title; the current step index is appended each frame.

    Returns the :class:`~matplotlib.animation.FuncAnimation` (keep a reference
    to it alive while displaying).
    """
    n = env.size
    fig, ax = plt.subplots(figsize=(0.6 * n + 1, 0.6 * n + 1))
    _draw_grid(env, ax)
    tv = _tv_overlay(env, ax)

    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs[0], ys[0], "o", color="tab:green", ms=10, zorder=4, label="start")
    ax.legend(loc="upper right", fontsize=8)

    (trail,) = ax.plot([], [], "-o", color="tab:blue", lw=2, ms=5, zorder=3)
    (agent,) = ax.plot([], [], "o", color="tab:red", ms=13, zorder=5)
    base = title or "trajectory"
    last = len(path) - 1

    def update(i: int):
        trail.set_data(xs[: i + 1], ys[: i + 1])
        agent.set_data([xs[i]], [ys[i]])
        ax.set_title(f"{base}   step {i}/{last}")
        if tv is not None:
            tv[0].set_data(tv[1]())  # flicker the TV static
            return trail, agent, tv[0]
        return trail, agent

    anim = FuncAnimation(
        fig, update, frames=len(path), interval=1000.0 / fps, blit=False, repeat=False
    )

    if save:
        anim.save(save, writer=PillowWriter(fps=max(1, round(fps))))
        print(f"saved {save}")
        plt.close(fig)
    else:
        plt.show()
    return anim


def play_trajectory(
    env: DeepSea,
    path: list[tuple[int, int]],
    fps: float = 5.0,
    title: str | None = None,
) -> None:
    """Play a trajectory live in an interactive window, then close it.

    Unlike :func:`animate_trajectory` (which blocks until the user closes the
    window, or writes a GIF), this drives the animation itself with
    :func:`matplotlib.pyplot.pause` and tears the figure down once the
    trajectory ends. It blocks for roughly ``len(path) / fps`` seconds and then
    returns -- so a training loop can flash a quick preview of the current
    policy and immediately resume. Harmless on a headless/``Agg`` backend (no
    window appears; it just sleeps).
    """
    n = env.size
    was_interactive = plt.isinteractive()
    plt.ion()
    fig, ax = plt.subplots(figsize=(0.6 * n + 1, 0.6 * n + 1))
    _draw_grid(env, ax)
    tv = _tv_overlay(env, ax)

    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs[0], ys[0], "o", color="tab:green", ms=10, zorder=4, label="start")
    ax.legend(loc="upper right", fontsize=8)

    (trail,) = ax.plot([], [], "-o", color="tab:blue", lw=2, ms=5, zorder=3)
    (agent,) = ax.plot([], [], "o", color="tab:red", ms=13, zorder=5)
    base = title or "trajectory"
    last = len(path) - 1

    try:
        for i in range(len(path)):
            trail.set_data(xs[: i + 1], ys[: i + 1])
            agent.set_data([xs[i]], [ys[i]])
            if tv is not None:
                tv[0].set_data(tv[1]())  # flicker the TV static
            ax.set_title(f"{base}   step {i}/{last}")
            plt.pause(1.0 / max(fps, 1e-6))
    finally:
        plt.close(fig)
        if not was_interactive:
            plt.ioff()


def demo(
    size: int = 10,
    seed: int = 0,
    save_path: str | None = "deepsea.png",
    noise_dim: int = 0,
) -> None:
    env = DeepSea(size=size, seed=seed, noise_dim=noise_dim)

    rand_path, rand_ret = rollout(env, random_policy, seed=seed)
    right_path, right_ret = rollout(env, always_right_policy, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(1.2 * size + 2, 0.6 * size + 1))
    plot_trajectory(
        env, rand_path, ax=axes[0], title=f"random  (return {rand_ret:.3f})"
    )
    plot_trajectory(
        env, right_path, ax=axes[1], title=f"always-right  (return {right_ret:.3f})"
    )
    tv = f", noisy TV cols {list(env.noise_columns)}" if noise_dim else ""
    fig.suptitle(
        f"DeepSea size={size}{tv}  (always-right {env.always_right_return:.3f})"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"saved {save_path}")
    else:
        plt.show()
