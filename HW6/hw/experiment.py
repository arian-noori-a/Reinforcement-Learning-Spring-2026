"""Multi-seed, multi-size experiment runner for the agent comparison.

Trains each agent across a grid of DeepSea sizes and seeds, threading a
:class:`metrics.RunRecorder` through every run and writing one ``.npz`` record
per run under ``<results_dir>/runs/<agent>/n<size>_seed<seed>.npz``. A small
``manifest.json` capturing the grid is written alongside so :mod:`analysis` can
reload the whole sweep without guessing.

The same :class:`ExperimentConfig` scales from the quick validation grid
(single size, a few seeds, ~50k steps) up to the full multi-size comparison just
by widening ``sizes`` / ``seeds`` / ``total_steps`` -- nothing else changes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from environment import make_env
from metrics import RunRecorder

AGENTS: tuple[str, ...] = ("double_dqn", "rnd_dqn", "bootstrap_dqn")


@dataclass
class ExperimentConfig:
    """Definition of a comparison sweep over agents x sizes x seeds."""

    agents: tuple[str, ...] = AGENTS
    # ``env_kind`` picks the environment for the whole sweep ("deepsea" or
    # "mountain_car"); ``sizes`` is the shared visitation-grid side length (also
    # N for DeepSea; the phase-space resolution for Mountain Car).
    env_kind: str = "deepsea"
    sizes: tuple[int, ...] = (15,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    total_steps: int = 50_000
    results_dir: str = "results"
    device: str | None = None
    snapshot_every: int | None = None
    # Number of runs to train concurrently. The runs are independent (each
    # writes its own record), so on a GPU that a single small run can't saturate,
    # a few workers sharing it raise utilization. 1 = the original sequential run.
    workers: int = 1
    # Per-agent config overrides applied when the attribute exists on that
    # agent's Config (e.g. epsilon_decay_steps only exists on the epsilon agents).
    overrides: dict[str, Any] = field(default_factory=dict)


def _build(
    agent: str,
    *,
    size: int,
    total_steps: int,
    seed: int,
    checkpoint_dir: str,
    run_name: str,
    device: str | None,
    overrides: dict[str, Any],
    env_kind: str = "deepsea",
):
    """Construct ``(config, run_training)`` for one agent at one size/seed."""
    if agent == "double_dqn":
        from double_dqn import Config, run_training
    elif agent == "rnd_dqn":
        from rnd_dqn import Config, run_training
    elif agent == "bootstrap_dqn":
        from bootstrap_dqn import Config, run_training
    else:
        raise ValueError(f"unknown agent {agent!r}")

    cfg = Config(
        size=size,
        total_steps=total_steps,
        seed=seed,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=total_steps,  # one checkpoint at the end, no clutter
        env_kind=env_kind,
    )
    # Anneal epsilon over the first ~80% of training so the schedule tracks the
    # run length rather than a fixed 20k steps (no-op on agents without epsilon).
    if hasattr(cfg, "epsilon_decay_steps"):
        # setattr (not attribute assignment) since cfg is a union of agent
        # Configs and only the epsilon agents declare this field.
        setattr(cfg, "epsilon_decay_steps", max(1, int(0.8 * total_steps)))
    if device:
        cfg.device = device
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg, run_training


def _run_one(exp: ExperimentConfig, agent: str, size: int, seed: int) -> str:
    """Train (or skip) a single ``agent``/``size``/``seed`` run; return a status.

    Self-contained so it can run in a worker process: it builds its own env,
    recorder and config, trains, and writes the ``.npz``. Returns a one-line
    status string the parent prints (kept here so prints don't interleave
    mid-line across workers).
    """
    runs_dir = Path(exp.results_dir) / "runs"
    ckpt_dir = str(Path(exp.results_dir) / "checkpoints")
    out = runs_dir / agent / f"n{size}_seed{seed}.npz"
    label = f"{agent} n={size} seed={seed}"
    if out.exists():
        return f"{label}: exists, skipping"

    cfg, run_training = _build(
        agent,
        size=size,
        total_steps=exp.total_steps,
        seed=seed,
        checkpoint_dir=ckpt_dir,
        run_name=f"{agent}_n{size}_s{seed}",
        device=exp.device,
        overrides=exp.overrides,
        env_kind=exp.env_kind,
    )
    # Build a matching env (deterministic given the cfg) so the recorder gets the
    # env's grid masks + reference return; run_training rebuilds its own instance.
    env = make_env(cfg)
    recorder = RunRecorder(
        size=size,
        total_steps=exp.total_steps,
        snapshot_every=exp.snapshot_every,
        agent=agent,
        seed=seed,
        always_right_return=env.reference_return,
        reachable_mask=env.reachable_mask(size),
        goal_mask=env.goal_mask(size),
    )
    run_training(cfg, recorder)
    recorder.save(out)
    return f"{label}: saved {out}"


def run_experiment(exp: ExperimentConfig) -> Path:
    """Run the full sweep, writing per-run records and a manifest.

    Returns the results directory. Existing run records are skipped, so an
    interrupted sweep can be resumed by re-running with the same config. With
    ``exp.workers > 1`` the independent runs train concurrently in worker
    processes (sharing one GPU), otherwise they run sequentially as before.
    """
    results = Path(exp.results_dir)
    start = time.time()

    jobs = [
        (agent, size, seed)
        for agent in exp.agents
        for size in exp.sizes
        for seed in exp.seeds
    ]
    total_runs = len(jobs)
    print(
        f"experiment[{exp.env_kind}]: {len(exp.agents)} agents x "
        f"{len(exp.sizes)} sizes x {len(exp.seeds)} seeds = {total_runs} runs, "
        f"{exp.total_steps} steps each, workers={exp.workers}"
    )

    if exp.workers <= 1:
        for done, (agent, size, seed) in enumerate(jobs, 1):
            print(f"[{done}/{total_runs}] {_run_one(exp, agent, size, seed)}")
    else:
        # spawn (not fork) so each worker initialises CUDA cleanly in its own
        # process; results then come back in completion order.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=exp.workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_run_one, exp, agent, size, seed): (agent, size, seed)
                for (agent, size, seed) in jobs
            }
            # Wait in short windows so a long-running batch still prints a sign of
            # life (the per-run step logs live in the worker processes and don't
            # reach here). Completion lines are unchanged.
            pending = set(futures)
            done = 0
            while pending:
                finished, pending = wait(
                    pending, timeout=30, return_when=FIRST_COMPLETED
                )
                for future in finished:
                    done += 1
                    print(f"[{done}/{total_runs}] {future.result()}")
                if pending and not finished:
                    print(
                        f"    ... still training: {done}/{total_runs} runs done "
                        f"({time.time() - start:.0f}s elapsed)"
                    )

    manifest = {
        "agents": list(exp.agents),
        "env_kind": exp.env_kind,
        "sizes": list(exp.sizes),
        "seeds": list(exp.seeds),
        "total_steps": exp.total_steps,
        "config": asdict(exp),
        "elapsed_seconds": time.time() - start,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"experiment done in {time.time() - start:.0f}s -> {results}")
    return results
