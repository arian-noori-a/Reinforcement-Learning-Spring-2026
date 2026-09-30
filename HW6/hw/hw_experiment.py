"""Drive the normal-vs-noisy DeepSea comparison for the homework.

A thin wrapper over :mod:`experiment` that runs the three-agent sweep
(Double DQN + RND + Bootstrapped DQN) twice -- once on ordinary DeepSea and once
on the noisy-TV variant -- writing the two sweeps under ``results/normal`` and
``results/noisy`` and then running :func:`analysis.analyze` on each so the
performance / exploration / mechanism / stability / rigor figures +
``summary.json`` are ready for the plotting cells. Double DQN is the plain
epsilon-greedy baseline the two deep-exploration agents are measured against.

The sweep reuses :func:`experiment.run_experiment` unchanged, so it is resumable
(existing run records are skipped) and can train runs concurrently in worker
processes via ``workers`` (``> 1`` on a GPU shares it across runs).
"""

from __future__ import annotations

import time

import analysis
from experiment import ExperimentConfig, run_experiment

HW_AGENTS: tuple[str, ...] = ("double_dqn", "rnd_dqn", "bootstrap_dqn")


def run_full_comparison(
    *,
    size: int = 10,
    seeds: tuple[int, ...] = (0, 1, 2),
    total_steps: int = 30_000,
    device: str | None = None,
    workers: int = 2,
    num_heads: int = 5,
    prior_scale: float = 3.0,
    mask_prob: float = 0.5,
    intrinsic_coef: float = 1.0,
    rnd_output_dim: int = 64,
    noise_dim: int = 20,
    noise_columns: tuple[int, ...] = (0,),
    results_dir: str = "results",
) -> None:
    """Run the Double-DQN/RND/Bootstrap sweep on normal and noisy DeepSea, then analyse.

    Both variants share the per-agent knobs (``num_heads`` / ``prior_scale`` /
    ``mask_prob`` reach Bootstrapped DQN; ``intrinsic_coef`` / ``rnd_output_dim``
    reach RND -- each override is applied only to the agent whose Config declares
    it). The noisy variant additionally turns on the localized noisy-TV features
    (``noise_dim`` random features active only in ``noise_columns``).

    Writes ``<results_dir>/normal`` and ``<results_dir>/noisy`` and calls
    :func:`analysis.analyze` on each. Idempotent / resumable.
    """
    start = time.time()

    # Per-agent knobs shared by both variants. Each key is applied only to the
    # agents whose Config declares it (see ``experiment._build``).
    agent_overrides = {
        "num_heads": num_heads,
        "prior_scale": prior_scale,
        "mask_prob": mask_prob,
        "intrinsic_coef": intrinsic_coef,
        "rnd_output_dim": rnd_output_dim,
    }

    variants = {
        "normal": dict(agent_overrides),
        "noisy": {
            **agent_overrides,
            "ds_noise_dim": noise_dim,
            "ds_noise_columns": noise_columns,
        },
    }

    for variant, overrides in variants.items():
        print(f"\n===== {variant.upper()} DeepSea (N={size}) =====")
        exp = ExperimentConfig(
            agents=HW_AGENTS,
            env_kind="deepsea",
            sizes=(size,),
            seeds=tuple(seeds),
            total_steps=total_steps,
            results_dir=f"{results_dir}/{variant}",
            device=device,
            workers=workers,
            overrides=overrides,
        )
        run_experiment(exp)

    print("\n===== ANALYSIS =====")
    for variant in variants:
        analysis.analyze(f"{results_dir}/{variant}")

    print(f"\nfull comparison done in {time.time() - start:.0f}s")
