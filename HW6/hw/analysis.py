"""Multi-panel comparison of the DeepSea exploration agents.

Loads the per-run records written by :mod:`experiment` and turns them into the
five families of analysis the comparison is built around, writing a figure per
panel under ``<results_dir>/figures/`` plus a machine-readable ``summary.json``:

* **Performance:** return-vs-env-steps learning curves with seed CI bands,
  steps-to-threshold sample efficiency, final performance and area-under-curve.
* **Exploration:** state-coverage curve, state-visitation heatmaps, visitation
  entropy curve, and first-visit time to the treasure.
* **Mechanism:** the deep-exploration signal (how far right the agent gets over
  training) and the exploration-bonus dynamics (RND novelty bonus / bootstrap
  ensemble disagreement decaying as states become familiar).
* **Stability:** seed-variance CI bands on every curve plus a per-strategy
  variance summary of final performance.
* **Statistical rigor:** rliable-style IQM with stratified-bootstrap CIs,
  performance profiles, and probability of improvement (:mod:`rliable_stats`).

All curves are aggregated across seeds with a percentile bootstrap over seeds for
the band; the snapshot grid is shared across runs so coverage/entropy stack
directly, while episode-indexed series (return, depth, bonus) are interpolated
onto that grid first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Render to the non-interactive Agg backend for headless figure writing (we only
# ever ``savefig``). Don't override an inline backend a notebook already selected
# via ``%matplotlib inline`` -- forcing Agg there makes the notebook's own
# ``plt.show()`` figures warn that Agg "cannot be shown" and stop displaying.
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metrics import load_run  # noqa: E402
from rliable_stats import (  # noqa: E402
    interquartile_mean,
    performance_profile,
    probability_of_improvement,
    stratified_bootstrap_ci,
)

AGENT_LABELS = {
    "double_dqn": "Double DQN (ε-greedy)",
    "rnd_dqn": "RND",
    "bootstrap_dqn": "Bootstrapped DQN",
}
AGENT_COLORS = {
    "double_dqn": "tab:gray",
    "rnd_dqn": "tab:orange",
    "bootstrap_dqn": "tab:blue",
}
SOLVE_FRACTION = 0.8  # "solved" = return >= 0.8 * reference return
RETURN_SMOOTH = 25  # episodes in the causal moving average for return curves
ENV_NAMES = {"deepsea": "DeepSea", "mountain_car": "MountainCar"}


def _size_desc(env_kind: str, size: int) -> str:
    """Short env+size descriptor for figure titles (e.g. ``DeepSea N=15``)."""
    unit = "N" if env_kind == "deepsea" else "bins"
    return f"{ENV_NAMES.get(env_kind, env_kind)} {unit}={size}"


# -- loading ---------------------------------------------------------------
def load_experiment(results_dir: str | Path) -> dict[str, Any]:
    """Load the manifest and every run record into a nested structure.

    Returns ``{"manifest": ..., "runs": {(agent, size): [run, ...]}}`` with each
    run list ordered by seed.
    """
    results = Path(results_dir)
    manifest = json.loads((results / "manifest.json").read_text())
    runs: dict[tuple[str, int], list[dict[str, np.ndarray]]] = {}
    for agent in manifest["agents"]:
        for size in manifest["sizes"]:
            collected = []
            for seed in manifest["seeds"]:
                path = results / "runs" / agent / f"n{size}_seed{seed}.npz"
                if path.exists():
                    collected.append(load_run(path))
            if collected:
                runs[agent, int(size)] = collected
    return {"manifest": manifest, "runs": runs}


# -- aggregation helpers ---------------------------------------------------
def _causal_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average with a growing window at the head (no look-ahead)."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values
    cum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(values.size)
    lo = np.maximum(0, idx + 1 - window)
    return (cum[idx + 1] - cum[lo]) / (idx + 1 - lo)


def _series_on_grid(
    steps: np.ndarray, values: np.ndarray, grid: np.ndarray, smooth: int
) -> np.ndarray:
    """Smooth an episode/update-indexed series and interpolate onto ``grid``."""
    if steps.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    return np.interp(grid, steps, _causal_smooth(values, smooth))


def _bootstrap_mean_ci(
    mat: np.ndarray, *, reps: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column mean of ``(seeds, T)`` with a percentile bootstrap CI band."""
    mat = np.asarray(mat, dtype=np.float64)
    mean = mat.mean(axis=0)
    n = mat.shape[0]
    if n < 2:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot = mat[idx].mean(axis=1)  # (reps, T)
    alpha = (1.0 - confidence) / 2.0
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1.0 - alpha, axis=0)
    return mean, lo, hi


def _grid(runs: list[dict[str, np.ndarray]]) -> np.ndarray:
    """The shared snapshot env-step grid (identical across runs)."""
    return runs[0]["snap_steps"].astype(np.float64)


def _return_matrix(runs: list[dict[str, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    """``(seeds, T)`` smoothed episode return on the shared grid."""
    return np.vstack(
        [
            _series_on_grid(r["ep_steps"], r["ep_returns"], grid, RETURN_SMOOTH)
            for r in runs
        ]
    )


def _depth_matrix(
    runs: list[dict[str, np.ndarray]], grid: np.ndarray, size: int
) -> np.ndarray:
    """``(seeds, T)`` normalised deepest column reached per episode."""
    return np.vstack(
        [
            _series_on_grid(
                r["ep_steps"], r["ep_max_col"] / max(size - 1, 1), grid, RETURN_SMOOTH
            )
            for r in runs
        ]
    )


def _snap_matrix(runs: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Stack a per-run snapshot series (already on the shared grid)."""
    return np.vstack([r[key] for r in runs])


def _scalar(run: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(run[key]).item())


def _final_perf(runs: list[dict[str, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    """Per-seed final normalised performance (mean of last 10% of the curve)."""
    ret = _return_matrix(runs, grid)
    tail = max(1, ret.shape[1] // 10)
    always_right = _scalar(runs[0], "always_right_return")
    return ret[:, -tail:].mean(axis=1) / always_right


# -- plotting --------------------------------------------------------------
def _band(ax, grid, mat, agent, *, label_suffix=""):
    mean, lo, hi = _bootstrap_mean_ci(mat)
    color = AGENT_COLORS[agent]
    ax.plot(grid, mean, color=color, lw=2, label=AGENT_LABELS[agent] + label_suffix)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.2)
    return mean


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")


def _performance(runs_by_agent, grid, desc, always_right, figdir) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = {}
    for agent, runs in runs_by_agent.items():
        mat = _return_matrix(runs, grid)
        _band(ax, grid, mat, agent)
        summary[agent] = mat
    ax.axhline(always_right, ls="--", color="k", lw=1, alpha=0.6, label="reference")
    ax.axhline(
        SOLVE_FRACTION * always_right,
        ls=":",
        color="k",
        lw=1,
        alpha=0.4,
        label=f"solve thresh ({SOLVE_FRACTION:g}x)",
    )
    ax.set_xlabel("env steps")
    ax.set_ylabel("episode return (smoothed)")
    ax.set_title(f"Performance: return vs env-steps  ({desc}, seed CI)")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, figdir / "performance_learning_curves.png")

    # Sample efficiency + AUC + final performance bars.
    thr = SOLVE_FRACTION * always_right
    agents = list(runs_by_agent)
    metrics: dict[str, dict] = {}
    for agent in agents:
        mat = summary[agent]
        steps_to: list[float] = []
        for row in mat:
            hit = np.argmax(row >= thr) if (row >= thr).any() else -1
            steps_to.append(float(grid[hit]) if hit >= 0 else float("inf"))
        auc = np.trapezoid(mat / always_right, grid, axis=1) / (grid[-1] - grid[0])
        final = _final_perf(runs_by_agent[agent], grid)
        n_solved = int(np.isfinite(steps_to).sum())
        finite = [s for s in steps_to if np.isfinite(s)]
        metrics[agent] = {
            "steps_to_threshold": [s if np.isfinite(s) else None for s in steps_to],
            "median_steps_to_threshold": float(np.median(finite)) if finite else None,
            "n_solved": n_solved,
            "n_seeds": len(steps_to),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std()),
            "final_perf_mean": float(final.mean()),
            "final_perf_std": float(final.std()),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(agents))
    colors = [AGENT_COLORS[a] for a in agents]
    axes[0].bar(
        x,
        [metrics[a]["auc_mean"] for a in agents],
        yerr=[metrics[a]["auc_std"] for a in agents],
        color=colors,
        capsize=4,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([AGENT_LABELS[a] for a in agents], rotation=15, fontsize=8)
    axes[0].set_ylabel("normalised area under return curve")
    axes[0].set_title("Performance: sample efficiency (AUC)")
    for i, a in enumerate(agents):
        m = metrics[a]
        axes[0].text(
            i,
            metrics[a]["auc_mean"],
            f"  {m['n_solved']}/{m['n_seeds']} solved",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].bar(
        x,
        [metrics[a]["final_perf_mean"] for a in agents],
        yerr=[metrics[a]["final_perf_std"] for a in agents],
        color=colors,
        capsize=4,
    )
    axes[1].axhline(1.0, ls="--", color="k", lw=1, alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGENT_LABELS[a] for a in agents], rotation=15, fontsize=8)
    axes[1].set_ylabel("final performance (/ always-right)")
    axes[1].set_title("Performance: final performance")
    _save(fig, figdir / "performance_sample_efficiency.png")
    return metrics


def _exploration(runs_by_agent, grid, desc, figdir) -> dict:
    # Coverage curve.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent, runs in runs_by_agent.items():
        _band(ax, grid, _snap_matrix(runs, "snap_coverage"), agent)
    ax.set_xlabel("env steps")
    ax.set_ylabel("fraction of reachable cells visited")
    ax.set_title(f"Exploration: state coverage vs env-steps  ({desc})")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, figdir / "exploration_coverage.png")

    # Entropy curve.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent, runs in runs_by_agent.items():
        _band(ax, grid, _snap_matrix(runs, "snap_entropy"), agent)
    ax.set_xlabel("env steps")
    ax.set_ylabel("visitation entropy (bits)")
    ax.set_title(f"Exploration: visitation entropy vs env-steps  ({desc})")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, figdir / "exploration_entropy.png")

    # Visitation heatmaps (seed-summed, log-scaled, unreachable cells masked).
    # The reachable + goal masks are stored per run (env-supplied), so this is
    # the same picture for any environment's grid view.
    agents = list(runs_by_agent)
    rep = next(iter(runs_by_agent.values()))[0]
    mask, goal = _run_masks(rep)
    fig, axes = plt.subplots(1, len(agents), figsize=(4.2 * len(agents), 4.2))
    if len(agents) == 1:
        axes = [axes]
    for ax, agent in zip(axes, agents):
        counts = np.sum(
            [r["visit_counts"] for r in runs_by_agent[agent]], axis=0
        ).astype(np.float64)
        shown = np.where(mask, np.log1p(counts), np.nan)
        im = ax.imshow(shown, cmap="viridis", origin="upper")
        for gr, gc in np.argwhere(goal):
            ax.add_patch(
                plt.Rectangle(
                    (gc - 0.5, gr - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.5
                )
            )
        ax.set_title(AGENT_LABELS[agent], fontsize=9)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, label="log(1+visits)")
    fig.suptitle(f"Exploration: state-visitation heatmaps  ({desc}, seeds summed)")
    _save(fig, figdir / "exploration_heatmaps.png")

    # First-visit time to the goal (treasure / reaching the hilltop).
    metrics: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(agents))
    means, errs, annots = [], [], []
    for agent in agents:
        fvs = [_first_visit_goal(r) for r in runs_by_agent[agent]]
        reached = [v for v in fvs if v is not None]
        metrics[agent] = {
            "first_visit_goal": [(-1 if v is None else v) for v in fvs],
            "n_reached_goal": len(reached),
            "mean_first_visit": float(np.mean(reached)) if reached else None,
        }
        means.append(np.mean(reached) if reached else 0.0)
        errs.append(np.std(reached) if len(reached) > 1 else 0.0)
        annots.append(f"{len(reached)}/{len(fvs)}")
    ax.bar(x, means, yerr=errs, color=[AGENT_COLORS[a] for a in agents], capsize=4)
    for i, txt in enumerate(annots):
        ax.text(i, means[i], f"  {txt} reached", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents], rotation=15, fontsize=8)
    ax.set_ylabel("env step of first goal visit")
    ax.set_title(f"Exploration: first-visit time to goal  ({desc})")
    _save(fig, figdir / "exploration_first_visit.png")
    return metrics


def _run_masks(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(reachable_mask, goal_mask)`` for a run, with a legacy DeepSea fallback.

    Run records now carry the env-supplied masks; records predating that
    (older DeepSea sweeps) fall back to the lower-triangle reachable mask and a
    single bottom-right treasure cell.
    """
    if "reachable_mask" in run:
        return (
            np.asarray(run["reachable_mask"], dtype=bool),
            np.asarray(run["goal_mask"], dtype=bool),
        )
    from metrics import _default_goal_mask, reachable_mask

    size = int(np.asarray(run["size"]).item())
    return reachable_mask(size), _default_goal_mask(size)


def _first_visit_goal(run: dict[str, np.ndarray]) -> float | None:
    if "goal_step" in run:
        gs = int(np.asarray(run["goal_step"]).item())
    else:  # legacy DeepSea record: derive from the treasure cell's first visit.
        size = int(np.asarray(run["size"]).item())
        gs = int(run["first_visit"][size - 1, size - 1])
    return None if gs < 0 else float(gs)


def _mechanism(runs_by_agent, grid, size, desc, figdir) -> dict:
    # Deep-exploration signal: normalised deepest column reached over training.
    # For DeepSea this is how far right the agent gets; for Mountain Car it is the
    # closest position bin to the goal.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics: dict[str, dict] = {}
    for agent, runs in runs_by_agent.items():
        mat = _depth_matrix(runs, grid, size)
        mean = _band(ax, grid, mat, agent)
        metrics[agent] = {"final_depth_frac": float(mean[-1])}
    ax.axhline(1.0, ls="--", color="k", lw=1, alpha=0.5, label="goal column reached")
    ax.set_xlabel("env steps")
    ax.set_ylabel("deepest column reached / (size-1)")
    ax.set_title(f"Mechanism: deep-exploration signal  ({desc})")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, figdir / "mechanism_deep_reach.png")

    # Exploration-bonus dynamics: RND novelty bonus and bootstrap disagreement.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bonus_agents = {
        "rnd_dqn": (axes[0], "normalised RND novelty bonus", "RND novelty bonus decay"),
        "bootstrap_dqn": (
            axes[1],
            "ensemble Q disagreement (std)",
            "Bootstrap disagreement decay",
        ),
    }
    for agent, (ax, ylabel, title) in bonus_agents.items():
        if agent not in runs_by_agent:
            ax.set_visible(False)
            continue
        runs = runs_by_agent[agent]
        mat = np.vstack(
            [
                _series_on_grid(r["intr_steps"], r["intr_values"], grid, 200)
                for r in runs
            ]
        )
        _band(ax, grid, mat, agent)
        ax.set_xlabel("env steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mechanism: {title}")
        ax.legend(fontsize=8)
    _save(fig, figdir / "mechanism_intrinsic.png")
    return metrics


def _stability(runs_by_agent, grid, figdir) -> dict:
    agents = list(runs_by_agent)
    metrics: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(agents))
    stds = []
    for agent in agents:
        final = _final_perf(runs_by_agent[agent], grid)
        stds.append(float(final.std()))
        metrics[agent] = {
            "final_perf_std": float(final.std()),
            "final_perf_iqr": float(np.subtract(*np.percentile(final, [75, 25]))),
            "per_seed_final_perf": final.tolist(),
        }
    ax.bar(x, stds, color=[AGENT_COLORS[a] for a in agents])
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in agents], rotation=15, fontsize=8)
    ax.set_ylabel("std of final performance across seeds")
    ax.set_title("Stability: cross-seed stability (lower = more reliable)")
    _save(fig, figdir / "stability_variance.png")
    return metrics


def _rigor(scores_by_agent, figdir) -> dict:
    agents = list(scores_by_agent)
    metrics: dict[str, dict] = {}

    # IQM with stratified-bootstrap CI.
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(agents) + 1.5))
    for i, agent in enumerate(agents):
        point, lo, hi = stratified_bootstrap_ci(
            scores_by_agent[agent], interquartile_mean
        )
        metrics[agent] = {"iqm": point, "iqm_ci": [lo, hi]}
        ax.errorbar(
            point,
            i,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=AGENT_COLORS[agent],
            capsize=5,
            ms=8,
        )
    ax.axvline(1.0, ls="--", color="k", lw=1, alpha=0.5)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([AGENT_LABELS[a] for a in agents])
    ax.set_xlabel("IQM of final performance (/ always-right)")
    ax.set_title("Statistical rigor: IQM with 95% stratified-bootstrap CI")
    _save(fig, figdir / "rigor_iqm.png")

    # Performance profiles.
    taus = np.linspace(0.0, 1.1, 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in agents:
        prof, lo, hi = performance_profile(scores_by_agent[agent], taus)
        ax.plot(taus, prof, color=AGENT_COLORS[agent], lw=2, label=AGENT_LABELS[agent])
        ax.fill_between(taus, lo, hi, color=AGENT_COLORS[agent], alpha=0.2)
    ax.set_xlabel("normalised performance τ")
    ax.set_ylabel("fraction of runs with score > τ")
    ax.set_title("Statistical rigor: performance profiles")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, figdir / "rigor_performance_profiles.png")

    # Probability of improvement for each ordered pair.
    pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i + 1 :]]
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(pairs) + 1.5))
    poi: dict[str, dict] = {}
    for j, (a, b) in enumerate(pairs):
        point, lo, hi = probability_of_improvement(
            scores_by_agent[a], scores_by_agent[b]
        )
        poi[f"{a}>{b}"] = {"p": point, "ci": [lo, hi]}
        ax.errorbar(
            point,
            j,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            capsize=5,
            ms=8,
            color="tab:purple",
        )
    ax.axvline(0.5, ls="--", color="k", lw=1, alpha=0.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(
        [f"{AGENT_LABELS[a]}\n> {AGENT_LABELS[b]}" for a, b in pairs], fontsize=7
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(X > Y)")
    ax.set_title("Statistical rigor: probability of improvement")
    _save(fig, figdir / "rigor_prob_improvement.png")
    metrics["probability_of_improvement"] = poi
    return metrics


# -- entrypoint ------------------------------------------------------------
def analyze(results_dir: str | Path = "results") -> dict:
    """Run all five analyses, write figures + ``summary.json``, return the summary."""
    results = Path(results_dir)
    data = load_experiment(results)
    manifest = data["manifest"]
    runs = data["runs"]
    sizes = [int(s) for s in manifest["sizes"]]
    env_kind = manifest.get("env_kind", "deepsea")
    figdir = results / "figures"

    summary: dict[str, Any] = {"manifest": manifest, "per_size": {}, "rigor": {}}

    # Per-size analyses (use the primary/first size for the curve figures).
    for size in sizes:
        runs_by_agent = {
            agent: runs[agent, size]
            for agent in manifest["agents"]
            if (agent, size) in runs
        }
        if not runs_by_agent:
            continue
        any_runs = next(iter(runs_by_agent.values()))
        grid = _grid(any_runs)
        always_right = _scalar(any_runs[0], "always_right_return")
        size_figdir = figdir if len(sizes) == 1 else figdir / f"n{size}"

        desc = _size_desc(env_kind, size)
        print(f"analysing {desc} ({len(runs_by_agent)} agents)...")
        summary["per_size"][str(size)] = {
            "performance": _performance(
                runs_by_agent, grid, desc, always_right, size_figdir
            ),
            "exploration": _exploration(runs_by_agent, grid, desc, size_figdir),
            "mechanism": _mechanism(runs_by_agent, grid, size, desc, size_figdir),
            "stability": _stability(runs_by_agent, grid, size_figdir),
        }

    # Statistical rigor: scores aggregated across all sizes (tasks) x seeds.
    scores_by_agent: dict[str, np.ndarray] = {}
    for agent in manifest["agents"]:
        cols = []
        for size in sizes:
            if (agent, size) in runs:
                run_list = runs[agent, size]
                grid = _grid(run_list)
                cols.append(_final_perf(run_list, grid))
        if cols:
            # (seeds, tasks); requires equal seed counts per size (true by design).
            scores_by_agent[agent] = np.column_stack(cols)
    if scores_by_agent:
        summary["rigor"] = _rigor(scores_by_agent, figdir)

    (results / "summary.json").write_text(_json_safe(summary))
    print(f"summary -> {results / 'summary.json'}")
    return summary


def _json_safe(obj: Any) -> str:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if o == float("inf"):
            return "inf"
        return str(o)

    return json.dumps(obj, indent=2, default=default)
