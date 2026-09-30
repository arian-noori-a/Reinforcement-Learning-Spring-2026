# ============================================================
# Moving-Target Reacher Environment
# ============================================================

import inspect
from typing import Callable, Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
from collections import deque
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "This code requires gymnasium. Install with:\n"
        "pip install gymnasium[mujoco]"
    ) from e


ArrayLike = np.ndarray

# ============================================================
# Target trajectory functions
# ============================================================

def circular_target(
    radius: float = 0.15,
    omega: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
):
    """
    Returns target_fn(t), target_vel_fn(t) for circular target motion.

    x(t) = cx + R cos(omega t)
    y(t) = cy + R sin(omega t)
    """

    cx, cy = center

    def target_fn(t: float) -> np.ndarray:
        return np.array(
            [
                cx + radius * np.cos(omega * t),
                cy + radius * np.sin(omega * t),
            ],
            dtype=np.float32,
        )

    def target_vel_fn(t: float) -> np.ndarray:
        return np.array(
            [
                -radius * omega * np.sin(omega * t),
                radius * omega * np.cos(omega * t),
            ],
            dtype=np.float32,
        )

    return target_fn, target_vel_fn


def lissajous_target(
    amp_x: float = 0.15,
    amp_y: float = 0.15,
    omega_x: float = 1.0,
    omega_y: float = 2.0,
    phase_y: float = np.pi / 2,
    center: Tuple[float, float] = (0.0, 0.0),
):
    """
    Returns target_fn(t), target_vel_fn(t) for Lissajous target motion.

    x(t) = cx + A_x sin(omega_x t)
    y(t) = cy + A_y sin(omega_y t + phase_y)
    """

    cx, cy = center

    def target_fn(t: float) -> np.ndarray:
        return np.array(
            [
                cx + amp_x * np.sin(omega_x * t),
                cy + amp_y * np.sin(omega_y * t + phase_y),
            ],
            dtype=np.float32,
        )

    def target_vel_fn(t: float) -> np.ndarray:
        return np.array(
            [
                amp_x * omega_x * np.cos(omega_x * t),
                amp_y * omega_y * np.cos(omega_y * t + phase_y),
            ],
            dtype=np.float32,
        )

    return target_fn, target_vel_fn


def static_target(
    xy: Tuple[float, float] = (0.1, 0.1),
):
    """
    Returns target_fn(t), target_vel_fn(t) for a fixed target.
    """

    xy = np.array(xy, dtype=np.float32)

    def target_fn(t: float) -> np.ndarray:
        return xy.copy()

    def target_vel_fn(t: float) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    return target_fn, target_vel_fn


# ============================================================
# Moving-Target Reacher Wrapper
# ============================================================

class MovingTargetReacher(gym.Wrapper):
    """
    Gymnasium wrapper for Reacher with a target following a user-defined
    function of time.

    The wrapper modifies the target position at every step and recomputes
    the reward using the moving target.

    The target trajectory is defined by

        target_fn(t) -> np.ndarray shape (2,)

    and optionally

        target_vel_fn(t) -> np.ndarray shape (2,)

    If target_vel_fn is not provided, velocity is estimated by finite
    differences.

    Important:
        This wrapper assumes a MuJoCo Reacher-like environment where the
        fingertip body is named "fingertip" and the target body is named
        "target". It also attempts to set target qpos[2:4], which matches
        common Gymnasium/MuJoCo Reacher implementations.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_id: str = "Reacher-v5",
        target_fn: Optional[Callable[[float], ArrayLike]] = None,
        target_vel_fn: Optional[Callable[[float], ArrayLike]] = None,
        include_base_obs: bool = True,
        include_target_pos: bool = True,
        include_target_vel: bool = True,
        include_time: bool = False,
        include_phase: bool = False,
        phase_omega: float = 1.0,
        distance_weight: float = 1.0,
        control_weight: float = 0.01,
        squared_distance: bool = False,
        max_episode_steps: Optional[int] = None,
        finite_diff_eps: float = 1e-4,
        render_mode: Optional[str] = None,
        **gym_kwargs,
    ):
        if max_episode_steps is not None:
            gym_kwargs["max_episode_steps"] = max_episode_steps
        
        env = gym.make(env_id, render_mode=render_mode, **gym_kwargs)
        super().__init__(env)

        if target_fn is None:
            target_fn, target_vel_fn_default = circular_target(
                radius=0.15,
                omega=1.0,
                center=(0.0, 0.0),
            )
            if target_vel_fn is None:
                target_vel_fn = target_vel_fn_default

        self.target_fn = target_fn
        self.target_vel_fn = target_vel_fn
        self.include_base_obs = include_base_obs
        self.include_target_pos = include_target_pos
        self.include_target_vel = include_target_vel
        self.include_time = include_time
        self.include_phase = include_phase
        self.phase_omega = phase_omega

        self.distance_weight = distance_weight
        self.control_weight = control_weight
        self.squared_distance = squared_distance
        self.max_episode_steps = max_episode_steps
        self.finite_diff_eps = finite_diff_eps

        self.t = 0.0
        self.step_count = 0

        self.dt = self._infer_dt()

        # Keep the original action space.
        self.action_space = self.env.action_space

        # Build augmented observation space.
        base_dim = int(np.prod(self.env.observation_space.shape))
        extra_dim = 0

        if self.include_target_pos:
            extra_dim += 2
        if self.include_target_vel:
            extra_dim += 2
        if self.include_time:
            extra_dim += 1
        if self.include_phase:
            extra_dim += 2

        if self.include_base_obs:
            obs_dim = base_dim + extra_dim
        else:
            obs_dim = extra_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Time and target helpers
    # --------------------------------------------------------

    def _infer_dt(self) -> float:
        """
        Infer MuJoCo environment time step if possible.
        """
        unwrapped = self.env.unwrapped

        if hasattr(unwrapped, "dt"):
            return float(unwrapped.dt)

        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "opt"):
            timestep = float(unwrapped.model.opt.timestep)
            frame_skip = getattr(unwrapped, "frame_skip", 1)
            return timestep * frame_skip

        return 0.02

    def _call_target_fn(self, fn: Callable, t: float) -> np.ndarray:
        """
        Calls target function. Supports both target_fn(t) and target_fn(t, step).
        """
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 2:
                out = fn(t, self.step_count)
            else:
                out = fn(t)
        except (TypeError, ValueError):
            out = fn(t)

        out = np.asarray(out, dtype=np.float32).reshape(-1)

        if out.shape != (2,):
            raise ValueError(
                f"Target function must return shape (2,), got {out.shape}."
            )

        return out

    def target_position(self, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.t
        return self._call_target_fn(self.target_fn, t)

    def target_velocity(self, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = self.t

        if self.target_vel_fn is not None:
            vel = self._call_target_fn(self.target_vel_fn, t)
            return vel.astype(np.float32)

        # Finite-difference estimate.
        eps = self.finite_diff_eps
        p_plus = self.target_position(t + eps)
        p_minus = self.target_position(t - eps)
        return ((p_plus - p_minus) / (2.0 * eps)).astype(np.float32)

    # --------------------------------------------------------
    # MuJoCo body / state helpers
    # --------------------------------------------------------

    def _get_body_xy(self, body_name: str) -> np.ndarray:
        """
        Robustly retrieves MuJoCo body xy position.
        """
        unwrapped = self.env.unwrapped

        # Gymnasium MuJoCo new API.
        try:
            return np.asarray(unwrapped.data.body(body_name).xpos[:2], dtype=np.float32)
        except Exception:
            pass

        # Older mujoco-py API.
        try:
            return np.asarray(unwrapped.data.get_body_xpos(body_name)[:2], dtype=np.float32)
        except Exception:
            pass

        raise RuntimeError(
            f"Could not retrieve body position for body '{body_name}'. "
            "Check that this is a Reacher-like MuJoCo environment."
        )

    def _set_target_xy(self, xy: np.ndarray, vel: Optional[np.ndarray] = None):
        """
        Attempts to set the MuJoCo target position.

        In common Reacher implementations, qpos[2:4] represents target x/y.
        This method updates qpos[2:4] and optionally qvel[2:4].
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        if vel is None:
            vel = np.zeros(2, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64).reshape(2)

        unwrapped = self.env.unwrapped

        if not hasattr(unwrapped, "data"):
            return

        qpos = np.array(unwrapped.data.qpos).copy()
        qvel = np.array(unwrapped.data.qvel).copy()

        if qpos.shape[0] >= 4:
            qpos[2:4] = xy

        if qvel.shape[0] >= 4:
            qvel[2:4] = vel

        # Gymnasium MuJoCo envs usually expose set_state.
        if hasattr(unwrapped, "set_state"):
            unwrapped.set_state(qpos, qvel)
        else:
            unwrapped.data.qpos[:] = qpos
            unwrapped.data.qvel[:] = qvel

        # Also try to move the target body/site visual marker if accessible.
        # This is not always necessary, because qpos[2:4] usually controls it.
        try:
            unwrapped.model.body("target").pos[:2] = xy
        except Exception:
            pass

        try:
            unwrapped.model.site("target").pos[:2] = xy
        except Exception:
            pass

    def _get_base_obs(self, fallback_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieves the underlying Reacher observation after target update.
        """
        unwrapped = self.env.unwrapped

        if hasattr(unwrapped, "_get_obs"):
            try:
                obs = unwrapped._get_obs()
                return np.asarray(obs, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        if fallback_obs is not None:
            return np.asarray(fallback_obs, dtype=np.float32).reshape(-1)

        raise RuntimeError("Could not retrieve base observation.")

    def _get_augmented_obs(self, fallback_obs: Optional[np.ndarray] = None) -> np.ndarray:
        parts = []

        if self.include_base_obs:
            parts.append(self._get_base_obs(fallback_obs))

        if self.include_target_pos:
            parts.append(self.target_position(self.t))

        if self.include_target_vel:
            parts.append(self.target_velocity(self.t))

        if self.include_time:
            parts.append(np.array([self.t], dtype=np.float32))

        if self.include_phase:
            phase = self.phase_omega * self.t
            parts.append(
                np.array(
                    [np.sin(phase), np.cos(phase)],
                    dtype=np.float32,
                )
            )

        if len(parts) == 0:
            raise ValueError("Observation cannot be empty. Enable at least one observation component.")

        return np.concatenate(parts, axis=0).astype(np.float32)

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        fingertip_xy = self._get_body_xy("fingertip")
        target_xy = self.target_position(self.t)

        diff = fingertip_xy - target_xy
        dist = float(np.linalg.norm(diff))

        if self.squared_distance:
            reward_dist = -self.distance_weight * float(np.sum(diff ** 2))
        else:
            reward_dist = -self.distance_weight * dist

        action = np.asarray(action, dtype=np.float32)
        reward_ctrl = -self.control_weight * float(np.sum(np.square(action)))

        reward = reward_dist + reward_ctrl

        info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "distance": dist,
            "target_x": float(target_xy[0]),
            "target_y": float(target_xy[1]),
            "fingertip_x": float(fingertip_xy[0]),
            "fingertip_y": float(fingertip_xy[1]),
            "time": float(self.t),
        }

        return float(reward), info

    # --------------------------------------------------------
    # Gymnasium API
    # --------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        base_obs, info = self.env.reset(seed=seed, options=options)

        self.t = 0.0
        self.step_count = 0

        target_xy = self.target_position(self.t)
        target_vel = self.target_velocity(self.t)
        self._set_target_xy(target_xy, target_vel)

        obs = self._get_augmented_obs(fallback_obs=base_obs)

        info = dict(info)
        info.update(
            {
                "target_x": float(target_xy[0]),
                "target_y": float(target_xy[1]),
                "target_vx": float(target_vel[0]),
                "target_vy": float(target_vel[1]),
                "time": float(self.t),
            }
        )

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Ensure target is at current time before physics step.
        self._set_target_xy(
            self.target_position(self.t),
            self.target_velocity(self.t),
        )

        base_obs, _, terminated, truncated, env_info = self.env.step(action)

        self.step_count += 1
        self.t = self.step_count * self.dt

        # Move target to its new position after the physics step.
        self._set_target_xy(
            self.target_position(self.t),
            self.target_velocity(self.t),
        )

        obs = self._get_augmented_obs(fallback_obs=base_obs)
        reward, reward_info = self._compute_reward(action)

        if self.max_episode_steps is not None and self.step_count >= self.max_episode_steps:
            truncated = True

        info = dict(env_info)
        info.update(reward_info)
        info.update(
            {
                "target_vx": float(self.target_velocity(self.t)[0]),
                "target_vy": float(self.target_velocity(self.t)[1]),
            }
        )

        return obs, reward, terminated, truncated, info

    def render(self):
        # Ensure rendered target is synchronized.
        self._set_target_xy(
            self.target_position(self.t),
            self.target_velocity(self.t),
        )
        return self.env.render()
    
# ============================================================
# Convenience factory
# ============================================================

def make_moving_reacher(
    motion: str = "circle",
    env_id: str = "Reacher-v5",
    render_mode: Optional[str] = None,
    max_episode_steps: int = 100,
    **kwargs,
) -> MovingTargetReacher:
    """
    Convenience constructor.

    motion:
        "static"
        "circle"
        "lissajous"
    """

    if motion == "static":
        target_fn, target_vel_fn = static_target(xy=kwargs.pop("xy", (0.1, 0.1)))

    elif motion == "circle":
        target_fn, target_vel_fn = circular_target(
            radius=kwargs.pop("radius", 0.15),
            omega=kwargs.pop("omega", 1.0),
            center=kwargs.pop("center", (0.0, 0.0)),
        )

    elif motion == "lissajous":
        target_fn, target_vel_fn = lissajous_target(
            amp_x=kwargs.pop("amp_x", 0.15),
            amp_y=kwargs.pop("amp_y", 0.15),
            omega_x=kwargs.pop("omega_x", 1.0),
            omega_y=kwargs.pop("omega_y", 2.0),
            phase_y=kwargs.pop("phase_y", np.pi / 2),
            center=kwargs.pop("center", (0.0, 0.0)),
        )

    else:
        raise ValueError(f"Unknown motion type: {motion}")

    env = MovingTargetReacher(
        env_id=env_id,
        target_fn=target_fn,
        target_vel_fn=target_vel_fn,
        include_base_obs=kwargs.pop("include_base_obs", True),
        include_target_pos=kwargs.pop("include_target_pos", True),
        include_target_vel=kwargs.pop("include_target_vel", True),
        include_time=kwargs.pop("include_time", False),
        include_phase=kwargs.pop("include_phase", True),
        phase_omega=kwargs.pop("phase_omega", 1.0),
        distance_weight=kwargs.pop("distance_weight", 1.0),
        control_weight=kwargs.pop("control_weight", 0.01),
        squared_distance=kwargs.pop("squared_distance", False),
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        **kwargs,
    )

    return env

# ============================================================
# No-render video recorder for MovingTargetReacher
# No env.render(), no OpenGL, no MuJoCo renderer.
# Produces a polished visualization with trails and overlays.
# ============================================================
# ------------------------------------------------------------
# Policy action helper
# ------------------------------------------------------------

def get_policy_action(policy, obs: np.ndarray, env, deterministic: bool = True, device=None) -> np.ndarray:
    """
    Converts a policy into an environment action.

    policy can be:
    - None: random action
    - callable: policy(obs) -> action
    - PyTorch nn.Module: actor(obs_tensor) -> action_tensor
    """
    if policy is None:
        return env.action_space.sample()

    is_torch_module = hasattr(policy, "parameters") and hasattr(policy, "eval") and callable(policy)

    if is_torch_module:
        import torch

        policy.eval()

        if device is None:
            try:
                device = next(policy.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t = policy(obs_t)

        action = action_t.detach().cpu().numpy()[0]

    elif callable(policy):
        action = policy(obs)

    else:
        raise TypeError("policy must be None, a callable, or a PyTorch module.")

    action = np.asarray(action, dtype=np.float32)

    if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
        action = np.clip(action, env.action_space.low, env.action_space.high)

    return action.astype(np.float32)


# ------------------------------------------------------------
# MuJoCo state extraction without rendering
# ------------------------------------------------------------

def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _get_qpos(env) -> np.ndarray:
    unwrapped = _unwrap_env(env)
    if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
        return np.asarray(unwrapped.data.qpos).copy()
    raise RuntimeError("Could not access env.unwrapped.data.qpos.")


def _get_body_xy(env, body_name: str) -> Optional[np.ndarray]:
    unwrapped = _unwrap_env(env)

    try:
        return np.asarray(unwrapped.data.body(body_name).xpos[:2], dtype=np.float32)
    except Exception:
        pass

    try:
        return np.asarray(unwrapped.data.get_body_xpos(body_name)[:2], dtype=np.float32)
    except Exception:
        pass

    return None


def _get_target_xy(env, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if isinstance(info, dict) and "target_x" in info and "target_y" in info:
        return np.array([info["target_x"], info["target_y"]], dtype=np.float32)

    if hasattr(env, "target_position"):
        return np.asarray(env.target_position(), dtype=np.float32)

    qpos = _get_qpos(env)
    if qpos.shape[0] >= 4:
        return np.asarray(qpos[2:4], dtype=np.float32)

    raise RuntimeError("Could not infer target position.")


def _get_fingertip_xy(env, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if isinstance(info, dict) and "fingertip_x" in info and "fingertip_y" in info:
        return np.array([info["fingertip_x"], info["fingertip_y"]], dtype=np.float32)

    fingertip = _get_body_xy(env, "fingertip")
    if fingertip is not None:
        return fingertip

    qpos = _get_qpos(env)
    theta1 = float(qpos[0])
    theta2 = float(qpos[1])
    l1, l2 = 0.1, 0.11
    elbow = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)], dtype=np.float32)
    tip = elbow + np.array(
        [l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)],
        dtype=np.float32,
    )
    return tip


def _get_link_points(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = _get_qpos(env)

    theta1 = float(qpos[0])
    theta2 = float(qpos[1])

    l1 = 0.1
    l2 = 0.11

    shoulder = np.array([0.0, 0.0], dtype=np.float32)

    elbow = shoulder + np.array(
        [l1 * np.cos(theta1), l1 * np.sin(theta1)],
        dtype=np.float32,
    )

    fingertip = elbow + np.array(
        [l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)],
        dtype=np.float32,
    )

    true_tip = _get_body_xy(env, "fingertip")
    if true_tip is not None:
        fingertip = true_tip.astype(np.float32)

    return shoulder, elbow, fingertip


# ------------------------------------------------------------
# Frame drawing
# ------------------------------------------------------------

def _draw_glow(ax, xy, color, radii=(0.055, 0.035, 0.018), alphas=(0.08, 0.18, 0.95), zorder=10):
    """
    Draw a soft glow using concentric circles.
    """
    for r, a in zip(radii, alphas):
        circ = Circle(xy, radius=r, facecolor=color, edgecolor="none", alpha=a, zorder=zorder)
        ax.add_patch(circ)


def _draw_trail(ax, points, color, max_width=5.0, min_width=1.0, max_alpha=0.75, zorder=4):
    """
    Draw a fading trail through a sequence of points.
    """
    if len(points) < 2:
        return

    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)

    for i in range(n - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        frac = (i + 1) / (n - 1)
        alpha = max_alpha * frac
        lw = min_width + (max_width - min_width) * frac
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )


def _draw_background(ax, xlim, ylim):
    """
    Draw a dark gradient background and a subtle arena.
    """
    nx, ny = 400, 400
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y)

    # radial gradient centered slightly off-origin
    R = np.sqrt((X / 0.45) ** 2 + (Y / 0.45) ** 2)
    G = np.clip(1.0 - R, 0.0, 1.0)

    # RGB gradient
    bg = np.zeros((ny, nx, 3), dtype=np.float32)
    bg[..., 0] = 0.05 + 0.04 * G
    bg[..., 1] = 0.06 + 0.05 * G
    bg[..., 2] = 0.08 + 0.08 * G

    ax.imshow(bg, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin="lower", zorder=0)

    # subtle arena ring
    arena = Circle((0.0, 0.0), radius=0.26, edgecolor=(1, 1, 1, 0.10), facecolor="none", linewidth=2.0, zorder=1)
    ax.add_patch(arena)

    # soft inner disc
    inner = Circle((0.0, 0.0), radius=0.23, edgecolor="none", facecolor=(1, 1, 1, 0.02), zorder=1)
    ax.add_patch(inner)

def _draw_info_panel(ax, step_idx, t, episode_return, distance, action=None):
    """
    Draw a readable dashboard panel in the corner.
    This version avoids text overlap by using fixed row spacing.
    """
    x0, y0 = 0.035, 0.64
    w, h = 0.36, 0.31

    panel = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.030",
        facecolor=(0.035, 0.050, 0.075, 0.88),
        edgecolor=(1, 1, 1, 0.16),
        linewidth=1.2,
        zorder=20,
    )
    ax.add_patch(panel)

    # Header
    ax.text(
        x0 + 0.025,
        y0 + h - 0.045,
        "MOVING REACHER",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        zorder=21,
    )

    # Subtle divider
    ax.plot(
        [x0 + 0.020, x0 + w - 0.020],
        [y0 + h - 0.070, y0 + h - 0.070],
        transform=ax.transAxes,
        color=(1, 1, 1, 0.18),
        linewidth=1.0,
        zorder=21,
    )

    # Rows
    row_y = [
        y0 + h - 0.105,
        y0 + h - 0.150,
        y0 + h - 0.195,
        y0 + h - 0.240,
        y0 + h - 0.285,
    ]

    label_x = x0 + 0.025
    value_x = x0 + 0.150

    rows = [
        ("step", f"{step_idx:>7d}", "#cfd8dc"),
        ("time", f"{t:>7.2f}", "#cfd8dc"),
        ("return", f"{episode_return:>7.2f}", "#ffe082"),
        ("dist", f"{distance:>7.3f}", "#80deea"),
    ]

    if action is not None:
        action = np.asarray(action).reshape(-1)
        if action.size == 1:
            action_txt = f"[{action[0]:+.2f}]"
        else:
            action_txt = "[" + ", ".join([f"{a:+.2f}" for a in action[:2]]) + "]"
        rows.append(("action", action_txt, "#ffcc80"))
    else:
        rows.append(("action", "   --", "#ffcc80"))

    for i, (label, value, color) in enumerate(rows):
        ax.text(
            label_x,
            row_y[i],
            label,
            transform=ax.transAxes,
            fontsize=9.5,
            color=(1, 1, 1, 0.55),
            zorder=21,
            family="monospace",
            ha="left",
            va="center",
        )
        ax.text(
            value_x,
            row_y[i],
            value,
            transform=ax.transAxes,
            fontsize=9.5,
            color=color,
            zorder=21,
            family="monospace",
            ha="left",
            va="center",
        )

def _draw_reacher_frame(
    env,
    info: Optional[Dict[str, Any]],
    step_idx: int,
    episode_return: float,
    target_trail: List[np.ndarray],
    fingertip_trail: List[np.ndarray],
    action: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-0.32, 0.32),
    ylim: Tuple[float, float] = (-0.32, 0.32),
    figsize: Tuple[float, float] = (6.4, 6.4),
    dpi: int = 120,
) -> np.ndarray:
    """
    Frame drawer that looks more like a visualization panel than a raw plot.
    """
    shoulder, elbow, fingertip = _get_link_points(env)
    target = _get_target_xy(env, info)

    if isinstance(info, dict) and "time" in info:
        t = float(info["time"])
    else:
        t = 0.0

    distance = float(np.linalg.norm(fingertip - target))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#081018")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#081018")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background
    _draw_background(ax, xlim, ylim)

    # Soft spotlight around target area
    _draw_glow(ax, target, color="#29b6f6", radii=(0.085, 0.055), alphas=(0.05, 0.08), zorder=2)

    # Trails
    _draw_trail(ax, target_trail, color="#29b6f6", max_width=6.0, min_width=1.5, max_alpha=0.70, zorder=3)
    _draw_trail(ax, fingertip_trail, color="#ffd54f", max_width=5.0, min_width=1.0, max_alpha=0.45, zorder=3)

    # Draw trail points as fading dots for target
    if len(target_trail) > 0:
        pts = np.asarray(target_trail)
        n = len(pts)
        for i, p in enumerate(pts):
            frac = (i + 1) / n
            ax.scatter(
                [p[0]], [p[1]],
                s=20 + 60 * frac,
                color="#29b6f6",
                alpha=0.05 + 0.35 * frac,
                zorder=4,
                edgecolors="none",
            )

    # Arm shadow
    ax.plot(
        [shoulder[0], elbow[0], fingertip[0]],
        [shoulder[1], elbow[1], fingertip[1]],
        color=(0, 0, 0, 0.45),
        linewidth=10,
        solid_capstyle="round",
        zorder=5,
    )

    # Arm main link
    ax.plot(
        [shoulder[0], elbow[0], fingertip[0]],
        [shoulder[1], elbow[1], fingertip[1]],
        color="#90caf9",
        linewidth=6.0,
        solid_capstyle="round",
        zorder=6,
    )

    # Arm highlight
    ax.plot(
        [shoulder[0], elbow[0], fingertip[0]],
        [shoulder[1], elbow[1], fingertip[1]],
        color="#e3f2fd",
        linewidth=2.2,
        alpha=0.9,
        solid_capstyle="round",
        zorder=7,
    )

    # Joints
    for p, s, c in [
        (shoulder, 120, "#eceff1"),
        (elbow, 100, "#b0bec5"),
    ]:
        _draw_glow(ax, p, color=c, radii=(0.028, 0.018), alphas=(0.12, 0.90), zorder=8)
        ax.scatter([p[0]], [p[1]], s=s, color=c, zorder=9, edgecolors="none")

    # Fingertip
    _draw_glow(ax, fingertip, color="#ffd54f", radii=(0.050, 0.030, 0.015), alphas=(0.10, 0.18, 1.00), zorder=10)
    ax.scatter([fingertip[0]], [fingertip[1]], s=130, color="#ffd54f", zorder=11, edgecolors="white", linewidths=0.8)

    # Current target
    _draw_glow(ax, target, color="#26c6da", radii=(0.060, 0.035, 0.018), alphas=(0.10, 0.25, 1.00), zorder=10)
    ax.scatter([target[0]], [target[1]], s=150, color="#26c6da", zorder=11, edgecolors="white", linewidths=0.8, marker="o")
    ax.scatter([target[0]], [target[1]], s=36, color="white", zorder=12, edgecolors="none")

    # Current distance link
    ax.plot(
        [fingertip[0], target[0]],
        [fingertip[1], target[1]],
        linestyle=(0, (4, 4)),
        linewidth=1.5,
        color=(1.0, 1.0, 1.0, 0.35),
        zorder=5,
    )

    # Small bottom title
    ax.text(
        0.50,
        0.04,
        "Target trail: last 20 steps",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=(1, 1, 1, 0.65),
        zorder=20,
    )

    # Info panel
    _draw_info_panel(
        ax,
        step_idx=step_idx,
        t=t,
        episode_return=episode_return,
        distance=distance,
        action=action,
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame


# ------------------------------------------------------------
# Video saving
# ------------------------------------------------------------

def save_frames_as_video(
    frames: List[np.ndarray],
    video_path: Union[str, os.PathLike],
    fps: int = 30,
) -> str:
    """
    Saves frames as .mp4 or .gif.
    If mp4 writing fails, falls back to gif.
    """
    if len(frames) == 0:
        raise ValueError("No frames to save.")

    import imageio.v2 as imageio

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = video_path.suffix.lower()

    if suffix == ".gif":
        imageio.mimsave(video_path, frames, fps=fps)
        return str(video_path)

    if suffix != ".mp4":
        raise ValueError("video_path must end in .mp4 or .gif.")

    try:
        imageio.mimsave(
            video_path,
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        )
        return str(video_path)
    except Exception:
        gif_path = video_path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        return str(gif_path)


# ------------------------------------------------------------
# Main rollout recorder
# ------------------------------------------------------------

def rollout_to_video(
    env,
    policy=None,
    video_path: Union[str, os.PathLike] = "videos/rollout.mp4",
    max_steps: int = 300,
    fps: int = 30,
    seed: Optional[int] = None,
    deterministic: bool = True,
    device=None,
    record_every: int = 1,
    trail_len: int = 20,
    xlim: Tuple[float, float] = (-0.32, 0.32),
    ylim: Tuple[float, float] = (-0.32, 0.32),
    figsize: Tuple[float, float] = (6.4, 6.4),
    dpi: int = 120,
) -> Tuple[str, Dict[str, Any]]:
    """
    Runs one rollout and produces a custom-rendered video.

    Does not call env.render().
    """
    obs, info = env.reset(seed=seed)

    frames = []
    rewards = []
    distances = []
    total_return = 0.0

    target_trail = deque(maxlen=trail_len)
    fingertip_trail = deque(maxlen=trail_len)

    # Initial state
    target_xy = _get_target_xy(env, info)
    fingertip_xy = _get_fingertip_xy(env, info)
    target_trail.append(target_xy.copy())
    fingertip_trail.append(fingertip_xy.copy())

    frames.append(
        _draw_reacher_frame(
            env=env,
            info=info,
            step_idx=0,
            episode_return=total_return,
            target_trail=list(target_trail),
            fingertip_trail=list(fingertip_trail),
            action=None,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            dpi=dpi,
        )
    )

    for step in range(max_steps):
        action = get_policy_action(
            policy=policy,
            obs=obs,
            env=env,
            deterministic=deterministic,
            device=device,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        reward = float(reward)
        total_return += reward
        rewards.append(reward)

        target_xy = _get_target_xy(env, info)
        fingertip_xy = _get_fingertip_xy(env, info)

        target_trail.append(target_xy.copy())
        fingertip_trail.append(fingertip_xy.copy())

        dist = float(np.linalg.norm(fingertip_xy - target_xy))
        distances.append(dist)

        if step % record_every == 0:
            frames.append(
                _draw_reacher_frame(
                    env=env,
                    info=info,
                    step_idx=step + 1,
                    episode_return=total_return,
                    target_trail=list(target_trail),
                    fingertip_trail=list(fingertip_trail),
                    action=action,
                    xlim=xlim,
                    ylim=ylim,
                    figsize=figsize,
                    dpi=dpi,
                )
            )

        if terminated or truncated:
            break

    saved_path = save_frames_as_video(frames, video_path=video_path, fps=fps)

    stats = {
        "return": total_return,
        "num_steps": step + 1,
        "num_frames": len(frames),
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else None,
        "mean_distance": float(np.mean(distances)) if len(distances) > 0 else None,
        "final_distance": float(distances[-1]) if len(distances) > 0 else None,
    }

    return saved_path, stats


# ------------------------------------------------------------
# Notebook embedding
# ------------------------------------------------------------

def embed_video(video_path, width: int = 700, height: Optional[int] = None):
    """
    Embed .mp4 or .gif in a notebook.
    """
    from IPython.display import HTML

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    suffix = video_path.suffix.lower()
    data = video_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")

    if suffix == ".gif":
        return HTML(f'<img src="data:image/gif;base64,{b64}" width="{width}">')

    if suffix == ".mp4":
        height_attr = f'height="{height}"' if height is not None else ""
        return HTML(f"""
        <video width="{width}" {height_attr} controls>
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """)

    raise ValueError("Only .mp4 and .gif are supported.")