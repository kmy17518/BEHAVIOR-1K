"""Stateful smoothing and rate limits for arm-pose and hand-joint commands."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    smoothing_alpha: float = 0.85
    max_arm_velocity: float = 2.0
    max_hand_velocity: float = 6.0
    max_arm_delta_per_tick: float | None = 0.10
    max_hand_delta_per_tick: float | None = 0.20
    log_clips: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1]")
        if self.max_arm_velocity <= 0 or self.max_hand_velocity <= 0:
            raise ValueError("maximum velocities must be positive")
        for name in ("max_arm_delta_per_tick", "max_hand_delta_per_tick"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")


class SafetyFilter:
    """Smooth and rate-limit a fixed-length command without changing its meaning."""

    def __init__(self, n_arm: int, n_hand: int, dt: float, config: SafetyConfig) -> None:
        if n_arm < 0 or n_hand < 0 or dt <= 0:
            raise ValueError("SafetyFilter dimensions must be non-negative and dt must be positive")
        self.config = config
        self.size = n_arm + n_hand
        self.dt = float(dt)
        self._previous: np.ndarray | None = None
        self._velocity_cap = np.concatenate(
            [np.full(n_arm, config.max_arm_velocity), np.full(n_hand, config.max_hand_velocity)]
        ) * self.dt
        self._delta_cap = np.concatenate(
            [
                np.full(n_arm, np.inf if config.max_arm_delta_per_tick is None else config.max_arm_delta_per_tick),
                np.full(n_hand, np.inf if config.max_hand_delta_per_tick is None else config.max_hand_delta_per_tick),
            ]
        )
        self._warned = False

    def reset(self, command: np.ndarray) -> None:
        self._previous = self._validate(command)

    def apply(self, command: np.ndarray, logger: logging.Logger | None = None) -> np.ndarray:
        target = self._validate(command)
        if not self.config.enabled or self._previous is None:
            self._previous = target
            return target.copy()
        alpha = self.config.smoothing_alpha
        target = alpha * target + (1.0 - alpha) * self._previous
        raw_delta = target - self._previous
        delta = np.clip(raw_delta, -self._velocity_cap, self._velocity_cap)
        delta = np.clip(delta, -self._delta_cap, self._delta_cap)
        if (
            logger is not None
            and self.config.log_clips
            and not self._warned
            and not np.allclose(delta, raw_delta)
        ):
            logger.warning("Teleoperation safety rate limit engaged")
            self._warned = True
        result = self._previous + delta
        self._previous = result
        return result.copy()

    def _validate(self, command: np.ndarray) -> np.ndarray:
        array = np.asarray(command, dtype=np.float64).reshape(-1).copy()
        if array.shape != (self.size,):
            raise ValueError(f"Expected command shape ({self.size},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Command contains a non-finite value")
        return array
