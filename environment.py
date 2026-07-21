from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple


Action = Tuple[float, float, bool]


@dataclass(frozen=True)
class ArmState:
    shoulder_angle: float
    elbow_angle: float
    shoulder_velocity: float
    elbow_velocity: float
    holding_ball: bool
    step_count: int


@dataclass
class StepResult:
    state: ArmState
    reward: float
    done: bool
    info: Dict[str, object]


@dataclass
class EpisodeTrace:
    joint_positions: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]
    ball_path: List[Tuple[float, float]]
    landing_point: Optional[Tuple[float, float]]
    landing_error: Optional[float]
    total_reward: float
    success: bool
    released: bool


class ThrowingArmEnvironment:
    """Small 2D throwing arm environment with simple kinematics."""

    shoulder_limits = (-math.radians(40), math.radians(150))
    elbow_limits = (-math.radians(30), math.radians(90))

    def __init__(
        self,
        target_distance: float = 3.0,
        max_steps: int = 28,
        dt: float = 0.08,
        link_lengths: Tuple[float, float] = (1.2, 1.0),
        gravity: float = 9.81,
        success_radius: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.target_distance = float(target_distance)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.link_lengths = link_lengths
        self.gravity = float(gravity)
        self.success_radius = float(success_radius)
        self.rng = random.Random(seed)
        self.actions: List[Action] = [
            (-1.0, 0.0, False),
            (1.0, 0.0, False),
            (0.0, -1.0, False),
            (0.0, 1.0, False),
            (-1.0, -1.0, False),
            (-1.0, 1.0, False),
            (1.0, -1.0, False),
            (1.0, 1.0, False),
            (0.0, 0.0, False),
            (0.0, 0.0, True),
        ]
        self.accel = math.radians(70)
        self.damping = 0.92
        self.max_velocity = math.radians(320)
        self.min_velocity = -self.max_velocity
        self.max_joint_velocity = 0.0
        self.min_release_step = 4
        self.reset()

    def reset(self) -> ArmState:
        self.state = ArmState(
            shoulder_angle=math.radians(120),
            elbow_angle=math.radians(30),
            shoulder_velocity=0.0,
            elbow_velocity=0.0,
            holding_ball=True,
            step_count=0,
        )
        self.done = False
        self.last_landing_point: Optional[Tuple[float, float]] = None
        self.last_ball_path: List[Tuple[float, float]] = [self.end_effector(self.state)]
        self.trace_joints: List[
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        ] = [self.joint_positions(self.state)]
        return self.state

    def get_possible_actions(self, state: Optional[ArmState] = None) -> List[Action]:
        state = state or self.state
        if self.done or not state.holding_ball:
            return []
        return list(self.actions)

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(self.state, 0.0, True, self._info())
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")

        shoulder_input, elbow_input, requested_release = action
        forced_release = (not requested_release) and self.state.step_count + 1 >= self.max_steps
        release = requested_release or forced_release
        old_tip = self.end_effector(self.state)

        shoulder_velocity = self._clip(
            (self.state.shoulder_velocity + shoulder_input * self.accel) * self.damping,
            self.min_velocity,
            self.max_joint_velocity,
        )
        elbow_velocity = self._clip(
            (self.state.elbow_velocity + elbow_input * self.accel) * self.damping,
            self.min_velocity,
            self.max_joint_velocity,
        )
        shoulder_angle = self._clip(
            self.state.shoulder_angle + shoulder_velocity * self.dt,
            *self.shoulder_limits,
        )
        elbow_angle = self._clip(
            self.state.elbow_angle + elbow_velocity * self.dt,
            *self.elbow_limits,
        )
        next_state = ArmState(
            shoulder_angle=shoulder_angle,
            elbow_angle=elbow_angle,
            shoulder_velocity=shoulder_velocity,
            elbow_velocity=elbow_velocity,
            holding_ball=not release,
            step_count=self.state.step_count + 1,
        )
        self.state = next_state
        new_tip = self.end_effector(next_state)
        self.trace_joints.append(self.joint_positions(next_state))

        control_penalty = -0.01 * (abs(shoulder_input) + abs(elbow_input))
        reward = control_penalty - 0.002
        info = self._info()

        if release:
            ball_velocity = ((new_tip[0] - old_tip[0]) / self.dt, (new_tip[1] - old_tip[1]) / self.dt)
            landing, path = self.projectile_path(new_tip, ball_velocity)
            error = abs(landing[0] - self.target_distance)
            early_penalty = max(0, self.min_release_step - next_state.step_count) * 1.5
            forward_speed_bonus = max(0.0, ball_velocity[0]) * 0.35
            forced_penalty = 5.0 if forced_release else 0.0
            closeness_reward = 22.0 * math.exp(-((error / 0.75) ** 2))
            reward += closeness_reward - 0.8 * error + forward_speed_bonus - early_penalty - forced_penalty
            self.last_landing_point = landing
            self.last_ball_path = path
            self.done = True
            info = self._info()
        else:
            immediate_velocity = ((new_tip[0] - old_tip[0]) / self.dt, (new_tip[1] - old_tip[1]) / self.dt)
            predicted_landing, _ = self.projectile_path(new_tip, immediate_velocity, samples=6)
            predicted_error = abs(predicted_landing[0] - self.target_distance)
            predicted_closeness = math.exp(-((predicted_error / 1.25) ** 2))
            forward_speed_hint = max(0.0, immediate_velocity[0]) * 0.015
            reward += 0.12 * predicted_closeness + forward_speed_hint

        return StepResult(next_state, reward, self.done, info)

    def end_effector(self, state: Optional[ArmState] = None) -> Tuple[float, float]:
        state = state or self.state
        shoulder = state.shoulder_angle
        elbow_total = state.shoulder_angle + state.elbow_angle
        x = self.link_lengths[0] * math.cos(shoulder) + self.link_lengths[1] * math.cos(elbow_total)
        y = self.link_lengths[0] * math.sin(shoulder) + self.link_lengths[1] * math.sin(elbow_total)
        return x, max(0.0, y)

    def joint_positions(
        self, state: Optional[ArmState] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        state = state or self.state
        base = (0.0, 0.0)
        elbow = (
            self.link_lengths[0] * math.cos(state.shoulder_angle),
            self.link_lengths[0] * math.sin(state.shoulder_angle),
        )
        tip = self.end_effector(state)
        return base, elbow, tip

    def projectile_path(
        self,
        start: Tuple[float, float],
        velocity: Tuple[float, float],
        samples: int = 40,
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
        x0, y0 = start
        vx, vy = velocity
        disc = max(0.0, vy * vy + 2.0 * self.gravity * y0)
        flight_time = max(0.0, (vy + math.sqrt(disc)) / self.gravity)
        path = []
        for i in range(samples + 1):
            t = flight_time * i / samples if samples else flight_time
            x = x0 + vx * t
            y = max(0.0, y0 + vy * t - 0.5 * self.gravity * t * t)
            path.append((x, y))
        landing = (x0 + vx * flight_time, 0.0)
        return landing, path

    def normalized_features(self, state: ArmState, action: Action) -> Dict[str, float]:
        tip = self.end_effector(state)
        tip_velocity = self.estimate_tip_velocity(state, action)
        predicted_landing, _ = self.projectile_path(tip, tip_velocity, samples=6)
        predicted_error = abs(predicted_landing[0] - self.target_distance)
        predicted_delta = self.target_distance - predicted_landing[0]
        shoulder_mid = sum(self.shoulder_limits) / 2.0
        elbow_mid = sum(self.elbow_limits) / 2.0
        shoulder_span = (self.shoulder_limits[1] - self.shoulder_limits[0]) / 2.0
        elbow_span = (self.elbow_limits[1] - self.elbow_limits[0]) / 2.0
        action_shoulder, action_elbow, release = action
        relative_target = (self.target_distance - tip[0]) / max(1.0, self.target_distance)
        target_scale = self._clip((self.target_distance - 3.0) / 5.0, -1.0, 1.0)
        action_name = self.action_name(action)
        base_features = {
            "bias": 1.0,
            "shoulder_angle": (state.shoulder_angle - shoulder_mid) / shoulder_span,
            "elbow_angle": (state.elbow_angle - elbow_mid) / elbow_span,
            "shoulder_velocity": state.shoulder_velocity / self.max_velocity,
            "elbow_velocity": state.elbow_velocity / self.max_velocity,
            "holding_ball": 1.0 if state.holding_ball else 0.0,
            "target_distance": self._clip(self.target_distance / 8.0, 0.0, 1.0),
            "target_scale": target_scale,
            "relative_target": self._clip(relative_target, -2.0, 2.0) / 2.0,
            "tip_height": self._clip(tip[1] / sum(self.link_lengths), 0.0, 1.0),
            "tip_vx": self._clip(tip_velocity[0] / 6.0, -1.0, 1.0),
            "tip_vy": self._clip(tip_velocity[1] / 6.0, -1.0, 1.0),
            "target_x_tip_vx": target_scale * self._clip(tip_velocity[0] / 6.0, -1.0, 1.0),
            "predicted_landing_error": self._clip(predicted_error / max(1.0, self.target_distance), 0.0, 2.0) / 2.0,
            "predicted_landing_delta": self._clip(predicted_delta / max(1.0, self.target_distance), -2.0, 2.0) / 2.0,
            "predicted_landing_closeness": 1.0 / (1.0 + predicted_error),
            "step_progress": state.step_count / max(1, self.max_steps),
            "action_shoulder": action_shoulder,
            "action_elbow": action_elbow,
            "release": 1.0 if release else 0.0,
        }
        features = dict(base_features)
        for name, value in base_features.items():
            if name not in {"bias", "action_shoulder", "action_elbow"}:
                features[f"{action_name}:{name}"] = value
        features[f"action={action_name}"] = 1.0
        return features

    def estimate_tip_velocity(self, state: ArmState, action: Action) -> Tuple[float, float]:
        shoulder_input, elbow_input, _ = action
        old_tip = self.end_effector(state)
        shoulder_velocity = self._clip(
            (state.shoulder_velocity + shoulder_input * self.accel) * self.damping,
            self.min_velocity,
            self.max_joint_velocity,
        )
        elbow_velocity = self._clip(
            (state.elbow_velocity + elbow_input * self.accel) * self.damping,
            self.min_velocity,
            self.max_joint_velocity,
        )
        next_state = ArmState(
            shoulder_angle=self._clip(state.shoulder_angle + shoulder_velocity * self.dt, *self.shoulder_limits),
            elbow_angle=self._clip(state.elbow_angle + elbow_velocity * self.dt, *self.elbow_limits),
            shoulder_velocity=shoulder_velocity,
            elbow_velocity=elbow_velocity,
            holding_ball=state.holding_ball,
            step_count=state.step_count + 1,
        )
        new_tip = self.end_effector(next_state)
        return (new_tip[0] - old_tip[0]) / self.dt, (new_tip[1] - old_tip[1]) / self.dt

    @staticmethod
    def action_name(action: Action) -> str:
        shoulder_input, elbow_input, release = action
        if release:
            return "release"
        if shoulder_input == 0.0 and elbow_input == 0.0:
            return "coast"
        parts = []
        if shoulder_input > 0.0:
            parts.append("shoulder+")
        elif shoulder_input < 0.0:
            parts.append("shoulder-")
        if elbow_input > 0.0:
            parts.append("elbow+")
        elif elbow_input < 0.0:
            parts.append("elbow-")
        return " ".join(parts)

    def episode_trace(self, total_reward: float) -> EpisodeTrace:
        landing_error = None
        success = False
        if self.last_landing_point is not None:
            landing_error = abs(self.last_landing_point[0] - self.target_distance)
            success = landing_error <= self.success_radius
        return EpisodeTrace(
            joint_positions=self.trace_joints,
            ball_path=self.last_ball_path,
            landing_point=self.last_landing_point,
            landing_error=landing_error,
            total_reward=total_reward,
            success=success,
            released=self.last_landing_point is not None,
        )

    def _info(self) -> Dict[str, object]:
        error = None
        if self.last_landing_point is not None:
            error = abs(self.last_landing_point[0] - self.target_distance)
        return {
            "landing_point": self.last_landing_point,
            "landing_error": error,
            "ball_path": self.last_ball_path,
            "joint_positions": self.trace_joints,
            "target_distance": self.target_distance,
        }

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
