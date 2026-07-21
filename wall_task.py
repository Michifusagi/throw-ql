from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from agent import ApproximateQAgent


WallAction = str


@dataclass(frozen=True)
class WallState:
    shoulder_angle: float
    elbow_angle: float
    shoulder_velocity: float
    elbow_velocity: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    wall_x: float
    wall_velocity: float
    bounced: bool
    catches: int
    step_count: int
    target_catches: int


@dataclass
class WallStepResult:
    state: WallState
    reward: float
    done: bool
    info: Dict[str, object]


@dataclass
class WallTrace:
    joint_positions: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]
    ball_path: List[Tuple[float, float]]
    wall_path: List[Tuple[float, float]]
    catch_points: List[Tuple[float, float]]
    total_reward: float
    catches: int
    target_catches: int
    success: bool
    missed: bool


@dataclass
class WallEpisodeStats:
    episode: int
    total_reward: float
    catches: int
    target_catches: int
    success: bool
    missed: bool


@dataclass
class WallMotionFrame:
    step: int
    joints: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ball_position: Tuple[float, float]
    wall_x: float
    catches: int
    bounced: bool


@dataclass
class WallMotionResult:
    stats: WallEpisodeStats
    trace: WallTrace
    frames: List[WallMotionFrame]
    actions: List[str]


class WallRallyEnvironment:
    """Wall-hit task with a returning ball and catch-by-tip objective."""

    actions: List[WallAction] = ["track", "track-high", "track-low"]
    shoulder_limits = (math.radians(20), math.radians(150))
    elbow_limits = (math.radians(-125), math.radians(45))

    def __init__(
        self,
        target_catches: int = 3,
        seed: Optional[int] = None,
        dt: float = 0.05,
    ) -> None:
        self.target_catches = int(target_catches)
        self.dt = float(dt)
        self.link_lengths = (1.0, 0.8)
        self.wall_start_x = 3.6
        self.wall_x = self.wall_start_x
        self.wall_velocity = 0.22
        self.wall_min_x = 3.25
        self.wall_max_x = 3.95
        self.catch_x = 1.2
        self.catch_radius = 0.2
        self.gravity = 9.81
        self.wall_restitution = 0.84
        self.ground_restitution = 0.72
        self.ground_friction = 0.92
        self.throw_velocity = (5.2, 2.55)
        self.max_velocity = math.radians(230)
        self.accel = math.radians(130)
        self.damping = 0.86
        self.max_steps = 220
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> WallState:
        self.done = False
        self.missed = False
        self.catch_points: List[Tuple[float, float]] = []
        self.wall_x = self.wall_start_x
        self.wall_direction = 1.0 if self.rng.random() < 0.5 else -1.0
        self.trace_joints: List[
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        ] = []
        self.ball_path: List[Tuple[float, float]] = []
        self.wall_path: List[Tuple[float, float]] = []
        self.state = WallState(
            shoulder_angle=math.radians(72),
            elbow_angle=math.radians(-45),
            shoulder_velocity=0.0,
            elbow_velocity=0.0,
            ball_x=0.0,
            ball_y=0.0,
            ball_vx=0.0,
            ball_vy=0.0,
            wall_x=self.wall_x,
            wall_velocity=self.wall_velocity * self.wall_direction,
            bounced=False,
            catches=0,
            step_count=0,
            target_catches=self.target_catches,
        )
        self._throw_from_tip()
        return self.state

    def get_possible_actions(self, state: Optional[WallState] = None) -> Iterable[WallAction]:
        if self.done:
            return []
        return list(self.actions)

    def step(self, action: WallAction) -> WallStepResult:
        if self.done:
            return WallStepResult(self.state, 0.0, True, self._info())
        if action not in self.actions:
            raise ValueError(f"Unknown wall action: {action}")

        old_tip = self.end_effector(self.state)
        old_distance = self._ball_tip_distance(self.state)
        shoulder_input, elbow_input = self._action_inputs(action)
        if action.startswith("track"):
            target = self._tracking_target(action)
            desired_shoulder, desired_elbow = self.inverse_kinematics(target)
            shoulder_velocity = self._clip(
                (desired_shoulder - self.state.shoulder_angle) / self.dt,
                -self.max_velocity,
                self.max_velocity,
            )
            elbow_velocity = self._clip(
                (desired_elbow - self.state.elbow_angle) / self.dt,
                -self.max_velocity,
                self.max_velocity,
            )
        else:
            shoulder_velocity = self._clip(
                (self.state.shoulder_velocity + shoulder_input * self.accel) * self.damping,
                -self.max_velocity,
                self.max_velocity,
            )
            elbow_velocity = self._clip(
                (self.state.elbow_velocity + elbow_input * self.accel) * self.damping,
                -self.max_velocity,
                self.max_velocity,
            )
        shoulder_angle = self._clip(
            self.state.shoulder_angle + shoulder_velocity * self.dt,
            *self.shoulder_limits,
        )
        elbow_angle = self._clip(
            self.state.elbow_angle + elbow_velocity * self.dt,
            *self.elbow_limits,
        )

        ball_x = self.state.ball_x + self.state.ball_vx * self.dt
        ball_y = self.state.ball_y + self.state.ball_vy * self.dt - 0.5 * self.gravity * self.dt * self.dt
        ball_vx = self.state.ball_vx
        ball_vy = self.state.ball_vy - self.gravity * self.dt
        wall_x = self.state.wall_x + self.state.wall_velocity * self.dt
        wall_velocity = self.state.wall_velocity
        if wall_x >= self.wall_max_x:
            wall_x = self.wall_max_x - (wall_x - self.wall_max_x)
            wall_velocity = -abs(wall_velocity)
        elif wall_x <= self.wall_min_x:
            wall_x = self.wall_min_x + (self.wall_min_x - wall_x)
            wall_velocity = abs(wall_velocity)
        bounced = self.state.bounced

        if ball_x >= wall_x and ball_vx > wall_velocity:
            ball_x = wall_x - (ball_x - wall_x)
            relative_vx = ball_vx - wall_velocity
            ball_vx = wall_velocity - abs(relative_vx) * self.wall_restitution

        if ball_y <= 0.0 and ball_vy < 0.0:
            ball_y = -ball_y
            ball_vy = abs(ball_vy) * self.ground_restitution
            ball_vx *= self.ground_friction
            bounced = True

        next_state = WallState(
            shoulder_angle=shoulder_angle,
            elbow_angle=elbow_angle,
            shoulder_velocity=shoulder_velocity,
            elbow_velocity=elbow_velocity,
            ball_x=ball_x,
            ball_y=max(0.0, ball_y),
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            wall_x=wall_x,
            wall_velocity=wall_velocity,
            bounced=bounced,
            catches=self.state.catches,
            step_count=self.state.step_count + 1,
            target_catches=self.target_catches,
        )

        self.state = next_state
        self.trace_joints.append(self.joint_positions(next_state))
        self.ball_path.append((next_state.ball_x, next_state.ball_y))
        self.wall_path.append((next_state.wall_x, next_state.ball_y))

        new_distance = self._ball_tip_distance(next_state)
        control_penalty = -0.004 * (abs(shoulder_input) + abs(elbow_input))
        if action == "track":
            control_penalty -= 0.006
        reward = 0.035 * (old_distance - new_distance) + control_penalty - 0.002
        if self._in_catch_window(next_state):
            reward += max(-0.12, 0.25 * (self.catch_radius - new_distance) / self.catch_radius)

        if self._caught(next_state):
            reward += 20.0
            catches = next_state.catches + 1
            self.catch_points.append((next_state.ball_x, next_state.ball_y))
            if catches >= self.target_catches:
                self.done = True
                reward += 35.0
                self.state = self._replace_counts(next_state, catches)
            else:
                self.state = self._replace_counts(next_state, catches)
                self._throw_from_tip()
        elif self._missed(next_state):
            self.done = True
            self.missed = True
            remaining = self.target_catches - next_state.catches
            reward -= 18.0 + 2.0 * remaining

        return WallStepResult(self.state, reward, self.done, self._info())

    def normalized_features(self, state: WallState, action: WallAction) -> Dict[str, float]:
        tip = self.end_effector(state)
        rel_x = state.ball_x - tip[0]
        rel_y = state.ball_y - tip[1]
        distance = math.hypot(rel_x, rel_y)
        shoulder_input, elbow_input = self._action_inputs(action)
        progress = state.catches / max(1, state.target_catches)
        catch_window = 1.0 if self._in_catch_window(state) else 0.0
        base = {
            "bias": 1.0,
            "shoulder_angle": self._norm_range(state.shoulder_angle, *self.shoulder_limits),
            "elbow_angle": self._norm_range(state.elbow_angle, *self.elbow_limits),
            "shoulder_velocity": state.shoulder_velocity / self.max_velocity,
            "elbow_velocity": state.elbow_velocity / self.max_velocity,
            "ball_x": self._clip(state.ball_x / self.wall_x, -0.5, 1.2),
            "ball_y": self._clip(state.ball_y / 2.2, 0.0, 1.5),
            "ball_vx": self._clip(state.ball_vx / 6.0, -1.0, 1.0),
            "ball_vy": self._clip(state.ball_vy / 6.0, -1.0, 1.0),
            "wall_x": self._clip((state.wall_x - self.wall_min_x) / (self.wall_max_x - self.wall_min_x), 0.0, 1.0),
            "wall_velocity": self._clip(state.wall_velocity / 0.5, -1.0, 1.0),
            "relative_ball_wall_x": self._clip((state.wall_x - state.ball_x) / max(1.0, self.wall_max_x), -1.0, 1.0),
            "relative_ball_x": self._clip(rel_x / self.wall_x, -1.0, 1.0),
            "relative_ball_y": self._clip(rel_y / 2.2, -1.0, 1.0),
            "tip_x": self._clip(tip[0] / self.wall_x, -0.5, 1.0),
            "tip_y": self._clip(tip[1] / 2.2, 0.0, 1.0),
            "distance_to_ball": self._clip(distance / 2.5, 0.0, 1.0),
            "bounced": 1.0 if state.bounced else 0.0,
            "catch_window": catch_window,
            "rally_progress": progress,
            "time_progress": state.step_count / max(1, self.max_steps),
            "action_shoulder": shoulder_input,
            "action_elbow": elbow_input,
        }
        features = dict(base)
        for name, value in base.items():
            if name not in {"bias", "action_shoulder", "action_elbow"}:
                features[f"{action}:{name}"] = value
        features[f"action={action}"] = 1.0
        return features

    def episode_trace(self, total_reward: float) -> WallTrace:
        return WallTrace(
            joint_positions=self.trace_joints,
            ball_path=self.ball_path,
            wall_path=self.wall_path,
            catch_points=self.catch_points,
            total_reward=total_reward,
            catches=self.state.catches,
            target_catches=self.target_catches,
            success=self.state.catches >= self.target_catches,
            missed=self.missed,
        )

    def end_effector(self, state: Optional[WallState] = None) -> Tuple[float, float]:
        state = state or self.state
        elbow_total = state.shoulder_angle + state.elbow_angle
        x = self.link_lengths[0] * math.cos(state.shoulder_angle) + self.link_lengths[1] * math.cos(elbow_total)
        y = self.link_lengths[0] * math.sin(state.shoulder_angle) + self.link_lengths[1] * math.sin(elbow_total)
        return x, max(0.0, y)

    def joint_positions(
        self, state: Optional[WallState] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        state = state or self.state
        base = (0.0, 0.0)
        elbow = (
            self.link_lengths[0] * math.cos(state.shoulder_angle),
            self.link_lengths[0] * math.sin(state.shoulder_angle),
        )
        return base, elbow, self.end_effector(state)

    def _throw_from_tip(self) -> None:
        tip = self.end_effector(self.state)
        self.state = WallState(
            shoulder_angle=self.state.shoulder_angle,
            elbow_angle=self.state.elbow_angle,
            shoulder_velocity=0.0,
            elbow_velocity=0.0,
            ball_x=tip[0],
            ball_y=tip[1],
            ball_vx=self.throw_velocity[0],
            ball_vy=self.throw_velocity[1],
            wall_x=self.state.wall_x,
            wall_velocity=self.state.wall_velocity,
            bounced=False,
            catches=self.state.catches,
            step_count=self.state.step_count,
            target_catches=self.target_catches,
        )
        self.trace_joints.append(self.joint_positions(self.state))
        self.ball_path.append((self.state.ball_x, self.state.ball_y))
        self.wall_path.append((self.state.wall_x, self.state.ball_y))

    def _action_inputs(self, action: WallAction) -> Tuple[float, float]:
        return 0.0, 0.0

    def _tracking_target(self, action: WallAction) -> Tuple[float, float]:
        y_offset = 0.0
        if action == "track-high":
            y_offset = 0.16
        elif action == "track-low":
            y_offset = -0.16
        if self._in_catch_window(self.state):
            return self.state.ball_x, max(0.08, self.state.ball_y + y_offset)
        return 1.05, 0.85 + y_offset

    def inverse_kinematics(self, point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        l1, l2 = self.link_lengths
        radius = self._clip(math.hypot(x, y), 0.15, l1 + l2 - 1e-4)
        cos_elbow = self._clip((radius * radius - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
        elbow = -math.acos(cos_elbow)
        shoulder = math.atan2(y, x) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
        return (
            self._clip(shoulder, *self.shoulder_limits),
            self._clip(elbow, *self.elbow_limits),
        )

    def _ball_tip_distance(self, state: WallState) -> float:
        tip = self.end_effector(state)
        return math.hypot(state.ball_x - tip[0], state.ball_y - tip[1])

    def _in_catch_window(self, state: WallState) -> bool:
        return state.bounced and state.ball_vx < 0.0 and state.ball_x <= self.catch_x + 0.35

    def _caught(self, state: WallState) -> bool:
        return self._in_catch_window(state) and self._ball_tip_distance(state) <= self.catch_radius

    def _missed(self, state: WallState) -> bool:
        return state.step_count >= self.max_steps or (
            state.bounced and state.ball_vx < 0.0 and state.ball_x < -0.15
        )

    def _replace_counts(self, state: WallState, catches: int) -> WallState:
        return WallState(
            shoulder_angle=state.shoulder_angle,
            elbow_angle=state.elbow_angle,
            shoulder_velocity=state.shoulder_velocity,
            elbow_velocity=state.elbow_velocity,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            ball_vx=state.ball_vx,
            ball_vy=state.ball_vy,
            wall_x=state.wall_x,
            wall_velocity=state.wall_velocity,
            bounced=state.bounced,
            catches=catches,
            step_count=state.step_count,
            target_catches=self.target_catches,
        )

    def _info(self) -> Dict[str, object]:
        return {
            "catches": self.state.catches,
            "target_catches": self.target_catches,
            "success": self.state.catches >= self.target_catches,
            "missed": self.missed,
            "ball_path": self.ball_path,
            "wall_path": self.wall_path,
            "joint_positions": self.trace_joints,
            "catch_points": self.catch_points,
        }

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _norm_range(value: float, low: float, high: float) -> float:
        mid = (low + high) / 2.0
        span = (high - low) / 2.0
        return (value - mid) / span


def make_wall_env_and_agent(
    target_catches: int,
    epsilon: float,
    learning_rate: float,
    discount: float,
    seed: int,
) -> Tuple[WallRallyEnvironment, ApproximateQAgent]:
    env = WallRallyEnvironment(target_catches=target_catches, seed=seed)
    agent = ApproximateQAgent(
        actions_fn=env.get_possible_actions,
        features_fn=env.normalized_features,
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed,
    )
    seed_wall_priors(agent)
    return env, agent


def seed_wall_priors(agent: ApproximateQAgent) -> None:
    priors = {
        "action=track": 5.0,
        "action=track-high": 4.0,
        "action=track-low": 4.0,
        "track:distance_to_ball": -0.2,
        "track:catch_window": 2.0,
        "track-high:catch_window": 1.5,
        "track-low:catch_window": 1.5,
        "catch_window:distance_to_ball": -0.6,
    }
    for name, value in priors.items():
        agent.weights.setdefault(name, value)


def run_wall_episode(
    env: WallRallyEnvironment,
    agent: ApproximateQAgent,
    episode: int = 1,
    train: bool = True,
    greedy: bool = False,
) -> Tuple[WallEpisodeStats, WallTrace]:
    state = env.reset()
    total_reward = 0.0
    action = agent.choose_action(state, greedy=greedy)
    while True:
        if action is None:
            break
        result = env.step(action)
        if train:
            next_action = None if result.done else agent.choose_action(result.state, greedy=False)
            prediction = agent.q_value(state, action)
            next_value = 0.0 if next_action is None else agent.q_value(result.state, next_action)
            td_error = result.reward + agent.discount * next_value - prediction
            for name, value in agent.features_fn(state, action).items():
                agent.weights[name] = agent.weights.get(name, 0.0) + agent.learning_rate * td_error * value
            action = next_action
        else:
            action = None if result.done else agent.choose_action(result.state, greedy=greedy)
        total_reward += result.reward
        state = result.state
        if result.done:
            break
    trace = env.episode_trace(total_reward)
    stats = WallEpisodeStats(
        episode=episode,
        total_reward=total_reward,
        catches=trace.catches,
        target_catches=trace.target_catches,
        success=trace.success,
        missed=trace.missed,
    )
    return stats, trace


def train_wall_agent(
    env: WallRallyEnvironment,
    agent: ApproximateQAgent,
    episodes: int,
) -> Tuple[List[WallEpisodeStats], WallTrace]:
    history: List[WallEpisodeStats] = []
    last_trace: Optional[WallTrace] = None
    for episode in range(1, episodes + 1):
        stats, trace = run_wall_episode(env, agent, episode=episode, train=True, greedy=False)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def evaluate_wall_greedy(
    env: WallRallyEnvironment,
    agent: ApproximateQAgent,
    episodes: int = 1,
) -> Tuple[List[WallEpisodeStats], WallTrace]:
    history: List[WallEpisodeStats] = []
    last_trace: Optional[WallTrace] = None
    for episode in range(1, episodes + 1):
        stats, trace = run_wall_episode(env, agent, episode=episode, train=False, greedy=True)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def record_wall_motion(env: WallRallyEnvironment, agent: ApproximateQAgent) -> WallMotionResult:
    state = env.reset()
    total_reward = 0.0
    actions: List[str] = []
    frames: List[WallMotionFrame] = [
        WallMotionFrame(
            step=state.step_count,
            joints=env.joint_positions(state),
            ball_position=(state.ball_x, state.ball_y),
            wall_x=state.wall_x,
            catches=state.catches,
            bounced=state.bounced,
        )
    ]
    while True:
        action = agent.choose_action(state, greedy=True)
        if action is None:
            break
        result = env.step(action)
        actions.append(action)
        total_reward += result.reward
        state = result.state
        frames.append(
            WallMotionFrame(
                step=state.step_count,
                joints=env.joint_positions(state),
                ball_position=(state.ball_x, state.ball_y),
                wall_x=state.wall_x,
                catches=state.catches,
                bounced=state.bounced,
            )
        )
        if result.done:
            break
    trace = env.episode_trace(total_reward)
    stats = WallEpisodeStats(
        episode=1,
        total_reward=total_reward,
        catches=trace.catches,
        target_catches=trace.target_catches,
        success=trace.success,
        missed=trace.missed,
    )
    return WallMotionResult(stats=stats, trace=trace, frames=frames, actions=actions)


def wall_success_rate(history: List[WallEpisodeStats]) -> float:
    if not history:
        return 0.0
    window = history[-50:]
    return sum(1 for item in window if item.success) / len(window)
