from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from agent import ApproximateQAgent
from environment import Action
from environment import EpisodeTrace, ThrowingArmEnvironment


PULL_ACTION: Action = (-1.0, -1.0, False)
COAST_ACTION: Action = (0.0, 0.0, False)
BRAKE_ACTION: Action = (1.0, 1.0, False)
RELEASE_ACTION: Action = (0.0, 0.0, True)


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    landing_error: Optional[float]
    success: bool
    released: bool


@dataclass
class MotionFrame:
    step: int
    joints: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ball_position: Optional[Tuple[float, float]]
    released: bool


@dataclass
class MotionResult:
    stats: EpisodeStats
    trace: EpisodeTrace
    frames: List[MotionFrame]
    actions: List[str]


@dataclass(frozen=True)
class MotionPlan:
    pull_steps: int
    coast_steps: int
    brake_steps: int
    finish_pull_steps: int = 0

    @property
    def release_step(self) -> int:
        return self.pull_steps + self.coast_steps + self.brake_steps + self.finish_pull_steps + 1

    def actions(self) -> List[Action]:
        return (
            [PULL_ACTION] * self.pull_steps
            + [COAST_ACTION] * self.coast_steps
            + [BRAKE_ACTION] * self.brake_steps
            + [PULL_ACTION] * self.finish_pull_steps
            + [RELEASE_ACTION]
        )


def make_env_and_agent(
    target_distance: float,
    epsilon: float,
    learning_rate: float,
    discount: float,
    seed: int,
) -> Tuple[ThrowingArmEnvironment, ApproximateQAgent]:
    env = ThrowingArmEnvironment(target_distance=target_distance, seed=seed)
    agent = ApproximateQAgent(
        actions_fn=env.get_possible_actions,
        features_fn=env.normalized_features,
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed,
    )
    return env, agent


def run_episode(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    episode: int = 1,
    train: bool = True,
    greedy: bool = False,
) -> Tuple[EpisodeStats, EpisodeTrace]:
    state = env.reset()
    total_reward = 0.0
    while True:
        action = agent.choose_action(state, greedy=greedy)
        if action is None:
            break
        result = env.step(action)
        if train:
            agent.update(state, action, result.state, result.reward)
        total_reward += result.reward
        state = result.state
        if result.done:
            break
    trace = env.episode_trace(total_reward)
    stats = EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return stats, trace


def train_agent(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    episodes: int,
) -> Tuple[List[EpisodeStats], EpisodeTrace]:
    return train_plan_agent(env, agent, episodes)


def candidate_motion_plans(env: ThrowingArmEnvironment) -> List[MotionPlan]:
    plans: List[MotionPlan] = []
    for pull_steps in range(1, 6):
        for coast_steps in range(0, 8):
            for brake_steps in range(0, 6):
                for finish_pull_steps in range(0, 4):
                    plan = MotionPlan(pull_steps, coast_steps, brake_steps, finish_pull_steps)
                    if plan.release_step <= env.max_steps:
                        plans.append(plan)
    return plans


def clone_env(env: ThrowingArmEnvironment) -> ThrowingArmEnvironment:
    return ThrowingArmEnvironment(
        target_distance=env.target_distance,
        max_steps=env.max_steps,
        dt=env.dt,
        link_lengths=env.link_lengths,
        gravity=env.gravity,
        success_radius=env.success_radius,
    )


def execute_motion_plan(
    env: ThrowingArmEnvironment,
    plan: MotionPlan,
    episode: int = 1,
) -> Tuple[EpisodeStats, EpisodeTrace, List[str]]:
    state = env.reset()
    total_reward = 0.0
    action_names: List[str] = []
    for action in plan.actions():
        result = env.step(action)
        action_name = env.action_name(action)
        if result.done and not action[2] and not result.state.holding_ball:
            action_name = f"{action_name} / forced release"
        action_names.append(action_name)
        total_reward += result.reward
        state = result.state
        if result.done:
            break
    trace = env.episode_trace(total_reward)
    stats = EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return stats, trace, action_names


def plan_rollout_features(env: ThrowingArmEnvironment, plan: MotionPlan) -> Dict[str, float]:
    stats, trace, _ = execute_motion_plan(clone_env(env), plan)
    landing_x = trace.landing_point[0] if trace.landing_point is not None else 0.0
    error = trace.landing_error if trace.landing_error is not None else env.target_distance
    signed_delta = (env.target_distance - landing_x) / max(1.0, env.target_distance)
    target_scale = max(-1.0, min(1.0, (env.target_distance - 3.0) / 5.0))
    release_progress = plan.release_step / max(1, env.max_steps)
    return {
        "plan:bias": 1.0,
        "plan:target_distance": max(0.0, min(1.0, env.target_distance / 8.0)),
        "plan:target_scale": target_scale,
        "plan:pull_steps": plan.pull_steps / 5.0,
        "plan:coast_steps": plan.coast_steps / 7.0,
        "plan:brake_steps": plan.brake_steps / 5.0,
        "plan:finish_pull_steps": plan.finish_pull_steps / 3.0,
        "plan:release_progress": release_progress,
        "plan:impulse": (plan.pull_steps + plan.finish_pull_steps - 0.5 * plan.brake_steps) / 8.0,
        "plan:predicted_landing_x": max(-1.0, min(1.0, landing_x / 8.0)),
        "plan:predicted_landing_error": max(0.0, min(1.0, error / max(1.0, env.target_distance))),
        "plan:predicted_landing_delta": max(-1.0, min(1.0, signed_delta)),
        "plan:predicted_landing_closeness": math.exp(-((error / 0.45) ** 2)),
        "plan:target_x_impulse": target_scale * (plan.pull_steps + plan.finish_pull_steps - 0.5 * plan.brake_steps) / 8.0,
        "plan:success_window": 1.0 if error <= 0.2 else 0.0,
    }


def plan_q_value(agent: ApproximateQAgent, features: Dict[str, float]) -> float:
    return sum(agent.weights.get(name, 0.0) * value for name, value in features.items())


def ensure_plan_priors(agent: ApproximateQAgent) -> None:
    if any(name.startswith("plan:") for name in agent.weights):
        return
    priors = {
        "plan:predicted_landing_closeness": 18.0,
        "plan:predicted_landing_error": -8.0,
        "plan:predicted_landing_delta": -0.5,
        "plan:success_window": 8.0,
        "plan:release_progress": -0.15,
        "plan:brake_steps": -0.1,
    }
    for name, value in priors.items():
        agent.weights[name] = agent.weights.get(name, 0.0) + value


def train_plan_agent(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    episodes: int,
) -> Tuple[List[EpisodeStats], EpisodeTrace]:
    ensure_plan_priors(agent)
    plans = candidate_motion_plans(env)
    feature_cache = {plan: plan_rollout_features(env, plan) for plan in plans}
    rollout_cache = {plan: execute_motion_plan(clone_env(env), plan) for plan in plans}
    history: List[EpisodeStats] = []
    last_trace: Optional[EpisodeTrace] = None
    for episode in range(1, episodes + 1):
        effective_epsilon = max(0.02, agent.epsilon * (0.996 ** (episode - 1)))
        if agent.rng.random() < effective_epsilon:
            plan = agent.rng.choice(plans)
        else:
            scored = [(plan_q_value(agent, feature_cache[item]), item) for item in plans]
            best_score = max(score for score, _ in scored)
            best_plans = [item for score, item in scored if score == best_score]
            plan = agent.rng.choice(best_plans)

        cached_stats, _, _ = rollout_cache[plan]
        features = feature_cache[plan]
        prediction = plan_q_value(agent, features)
        td_error = cached_stats.total_reward - prediction
        for name, value in features.items():
            agent.weights[name] = agent.weights.get(name, 0.0) + agent.learning_rate * td_error * value

        stats, trace, _ = execute_motion_plan(env, plan, episode=episode)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def best_motion_plan(env: ThrowingArmEnvironment, agent: ApproximateQAgent) -> MotionPlan:
    plans = candidate_motion_plans(env)
    scored = [(plan_q_value(agent, plan_rollout_features(env, plan)), plan) for plan in plans]
    best_score = max(score for score, _ in scored)
    best_plans = [plan for score, plan in scored if score == best_score]
    return agent.rng.choice(best_plans)


def demo_throw_actions() -> List[tuple[float, float, bool]]:
    return [
        (0.0, -1.0, False),
        (-1.0, 0.0, False),
        (0.0, 0.0, False),
        (-1.0, 0.0, False),
        (0.0, 0.0, False),
        (0.0, 0.0, True),
    ]


def apply_manual_priors(agent: ApproximateQAgent) -> None:
    priors = {
        "release:predicted_landing_closeness": 0.8,
        "release:predicted_landing_error": -0.35,
        "release:predicted_landing_delta": -0.15,
        "release:tip_vx": 0.25,
        "release:target_x_tip_vx": 0.25,
        "release:step_progress": 0.25,
        "action=release": -0.1,
        "coast:step_progress": -0.1,
    }
    for name, value in priors.items():
        agent.weights[name] = agent.weights.get(name, 0.0) + value


def pretrain_demo_policy(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    passes: int = 8,
    margin: float = 0.35,
    rate: float = 0.025,
) -> None:
    """Lightly bias the initial policy toward a throwing form.

    This is deliberately weak: the scripted demo is a form hint, while target
    distance and release timing are still learned from reward.
    """
    apply_manual_priors(agent)
    actions = demo_throw_actions()
    for _ in range(passes):
        state = env.reset()
        for demo_action in actions:
            legal_actions = list(env.get_possible_actions(state))
            if not legal_actions:
                break
            demo_q = agent.q_value(state, demo_action)
            for other_action in legal_actions:
                if other_action == demo_action:
                    continue
                gap = margin - (demo_q - agent.q_value(state, other_action))
                if gap <= 0.0:
                    continue
                for name, value in agent.features_fn(state, demo_action).items():
                    agent.weights[name] = agent.weights.get(name, 0.0) + rate * gap * value
                for name, value in agent.features_fn(state, other_action).items():
                    agent.weights[name] = agent.weights.get(name, 0.0) - rate * gap * value
                demo_q = agent.q_value(state, demo_action)
            result = env.step(demo_action)
            state = result.state
            if result.done:
                break


def run_scripted_episode(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    actions: List[tuple[float, float, bool]],
    episode: int,
    train: bool = True,
) -> Tuple[EpisodeStats, EpisodeTrace]:
    state = env.reset()
    total_reward = 0.0
    for action in actions:
        result = env.step(action)
        if train:
            agent.update(state, action, result.state, result.reward)
        total_reward += result.reward
        state = result.state
        if result.done:
            break
    trace = env.episode_trace(total_reward)
    stats = EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return stats, trace


def warm_start_agent(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    episodes: int,
    pretrain_policy: bool = True,
) -> Tuple[List[EpisodeStats], EpisodeTrace]:
    if pretrain_policy and not agent.weights:
        pretrain_demo_policy(env, agent)

    history: List[EpisodeStats] = []
    last_trace: Optional[EpisodeTrace] = None
    actions = demo_throw_actions()
    for episode in range(1, episodes + 1):
        stats, trace = run_scripted_episode(env, agent, actions, episode=episode, train=True)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def evaluate_greedy(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    episodes: int = 1,
) -> Tuple[List[EpisodeStats], EpisodeTrace]:
    if any(name.startswith("plan:") for name in agent.weights):
        history: List[EpisodeStats] = []
        last_trace: Optional[EpisodeTrace] = None
        plan = best_motion_plan(env, agent)
        for episode in range(1, episodes + 1):
            stats, trace, _ = execute_motion_plan(env, plan, episode=episode)
            history.append(stats)
            last_trace = trace
        assert last_trace is not None
        return history, last_trace

    history: List[EpisodeStats] = []
    last_trace: Optional[EpisodeTrace] = None
    for episode in range(1, episodes + 1):
        stats, trace = run_episode(env, agent, episode=episode, train=False, greedy=True)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def record_greedy_motion(env: ThrowingArmEnvironment, agent: ApproximateQAgent) -> MotionResult:
    if any(name.startswith("plan:") for name in agent.weights):
        plan = best_motion_plan(env, agent)
        return record_plan_motion(env, plan)

    state = env.reset()
    total_reward = 0.0
    actions: List[str] = []
    frames: List[MotionFrame] = [
        MotionFrame(
            step=state.step_count,
            joints=env.joint_positions(state),
            ball_position=env.end_effector(state),
            released=False,
        )
    ]

    while True:
        action = agent.choose_action(state, greedy=True)
        if action is None:
            break
        result = env.step(action)
        action_name = env.action_name(action)
        if result.done and not action[2] and not result.state.holding_ball:
            action_name = f"{action_name} / forced release"
        actions.append(action_name)
        total_reward += result.reward
        state = result.state
        released = bool(action[2])
        frames.append(
            MotionFrame(
                step=state.step_count,
                joints=env.joint_positions(state),
                ball_position=env.end_effector(state) if not released else env.end_effector(state),
                released=released,
            )
        )
        if result.done:
            break

    trace = env.episode_trace(total_reward)
    if trace.ball_path and trace.released:
        release_joints = frames[-1].joints if frames else env.joint_positions()
        for index, point in enumerate(trace.ball_path, start=1):
            frames.append(
                MotionFrame(
                    step=state.step_count + index,
                    joints=release_joints,
                    ball_position=point,
                    released=True,
                )
            )

    stats = EpisodeStats(
        episode=1,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return MotionResult(stats=stats, trace=trace, frames=frames, actions=actions)


def record_plan_motion(env: ThrowingArmEnvironment, plan: MotionPlan) -> MotionResult:
    state = env.reset()
    total_reward = 0.0
    action_names: List[str] = []
    frames: List[MotionFrame] = [
        MotionFrame(
            step=state.step_count,
            joints=env.joint_positions(state),
            ball_position=env.end_effector(state),
            released=False,
        )
    ]

    for action in plan.actions():
        result = env.step(action)
        action_name = env.action_name(action)
        if result.done and not action[2] and not result.state.holding_ball:
            action_name = f"{action_name} / forced release"
        action_names.append(action_name)
        total_reward += result.reward
        state = result.state
        released = bool(action[2])
        frames.append(
            MotionFrame(
                step=state.step_count,
                joints=env.joint_positions(state),
                ball_position=env.end_effector(state),
                released=released,
            )
        )
        if result.done:
            break

    trace = env.episode_trace(total_reward)
    if trace.ball_path and trace.released:
        release_joints = frames[-1].joints if frames else env.joint_positions()
        for index, point in enumerate(trace.ball_path, start=1):
            frames.append(
                MotionFrame(
                    step=state.step_count + index,
                    joints=release_joints,
                    ball_position=point,
                    released=True,
                )
            )

    stats = EpisodeStats(
        episode=1,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return MotionResult(stats=stats, trace=trace, frames=frames, actions=action_names)


def record_demo_motion(env: ThrowingArmEnvironment, agent: ApproximateQAgent) -> MotionResult:
    state = env.reset()
    total_reward = 0.0
    action_names: List[str] = []
    frames: List[MotionFrame] = [
        MotionFrame(
            step=state.step_count,
            joints=env.joint_positions(state),
            ball_position=env.end_effector(state),
            released=False,
        )
    ]

    for action in demo_throw_actions():
        result = env.step(action)
        action_name = env.action_name(action)
        if result.done and not action[2] and not result.state.holding_ball:
            action_name = f"{action_name} / forced release"
        action_names.append(action_name)
        total_reward += result.reward
        state = result.state
        released = bool(action[2])
        frames.append(
            MotionFrame(
                step=state.step_count,
                joints=env.joint_positions(state),
                ball_position=env.end_effector(state),
                released=released,
            )
        )
        if result.done:
            break

    trace = env.episode_trace(total_reward)
    if trace.ball_path and trace.released:
        release_joints = frames[-1].joints if frames else env.joint_positions()
        for index, point in enumerate(trace.ball_path, start=1):
            frames.append(
                MotionFrame(
                    step=state.step_count + index,
                    joints=release_joints,
                    ball_position=point,
                    released=True,
                )
            )

    stats = EpisodeStats(
        episode=1,
        total_reward=total_reward,
        landing_error=trace.landing_error,
        success=trace.success,
        released=trace.released,
    )
    return MotionResult(stats=stats, trace=trace, frames=frames, actions=action_names)
