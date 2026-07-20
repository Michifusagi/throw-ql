from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent import ApproximateQAgent
from environment import EpisodeTrace, ThrowingArmEnvironment


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
    history: List[EpisodeStats] = []
    last_trace: Optional[EpisodeTrace] = None
    for episode in range(1, episodes + 1):
        stats, trace = run_episode(env, agent, episode=episode, train=True, greedy=False)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


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
        "release:predicted_landing_closeness": 2.0,
        "release:predicted_landing_error": -0.8,
        "release:tip_vx": 0.6,
        "release:step_progress": 0.7,
        "action=release": -0.25,
        "coast:step_progress": -0.25,
    }
    for name, value in priors.items():
        agent.weights[name] = agent.weights.get(name, 0.0) + value


def pretrain_demo_policy(
    env: ThrowingArmEnvironment,
    agent: ApproximateQAgent,
    passes: int = 40,
    margin: float = 1.0,
    rate: float = 0.08,
) -> None:
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
    history: List[EpisodeStats] = []
    last_trace: Optional[EpisodeTrace] = None
    for episode in range(1, episodes + 1):
        stats, trace = run_episode(env, agent, episode=episode, train=False, greedy=True)
        history.append(stats)
        last_trace = trace
    assert last_trace is not None
    return history, last_trace


def record_greedy_motion(env: ThrowingArmEnvironment, agent: ApproximateQAgent) -> MotionResult:
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
