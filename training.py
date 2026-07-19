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
