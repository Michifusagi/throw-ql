from pathlib import Path

from training import (
    evaluate_greedy,
    make_env_and_agent,
    pretrain_demo_policy,
    record_demo_motion,
    record_greedy_motion,
    train_agent,
    warm_start_agent,
)


def test_training_updates_weights():
    env, agent = make_env_and_agent(
        target_distance=4.0,
        epsilon=0.3,
        learning_rate=0.2,
        discount=0.9,
        seed=7,
    )
    history, trace = train_agent(env, agent, episodes=10)

    assert len(history) == 10
    assert trace.released or history[-1].released is False
    assert agent.weights
    assert any(abs(value) > 0 for value in agent.weights.values())


def test_greedy_evaluation_does_not_change_weights():
    env, agent = make_env_and_agent(
        target_distance=4.0,
        epsilon=0.3,
        learning_rate=0.2,
        discount=0.9,
        seed=3,
    )
    train_agent(env, agent, episodes=5)
    before = dict(agent.weights)

    history, _ = evaluate_greedy(env, agent, episodes=2)

    assert len(history) == 2
    assert agent.weights == before


def test_record_greedy_motion_creates_frames_without_learning():
    env, agent = make_env_and_agent(
        target_distance=3.0,
        epsilon=0.35,
        learning_rate=0.15,
        discount=0.9,
        seed=9,
    )
    warm_start_agent(env, agent, episodes=30)
    train_agent(env, agent, episodes=50)
    before = dict(agent.weights)

    motion = record_greedy_motion(env, agent)

    assert motion.frames
    assert motion.actions
    assert motion.stats.released
    assert motion.trace.landing_point is not None
    assert agent.weights == before


def test_record_demo_motion_creates_scripted_frames_without_learning():
    env, agent = make_env_and_agent(
        target_distance=3.0,
        epsilon=0.35,
        learning_rate=0.15,
        discount=0.9,
        seed=4,
    )
    before = dict(agent.weights)

    motion = record_demo_motion(env, agent)

    assert motion.frames
    assert motion.actions[-1] == "release"
    assert motion.stats.released
    assert motion.trace.landing_error is not None
    assert agent.weights == before


def test_warm_start_updates_weights_with_scripted_throw():
    env, agent = make_env_and_agent(
        target_distance=3.0,
        epsilon=0.2,
        learning_rate=0.15,
        discount=0.9,
        seed=2,
    )

    history, trace = warm_start_agent(env, agent, episodes=3)

    assert len(history) == 3
    assert trace.released
    assert trace.landing_error is not None
    assert agent.weights
    assert any(key.startswith("release:") for key in agent.weights)


def test_demo_policy_pretraining_makes_greedy_release():
    env, agent = make_env_and_agent(
        target_distance=3.0,
        epsilon=0.35,
        learning_rate=0.15,
        discount=0.9,
        seed=0,
    )

    pretrain_demo_policy(env, agent)
    motion = record_greedy_motion(env, agent)

    assert motion.stats.released
    assert motion.actions[-1] == "release"
    assert motion.trace.landing_error is not None


def test_demo_anchor_after_exploration_keeps_greedy_throwing():
    env, agent = make_env_and_agent(
        target_distance=3.0,
        epsilon=0.35,
        learning_rate=0.15,
        discount=0.9,
        seed=0,
    )

    warm_start_agent(env, agent, episodes=100)
    train_agent(env, agent, episodes=50)
    pretrain_demo_policy(env, agent, passes=10, rate=0.02)
    motion = record_greedy_motion(env, agent)

    assert motion.stats.released
    assert motion.trace.landing_error is not None
    assert motion.trace.landing_error < 1.0


def test_weight_save_and_load(tmp_path: Path):
    env, agent = make_env_and_agent(
        target_distance=4.0,
        epsilon=0.0,
        learning_rate=0.2,
        discount=0.9,
        seed=5,
    )
    train_agent(env, agent, episodes=3)
    path = tmp_path / "weights.json"
    agent.save_weights(path)

    _, loaded = make_env_and_agent(4.0, 0.0, 0.2, 0.9, 5)
    loaded.load_weights(path)

    assert loaded.weights == agent.weights
