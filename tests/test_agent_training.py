from pathlib import Path

from training import evaluate_greedy, make_env_and_agent, train_agent


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
