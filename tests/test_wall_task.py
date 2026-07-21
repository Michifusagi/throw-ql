from wall_task import (
    WallRallyEnvironment,
    evaluate_wall_greedy,
    make_wall_env_and_agent,
    record_wall_motion,
    train_wall_agent,
)


def test_wall_ball_hits_moving_wall_and_bounces():
    env = WallRallyEnvironment(target_catches=1, seed=1)
    state = env.reset()

    saw_wall_return = False
    saw_ground_bounce = False
    saw_wall_move = False
    for _ in range(env.max_steps):
        result = env.step("track")
        saw_wall_return = saw_wall_return or result.state.ball_vx < 0.0
        saw_ground_bounce = saw_ground_bounce or result.state.bounced
        saw_wall_move = saw_wall_move or result.state.wall_x != state.wall_x
        if result.done:
            break

    assert saw_wall_return
    assert saw_ground_bounce
    assert saw_wall_move
    trace = env.episode_trace(0.0)
    assert trace.wall_path
    assert trace.missed or trace.success


def test_wall_features_include_ball_arm_and_action_terms():
    env = WallRallyEnvironment(target_catches=3, seed=2)
    state = env.reset()
    features = env.normalized_features(state, "track")

    assert features["bias"] == 1.0
    assert "ball_x" in features
    assert "wall_x" in features
    assert "wall_velocity" in features
    assert "relative_ball_wall_x" in features
    assert "relative_ball_y" in features
    assert "track:distance_to_ball" in features
    assert features["action_shoulder"] == 0.0


def test_wall_training_updates_weights_and_evaluation_is_read_only():
    env, agent = make_wall_env_and_agent(
        target_catches=1,
        epsilon=0.3,
        learning_rate=0.12,
        discount=0.95,
        seed=3,
    )

    history, trace = train_wall_agent(env, agent, episodes=5)
    before = dict(agent.weights)
    eval_history, _ = evaluate_wall_greedy(env, agent, episodes=1)

    assert len(history) == 5
    assert trace.target_catches == 1
    assert agent.weights
    assert eval_history[0].target_catches == 1
    assert agent.weights == before


def test_record_wall_motion_does_not_change_weights():
    env, agent = make_wall_env_and_agent(
        target_catches=1,
        epsilon=0.3,
        learning_rate=0.12,
        discount=0.95,
        seed=4,
    )
    train_wall_agent(env, agent, episodes=3)
    before = dict(agent.weights)

    motion = record_wall_motion(env, agent)

    assert motion.frames
    assert motion.actions
    assert motion.trace.target_catches == 1
    assert agent.weights == before
