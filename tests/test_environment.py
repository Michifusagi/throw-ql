import math

from environment import ThrowingArmEnvironment


def test_release_ends_episode_and_reports_landing():
    env = ThrowingArmEnvironment(target_distance=4.0, seed=1)
    env.reset()
    result = env.step((0.0, 0.0, True))

    assert result.done
    assert result.info["landing_point"] is not None
    assert result.info["landing_error"] is not None
    assert result.state.holding_ball is False


def test_reset_uses_high_windup_initial_pose_and_updated_limits():
    env = ThrowingArmEnvironment(target_distance=3.0)
    state = env.reset()

    assert math.isclose(math.degrees(state.shoulder_angle), 120.0)
    assert math.isclose(math.degrees(state.elbow_angle), 30.0)
    assert tuple(round(math.degrees(value)) for value in env.shoulder_limits) == (-40, 150)
    assert tuple(round(math.degrees(value)) for value in env.elbow_limits) == (-30, 90)


def test_joint_velocity_is_never_positive_even_with_positive_acceleration():
    env = ThrowingArmEnvironment(target_distance=3.0)
    state = env.reset()

    result = env.step((1.0, 1.0, False))
    assert result.state.shoulder_velocity <= 0.0
    assert result.state.elbow_velocity <= 0.0

    env.state = state.__class__(
        shoulder_angle=state.shoulder_angle,
        elbow_angle=state.elbow_angle,
        shoulder_velocity=-env.accel,
        elbow_velocity=-env.accel,
        holding_ball=True,
        step_count=0,
    )
    result = env.step((1.0, 1.0, False))
    assert result.state.shoulder_velocity <= 0.0
    assert result.state.elbow_velocity <= 0.0


def test_projectile_lands_farther_with_positive_horizontal_velocity():
    env = ThrowingArmEnvironment(target_distance=4.0)
    landing, path = env.projectile_path((1.0, 1.0), (2.0, 1.0))

    assert landing[0] > 1.0
    assert landing[1] == 0.0
    assert path[-1][1] == 0.0


def test_features_are_normalized_and_include_action_terms():
    env = ThrowingArmEnvironment(target_distance=5.0)
    state = env.reset()
    features = env.normalized_features(state, (1.0, 0.0, False))

    assert features["bias"] == 1.0
    assert -1.0 <= features["shoulder_angle"] <= 1.0
    assert -1.0 <= features["elbow_angle"] <= 1.0
    assert math.isclose(features["action_shoulder"], 1.0)
    assert features["release"] == 0.0
    assert 0.0 <= features["target_distance"] <= 1.0
    assert -1.0 <= features["target_scale"] <= 1.0
    assert "shoulder+:relative_target" in features
    assert "shoulder+:tip_vx" in features
    assert "shoulder+:target_distance" in features


def test_release_features_include_predicted_landing_error():
    env = ThrowingArmEnvironment(target_distance=3.0)
    state = env.reset()
    features = env.normalized_features(state, (0.0, 0.0, True))

    assert "release:predicted_landing_error" in features
    assert "release:predicted_landing_delta" in features
    assert "release:predicted_landing_closeness" in features
    assert 0.0 <= features["release:predicted_landing_error"] <= 1.0
    assert -1.0 <= features["release:predicted_landing_delta"] <= 1.0
    assert 0.0 <= features["release:predicted_landing_closeness"] <= 1.0


def test_max_steps_forces_release_and_reports_landing():
    env = ThrowingArmEnvironment(target_distance=3.0, max_steps=2)
    env.reset()
    env.step((0.0, 0.0, False))
    result = env.step((0.0, 0.0, False))

    assert result.done
    assert result.state.holding_ball is False
    assert result.info["landing_point"] is not None
    assert result.info["landing_error"] is not None


def test_release_reward_is_continuous_and_higher_near_target():
    probe_env = ThrowingArmEnvironment(target_distance=3.0)
    probe_env.reset()
    probe_result = probe_env.step((0.0, 0.0, True))
    landing_x = probe_result.info["landing_point"][0]

    near_env = ThrowingArmEnvironment(target_distance=landing_x)
    near_env.reset()
    near_result = near_env.step((0.0, 0.0, True))

    far_env = ThrowingArmEnvironment(target_distance=landing_x + 2.0)
    far_env.reset()
    far_result = far_env.step((0.0, 0.0, True))

    assert abs(landing_x - near_env.target_distance) < abs(landing_x - far_env.target_distance)
    assert near_result.reward > far_result.reward


def test_non_release_reward_contains_small_predicted_landing_hint():
    probe_env = ThrowingArmEnvironment(target_distance=3.0)
    state = probe_env.reset()
    velocity = probe_env.estimate_tip_velocity(state, (0.0, -1.0, False))
    landing, _ = probe_env.projectile_path(probe_env.end_effector(state), velocity)

    near_env = ThrowingArmEnvironment(target_distance=landing[0])
    near_env.reset()
    near_result = near_env.step((0.0, -1.0, False))

    far_env = ThrowingArmEnvironment(target_distance=landing[0] + 3.0)
    far_env.reset()
    far_result = far_env.step((0.0, -1.0, False))

    assert near_result.done is False
    assert near_result.reward > far_result.reward
