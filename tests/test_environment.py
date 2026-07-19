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
